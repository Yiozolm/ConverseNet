#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Half.h>
#include <c10/util/BFloat16.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <string>

#ifndef MIXED_FUSION_BLOCK_THREADS
#error "The checked loader must set MIXED_FUSION_BLOCK_THREADS"
#endif

namespace {
constexpr int kThreads = MIXED_FUSION_BLOCK_THREADS;
static_assert(kThreads == 128 || kThreads == 256 || kThreads == 512,
              "Only the predeclared block configurations are supported");
enum PaddingMode { Constant = 0, Replicate = 1, Reflect = 2, Circular = 3 };

template <typename Storage, int Mode>
__global__ void pad_cast_kernel(const Storage* __restrict__ input,
                                float* __restrict__ output, int total,
                                int height, int width, int output_plane,
                                int output_width, int padding) {
    // Unsigned launch arithmetic also keeps the final out-of-range threads
    // well-defined when total is close to INT_MAX. Cast only after the guard.
    const uint32_t linear = blockIdx.x * uint32_t(blockDim.x) + threadIdx.x;
    if (linear >= uint32_t(total)) return;
    const int index = static_cast<int>(linear);
    const int plane = index / output_plane;
    const int spatial = index - plane * output_plane;
    const int output_y = spatial / output_width;
    const int output_x = spatial - output_y * output_width;
    int y = output_y - padding;
    int x = output_x - padding;
    if constexpr (Mode == Constant) {
        if (y < 0 || y >= height || x < 0 || x >= width) {
            output[index] = 0.0f;
            return;
        }
    } else if constexpr (Mode == Replicate) {
        y = y < 0 ? 0 : (y >= height ? height - 1 : y);
        x = x < 0 ? 0 : (x >= width ? width - 1 : x);
    } else if constexpr (Mode == Reflect) {
        y = y < 0 ? -y : (y >= height ? 2 * height - 2 - y : y);
        x = x < 0 ? -x : (x >= width ? 2 * width - 2 - x : x);
    } else {
        y = y < 0 ? y + height : (y >= height ? y - height : y);
        x = x < 0 ? x + width : (x >= width ? x - width : x);
    }
    const int source = plane * (height * width) + y * width + x;
    // The only floating-point operation is lossless widening of the original
    // 16-bit value. No FP32 unpadded intermediate or low-precision arithmetic.
    output[index] = static_cast<float>(input[source]);
}

template <typename Storage>
void launch(const at::Tensor& input, at::Tensor& output, int padding,
            int mode, cudaStream_t stream) {
    const int total = static_cast<int>(output.numel());
    const int blocks = (total - 1) / kThreads + 1;
    const int height = static_cast<int>(input.size(2));
    const int width = static_cast<int>(input.size(3));
    const int output_width = static_cast<int>(output.size(3));
    const int output_plane = static_cast<int>(output.size(2) * output.size(3));
#define PAD_CAST_LAUNCH(Mode) \
    pad_cast_kernel<Storage, Mode><<<blocks, kThreads, 0, stream>>>( \
        input.data_ptr<Storage>(), output.data_ptr<float>(), total, height, width, \
        output_plane, output_width, padding)
    switch (mode) {
        case Constant: PAD_CAST_LAUNCH(Constant); break;
        case Replicate: PAD_CAST_LAUNCH(Replicate); break;
        case Reflect: PAD_CAST_LAUNCH(Reflect); break;
        case Circular: PAD_CAST_LAUNCH(Circular); break;
    }
#undef PAD_CAST_LAUNCH
}
} // namespace

at::Tensor mixed_pad_cast_cuda(const at::Tensor& x, int64_t padding, const std::string& mode) {
    TORCH_CHECK(x.is_cuda(), "pad_cast requires CUDA input");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16,
                "pad_cast requires FP16 or BF16 storage");
    TORCH_CHECK(x.layout() == c10::kStrided && x.dim() == 4 && x.numel() > 0,
                "pad_cast requires nonempty strided NCHW input");
    TORCH_CHECK(x.is_contiguous() && !x.is_neg() && !x.is_conj(),
                "pad_cast requires contiguous NCHW input without lazy flags");
    TORCH_CHECK(!(at::GradMode::is_enabled() && x.requires_grad()),
                "pad_cast is inference-only; differentiable GradMode must use the original path");
    TORCH_CHECK(!at::autocast::is_autocast_enabled(at::kCUDA), "pad_cast does not support autocast");
    TORCH_CHECK(padding > 0 && padding <= INT_MAX, "pad_cast requires positive padding within INT32");
    TORCH_CHECK(x.numel() <= INT_MAX, "pad_cast input exceeds the checked INT32 index domain");
    const int64_t height = x.size(2), width = x.size(3);
    TORCH_CHECK(padding <= (INT_MAX - height) / 2 && padding <= (INT_MAX - width) / 2,
                "pad_cast padded dimensions exceed INT32");
    int selected = -1;
    if (mode == "constant") selected = Constant;
    else if (mode == "replicate") selected = Replicate;
    else if (mode == "reflect") {
        TORCH_CHECK(padding < height && padding < width, "reflect padding must be smaller than both spatial dimensions");
        selected = Reflect;
    } else if (mode == "circular") {
        TORCH_CHECK(padding <= height && padding <= width, "circular padding may not wrap more than once");
        selected = Circular;
    }
    TORCH_CHECK(selected >= 0, "padding mode must be constant, replicate, reflect or circular");
    const int64_t output_height = height + 2 * padding, output_width = width + 2 * padding;
    TORCH_CHECK(output_height <= INT_MAX / output_width, "pad_cast output plane exceeds INT32");
    const int64_t output_plane = output_height * output_width;
    TORCH_CHECK(x.size(0) * x.size(1) <= INT_MAX / output_plane,
                "pad_cast output exceeds the checked INT32 index domain");
    const c10::cuda::CUDAGuard guard(x.device());
    // Do not change the caller's GradMode. Frozen GradMode inputs retain the
    // surrounding solver's existing dispatch; this new buffer has no VJP.
    auto output = at::empty({x.size(0), x.size(1), output_height, output_width}, x.options().dtype(at::kFloat));
    const auto stream = c10::cuda::getCurrentCUDAStream(x.get_device());
    if (x.scalar_type() == at::kHalf) launch<c10::Half>(x, output, static_cast<int>(padding), selected, stream);
    else launch<c10::BFloat16>(x, output, static_cast<int>(padding), selected, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
