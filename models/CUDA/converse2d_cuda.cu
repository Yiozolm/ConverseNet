#include <torch/extension.h>
#include <vector>
#include <cufft.h>


#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


torch::Tensor p2o_cuda(torch::Tensor psf, const std::vector<long>& shape) {
    auto otf = torch::zeros(torch::IntArrayRef({psf.size(0), psf.size(1), shape[0], shape[1]})).to(psf.device());
    otf.slice(2, 0, psf.size(2)).slice(3, 0, psf.size(3)).copy_(psf);
    
    otf = torch::roll(otf, {-psf.size(2) / 2, -psf.size(3) / 2}, {2, 3});
    return torch::fft::fftn(otf, c10::nullopt, c10::IntArrayRef({-2, -1}));
}

torch::Tensor splits_cuda(torch::Tensor a, int scale) {
    auto sizes = a.sizes();
    long W = sizes[2];
    long H = sizes[3];
    long W_s = W / scale;
    long H_s = H / scale;

    auto b = a.view({sizes[0], sizes[1], scale, W_s, scale, H_s});
    b = b.permute({0, 1, 3, 5, 2, 4}).contiguous();
    return b.view({sizes[0], sizes[1], W_s, H_s, scale * scale});
}

torch::Tensor converse2d_cuda_forward(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor bias,
    int scale,
    float eps
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "Weight tensor must be a CUDA tensor");
    TORCH_CHECK(bias.is_cuda(), "Bias tensor must be a CUDA tensor");

    x = x.contiguous();
    weight = weight.contiguous();
    bias = bias.contiguous();

    const int N = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int H_up = H * scale;
    const int W_up = W * scale;

    auto biaseps = (torch::sigmoid(bias - 9.0f) + eps).contiguous();

    auto STy = torch::zeros({N, C, H_up, W_up}, x.options());
    STy.slice(2, 0, H_up, scale).slice(3, 0, W_up, scale).copy_(x);

    if (scale != 1) {
        x = torch::nn::functional::interpolate(x,
            torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>({(double)scale, (double)scale})).mode(torch::kNearest));
    }

    auto FB = p2o_cuda(weight, {H_up, W_up}).contiguous();
    auto FBC = torch::conj(FB).contiguous();
    auto F2B = torch::pow(torch::abs(FB), 2).contiguous();

    auto STy_fft = torch::fft::fftn(STy, c10::nullopt, c10::IntArrayRef({-2, -1})).contiguous();
    auto FBFy = (FBC * STy_fft).contiguous();

    auto x_fft = torch::fft::fftn(biaseps * x, c10::nullopt, c10::IntArrayRef({-2, -1})).contiguous();

    auto FR = FBFy + x_fft;
    auto x1 = FB.mul(FR);
    
    auto FBR = torch::mean(splits_cuda(x1, scale), -1, false);
    auto invW = torch::mean(splits_cuda(F2B.to(torch::kComplexFloat), scale), -1, false);

    auto invWBR = FBR.div(invW + biaseps.to(torch::kComplexFloat));
    auto FCBinvWBR = FBC * invWBR.repeat({1, 1, scale, scale});

    auto FX = (FR - FCBinvWBR) / biaseps.to(torch::kComplexFloat);
    
    auto out_complex = torch::fft::ifftn(FX, c10::nullopt, c10::IntArrayRef({-2, -1}));
    auto out = torch::real(out_complex);

    return out;
}