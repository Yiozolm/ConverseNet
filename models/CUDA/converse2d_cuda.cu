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

inline torch::Tensor splits_cuda(torch::Tensor a, int scale) {
    auto sizes = a.sizes();
    long W = sizes[2];
    long H = sizes[3];
    long W_s = W / scale;
    long H_s = H / scale;

    auto b = a.view({sizes[0], sizes[1], scale, W_s, scale, H_s});
    b = b.permute({0, 1, 3, 5, 2, 4}).contiguous();
    return b.view({sizes[0], sizes[1], W_s, H_s, scale * scale});
}

inline torch::Tensor unsplit_cuda(torch::Tensor b, int scale) {
    auto sizes = b.sizes(); 
    long N = sizes[0];
    long C = sizes[1];
    long W_s = sizes[2];
    long H_s = sizes[3];
    
    auto a = b.view({N, C, W_s, H_s, scale, scale});
    a = a.permute({0, 1, 4, 2, 5, 3}).contiguous();
    return a.view({N, C, W_s * scale, H_s * scale});
}

inline torch::Tensor interpolate_backward_nearest_cuda(torch::Tensor grad_out, int scale) {
    if (scale == 1) return grad_out;
    auto options = torch::nn::functional::AvgPool2dFuncOptions({scale, scale}).stride({scale, scale});
    auto grad_in = torch::nn::functional::avg_pool2d(grad_out, options);
    return grad_in * (scale * scale);
}


std::vector<torch::Tensor> converse2d_cuda_forward(
    torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int scale, float eps
) {
    TORCH_CHECK(x.is_cuda(), "Input tensor must be a CUDA tensor");
    const int H_up = x.size(2) * scale;
    const int W_up = x.size(3) * scale;
    auto biaseps = (torch::sigmoid(bias - 9.0f) + eps).contiguous();
    auto STy = torch::zeros({x.size(0), x.size(1), H_up, W_up}, x.options());
    STy.slice(2, 0, H_up, scale).slice(3, 0, W_up, scale).copy_(x);
    auto x_interp = x;
    if (scale != 1) {
        x_interp = torch::nn::functional::interpolate(x,
            torch::nn::functional::InterpolateFuncOptions().scale_factor(std::vector<double>({(double)scale, (double)scale})).mode(torch::kNearest));
    }
    auto FB = p2o_cuda(weight, {H_up, W_up}).contiguous();
    auto FBC = torch::conj(FB).contiguous();
    auto STy_fft = torch::fft::fftn(STy, c10::nullopt, c10::IntArrayRef({-2, -1})).contiguous();
    auto x_fft = torch::fft::fftn(biaseps * x_interp, c10::nullopt, c10::IntArrayRef({-2, -1})).contiguous();
    auto FR = (FBC * STy_fft) + x_fft;
    auto invW = torch::mean(splits_cuda(torch::pow(torch::abs(FB), 2).to(torch::kComplexFloat), scale), -1, false);
    auto FBR = torch::mean(splits_cuda(FB.mul(FR), scale), -1, false);
    auto invWBR = FBR.div(invW + biaseps.to(torch::kComplexFloat));
    auto FCBinvWBR = FBC * invWBR.repeat({1, 1, scale, scale});
    auto FX = (FR - FCBinvWBR) / biaseps.to(torch::kComplexFloat);
    auto out = torch::real(torch::fft::ifftn(FX, c10::nullopt, c10::IntArrayRef({-2, -1})));

    return {out, x_interp, biaseps, FB, FBC, FR, invW, FBR, invWBR, STy_fft};
}

std::vector<torch::Tensor> converse2d_cuda_backward(
    torch::Tensor grad_out, torch::Tensor x, torch::Tensor weight, torch::Tensor bias, int scale,
    const std::vector<torch::Tensor>& saved_tensors
) {
    // --- Unpack saved tensors ---
    auto x_interp = saved_tensors[0];
    auto biaseps  = saved_tensors[1];
    auto FB       = saved_tensors[2];
    auto FBC      = saved_tensors[3];
    auto FR       = saved_tensors[4];
    auto invW     = saved_tensors[5];
    auto FBR      = saved_tensors[6];
    auto invWBR   = saved_tensors[7];
    auto STy_fft  = saved_tensors[8];


    auto grad_out_c = grad_out.to(torch::kComplexFloat);
    auto grad_FX = torch::fft::fftn(grad_out_c, c10::nullopt, c10::IntArrayRef({-2, -1}));
    auto FCBinvWBR = FBC * invWBR.repeat({1, 1, scale, scale});
    auto grad_FR = grad_FX / biaseps.to(torch::kComplexFloat);
    auto grad_FCBinvWBR = -grad_FR;
    auto grad_biaseps = -torch::sum(torch::real(grad_FX * (FR - FCBinvWBR) / torch::pow(biaseps.to(torch::kComplexFloat), 2)), {0, 2, 3}, true);
    auto grad_FBC = grad_FCBinvWBR * invWBR.repeat({1, 1, scale, scale});
    auto grad_invWBR = torch::sum(splits_cuda(grad_FCBinvWBR * FBC, scale), -1, false);
    auto denom = invW + biaseps.to(torch::kComplexFloat);
    auto grad_FBR = grad_invWBR / denom;
    auto grad_denom = -grad_invWBR * FBR / torch::pow(denom, 2);
    grad_biaseps += torch::sum(torch::real(grad_denom), {0, 2, 3}, true);
    auto F2B = torch::pow(torch::abs(FB), 2);
    auto grad_F2B = unsplit_cuda( (grad_denom).unsqueeze(-1).expand({-1, -1, -1, -1, scale * scale}) / (float)(scale * scale), scale).to(F2B.dtype());
    auto grad_x1 = unsplit_cuda( (grad_FBR).unsqueeze(-1).expand({-1, -1, -1, -1, scale * scale}) / (float)(scale * scale), scale);
    auto grad_FB = grad_x1 * torch::conj(FR);
    grad_FR += grad_x1 * torch::conj(FB);
    auto grad_x_fft = grad_FR;
    auto grad_biaseps_x_interp = torch::real(torch::fft::ifftn(grad_x_fft, c10::nullopt, c10::IntArrayRef({-2, -1})));
    grad_biaseps += torch::sum(grad_biaseps_x_interp * x_interp, {0, 2, 3}, true);
    auto grad_x_interp = grad_biaseps_x_interp * biaseps;
    grad_FBC += grad_FR * torch::conj(STy_fft);
    auto grad_STy_fft = grad_FR * torch::conj(FBC);
    auto grad_STy = torch::real(torch::fft::ifftn(grad_STy_fft, c10::nullopt, c10::IntArrayRef({-2, -1})));
    const int H_up = x.size(2) * scale; const int W_up = x.size(3) * scale;
    auto grad_x = grad_STy.slice(2, 0, H_up, scale).slice(3, 0, W_up, scale) + interpolate_backward_nearest_cuda(grad_x_interp, scale);
    grad_FB += 2 * F2B.to(torch::kComplexFloat) * FBC;
    grad_FB += torch::conj(grad_FBC);
    auto grad_otf = torch::real(torch::fft::ifftn(grad_FB, c10::nullopt, c10::IntArrayRef({-2, -1})));
    auto grad_weight = torch::roll(grad_otf, {weight.size(2) / 2, weight.size(3) / 2}, {2, 3}).slice(2, 0, weight.size(2)).slice(3, 0, weight.size(3)).clone();
    auto sig = torch::sigmoid(bias - 9.0f);
    auto grad_bias = grad_biaseps * sig * (1.0f - sig);

    return {grad_x, grad_weight, grad_bias};
}