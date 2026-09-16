#pragma once

// FP32/FP64 operator implementation. Included by converse2d.cpp after the
// shared spectrum/cache and FFT scheduling helpers. The low-precision CPU
// fallback also reuses this differentiable mathematical core.
namespace converse2d::fp32 {
#ifdef CONVERSE2D_WITH_CUDA
static Tensor tiled_inference(const Tensor& x, const Tensor& x0, bool same_prior,
    const Tensor& fb, const Tensor& invw, const Tensor& lambda,
    int64_t scale, int64_t tile) {
    const auto batch = x.size(0), h = x.size(2), w = x.size(3);
    const auto hs = x0.size(2), ws = x0.size(3);
    auto output = at::empty(x0.sizes(), x0.options());
    for (int64_t start = 0; start < batch; start += tile) {
        const auto count = std::min(tile, batch - start);
        auto y = at::fft_rfft2(x.narrow(0, start, count));
        auto prior = same_prior ? y : at::fft_rfft2(x0.narrow(0, start, count));
        // Prepare/cache the complete filter once, then select per-example
        // spectra without turning weight views into new cache keys.
        auto filter = fb.size(0) == 1 ? fb : fb.narrow(0, start, count);
        auto power = !invw.defined() || invw.size(0) == 1 ? invw : invw.narrow(0, start, count);
        auto corrected = converse_spectral_cuda(y, prior, filter, power, lambda, h, w, scale);
        auto destination = output.narrow(0, start, count);
        converse2d::c2r::inverse(corrected, hs, ws, destination);
    }
    return output;
}

static Tensor nearest_inference(const Tensor& x, const Tensor& fb, const Tensor& invw,
    const Tensor& lambda, const Tensor& phase, int64_t scale, int64_t tile,
    at::ScalarType output_dtype) {
    const auto batch=x.size(0), h=x.size(2), w=x.size(3), hs=h*scale, ws=w*scale;
    if (tile == batch) {
        auto fy=at::fft_rfft2(x);
        auto fx=converse_nearest_spectral_cuda(fy,fb,invw,lambda,phase,h,w,scale);
        return converse2d::c2r::inverse(fx,hs,ws,Tensor(),output_dtype);
    }
    auto output=at::empty({batch,x.size(1),hs,ws},x.options());
    for (int64_t start=0; start<batch; start+=tile) {
        const auto count=std::min(tile,batch-start);
        auto fy=at::fft_rfft2(x.narrow(0,start,count));
        auto filter=fb.size(0)==1 ? fb : fb.narrow(0,start,count);
        auto power=!invw.defined() || invw.size(0)==1 ? invw : invw.narrow(0,start,count);
        auto fx=converse_nearest_spectral_cuda(fy,filter,power,lambda,phase,h,w,scale);
        auto destination=output.narrow(0,start,count);
        converse2d::c2r::inverse(fx,hs,ws,destination);
    }
    return output;
}

#endif

static Tensor forward(Tensor x, Tensor x0, Tensor weight, Tensor bias,
                      int64_t scale, double eps, bool nearest) {
    TORCH_INTERNAL_ASSERT(x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble);
    const auto B=x.size(0), C=x.size(1), H=x.size(2), W=x.size(3);
    const auto Hs=H*scale, Ws=W*scale;
    const auto output_dtype=x.scalar_type(), compute_dtype=output_dtype;
    bool fused_nearest = false;
#ifdef CONVERSE2D_WITH_CUDA
    fused_nearest = nearest && scale > 1 && x.is_cuda() && !at::GradMode::is_enabled();
#endif
    const bool input_same_prior = !nearest && x.is_same(x0);
    x = x.to(compute_dtype).contiguous();
    if (nearest && !fused_nearest) {
        // The reference path retains differentiable nearest interpolation.
        x0 = scale == 1 ? x : at::upsample_nearest2d(x,at::IntArrayRef({Hs,Ws}),
                                                   double(scale),double(scale));
    }
    const bool same_prior = !fused_nearest && (input_same_prior || (nearest && scale == 1));
    auto source = weight;
    if (!fused_nearest) x0 = same_prior ? x : x0.to(compute_dtype).contiguous();
    weight = weight.to(compute_dtype);
    bias = bias.to(compute_dtype).contiguous();
    auto lambda = at::sigmoid(bias - 9.0) + eps;
    auto spectra = spectrum(source, weight, Hs, Ws, scale);
    auto fb = spectra.first, invw = spectra.second;
#ifdef CONVERSE2D_WITH_CUDA
    if (fused_nearest) {
        auto phase=nearest_phase(x,Hs,Ws,scale);
        const auto tile=output_dtype == at::kFloat ? fft_batch_size(x,C*Hs*Ws*4) : B;
        return nearest_inference(x,fb,invw,lambda,phase,scale,tile,output_dtype);
    }
    if (x.is_cuda() && output_dtype == at::kFloat && !at::GradMode::is_enabled()) {
        const auto tile = fft_batch_size(x0,x0.nbytes()/B);
        if (tile < B) return tiled_inference(x, x0, same_prior, fb, invw, lambda, scale, tile);
    }
#endif
    auto fy = at::fft_rfft2(x);
    auto fx0 = same_prior ? fy : at::fft_rfft2(x0);
    Tensor fx;
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && !at::GradMode::is_enabled()) {
        fx = converse_spectral_cuda(fy, fx0, fb, invw, lambda, H, W, scale);
    } else
#endif
    {
        auto prediction = fb * fx0;
        if (scale > 1) prediction = full_spectrum(prediction, Ws);
        prediction = alias_mean(prediction, scale);
        if (scale > 1) prediction = prediction.slice(-1, 0, W / 2 + 1);
        auto correction = (fy - prediction) / (invw + lambda);
        if (scale > 1) {
            correction = full_spectrum(correction, W);
            correction = correction.repeat({1,1,scale,scale});
            correction = correction.slice(-1, 0, Ws / 2 + 1);
        }
        fx = fx0 + fb.conj() * correction;
    }
    Tensor out;
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && !at::GradMode::is_enabled()) out = converse2d::c2r::inverse(fx,Hs,Ws,Tensor(),output_dtype);
    else
#endif
    out = at::fft_irfft2(fx, at::IntArrayRef({Hs,Ws}));
    return out.to(output_dtype);
}

} // namespace converse2d::fp32
