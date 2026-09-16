#pragma once

// FP16/BF16 operator implementation. Included after the shared infrastructure
// and FP32 core; input/output storage, tiling and training boundaries live here.
namespace converse2d::low_precision {
#ifdef CONVERSE2D_WITH_CUDA
// Low-precision activations stay in their original storage between tiles. Only
// one tile is promoted; C2R normalizes directly into the final FP16/BF16 slice.
static Tensor tiled_inference(const Tensor& x, const Tensor& x0,
    const Tensor& fb, const Tensor& invw, const Tensor& lambda, const Tensor& phase,
    int64_t scale, int64_t tile, bool nearest) {
    const auto batch=x.size(0), h=x.size(2), w=x.size(3), hs=h*scale, ws=w*scale;
    const bool same_prior = nearest ? scale == 1 : x.is_same(x0);
    auto output = at::empty({batch,x.size(1),hs,ws},x.options());
    for (int64_t start=0; start<batch; start+=tile) {
        const auto count=std::min(tile,batch-start);
        auto input = x.narrow(0,start,count).to(at::kFloat).contiguous();
        auto fy = native_fft::real_fft(input,x.scalar_type());
        auto filter = fb.size(0)==1 ? fb : fb.narrow(0,start,count);
        auto power = !invw.defined() || invw.size(0)==1 ? invw : invw.narrow(0,start,count);
        Tensor fx;
        if (nearest && scale > 1) {
            fx = converse_nearest_spectral_cuda(fy,filter,power,lambda,phase,h,w,scale);
        } else {
            auto prior = same_prior ? fy : native_fft::real_fft(x0.narrow(0,start,count).to(at::kFloat).contiguous(),x.scalar_type());
            fx = converse_spectral_cuda(fy,prior,filter,power,lambda,h,w,scale);
        }
        auto destination = output.narrow(0,start,count);
        native_fft::inference_inverse(fx,hs,ws,x.scalar_type(),destination);
    }
    return output;
}
#endif

static Tensor forward(Tensor x, Tensor x0, Tensor weight, Tensor bias,
                      int64_t scale, double eps, bool nearest) {
    const auto output_dtype = x.scalar_type();
    TORCH_INTERNAL_ASSERT(output_dtype == at::kHalf || output_dtype == at::kBFloat16);
    const bool same_prior = nearest ? scale == 1 : x.is_same(x0);
#ifdef CONVERSE2D_WITH_CUDA
    if (x.is_cuda() && at::GradMode::is_enabled())
        return converse2d::training::forward(x,x0,weight,bias,scale,eps,nearest);
    if (x.is_cuda() && !at::GradMode::is_enabled()) {
        const auto B=x.size(0), C=x.size(1), H=x.size(2), W=x.size(3);
        const auto Hs=H*scale, Ws=W*scale;
        auto filter = weight.to(at::kFloat);
        auto lambda = at::sigmoid(bias.to(at::kFloat).contiguous() - 9.0) + eps;
        // Keep the original parameter identity for cache lookup/invalidation.
        auto spectra = spectrum(weight,filter,Hs,Ws,scale);
        auto phase = nearest && scale > 1 ? nearest_phase(filter,Hs,Ws,scale) : Tensor();
        // The capacity is based on FP32 working bytes, not storage bytes.
        const auto tile = fft_batch_size(x,C*Hs*Ws*4);
        if (tile < B)
            return tiled_inference(x,x0,spectra.first,spectra.second,lambda,phase,scale,tile,nearest);
        auto input = x.to(at::kFloat).contiguous();
        auto fy = native_fft::real_fft(input,output_dtype);
        Tensor fx;
        if (nearest && scale > 1) {
            fx = converse_nearest_spectral_cuda(fy,spectra.first,spectra.second,lambda,phase,H,W,scale);
        } else {
            auto prior = same_prior ? fy : native_fft::real_fft(x0.to(at::kFloat).contiguous(),output_dtype);
            fx = converse_spectral_cuda(fy,prior,spectra.first,spectra.second,lambda,H,W,scale);
        }
        return native_fft::inference_inverse(fx,Hs,Ws,output_dtype);
    }
#endif
    // CPU fallback. Promotion precedes nearest interpolation so
    // its backward reduces in FP32. Casts preserve parameter/input gradients
    // and the shared differentiable core retains second derivatives.
    auto input = x.to(at::kFloat);
    auto prior = nearest ? Tensor() : (same_prior ? input : x0.to(at::kFloat));
    return converse2d::fp32::forward(input,prior,weight.to(at::kFloat),bias.to(at::kFloat),
                                    scale,eps,nearest).to(output_dtype);
}
} // namespace converse2d::low_precision
