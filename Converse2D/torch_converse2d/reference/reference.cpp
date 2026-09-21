#include "reference.h"
#include "../common/spectrum_ops.h"
namespace converse2d::reference {
using namespace converse2d::detail;
using at::Tensor;
Tensor spectral(const Tensor& fy,const Tensor& fx0,const Tensor& fb,const Tensor& invw,
    const Tensor& lambda,int64_t W,int64_t Ws,int64_t scale,bool real_fft) {
Tensor fx;
        auto prediction = fb * fx0;
        if (real_fft && scale > 1) prediction = full_spectrum(prediction, Ws);
        prediction = alias_mean(prediction, scale);
        if (real_fft && scale > 1) prediction = prediction.slice(-1, 0, W / 2 + 1);
        auto correction = (fy - prediction) / (invw + lambda);
        if (scale > 1) {
            if (real_fft) correction = full_spectrum(correction, W);
            correction = correction.repeat({1,1,scale,scale});
            if (real_fft) correction = correction.slice(-1, 0, Ws / 2 + 1);
        }
        fx = fx0 + fb.conj() * correction;

return fx;
}
}
