#include "../common/fp32_dispatch.h"
#include "launchers.cuh"
#include "detail/alias_kernels.cuh"
using namespace converse2d::inference_detail;
void launch_inference_scale3(const at::Tensor& y,const at::Tensor& prior,const at::Tensor& kernel,const at::Tensor& denom,const at::Tensor& lambda,at::Tensor& q,at::Tensor& out,int64_t H,int64_t W,int64_t s,bool half,cudaStream_t stream) {
const int64_t n=prior.numel();
constexpr int threads=256;
CONVERSE_DISPATCH_FP32(lambda.scalar_type(), "converse_spectral", [&] {
            const bool small = n <= INT_MAX - threads && H*s <= INT_MAX && W*s <= INT_MAX;
            if (half) {
                if (!denom.defined()) {
                    if (small) launch_scaled<3,scalar_t,true,int,true>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                    else launch_scaled<3,scalar_t,true,int64_t,true>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                } else {
                    if (small) launch_scaled<3,scalar_t,true,int>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                    else launch_scaled<3,scalar_t,true,int64_t>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                }
            } else {
                if (small) launch_scaled<3,scalar_t,false,int>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
                else launch_scaled<3,scalar_t,false,int64_t>(y,prior,kernel,denom,lambda,q,out,H,W,s,stream);
            }
});
}
