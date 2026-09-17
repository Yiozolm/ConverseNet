// Include the same headers first so the Windows SDK's RPC `small` macro cannot
// replace the production kernel launcher's local variable of that name.
#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
#include <climits>
#ifdef small
#undef small
#endif
#include "../../Converse2D/torch_converse2d/converse2d_kernels.cu"
