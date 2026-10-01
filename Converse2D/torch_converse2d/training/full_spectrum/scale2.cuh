#include "../../common/fp32_dispatch.h"
// Included within converse2d::full_training, after the generic CUDA helpers.
// Only W>1 and ATen 32-bit-byte-offset alias reductions use this candidate.
// For contiguous B,C,2,H,2,W input those reductions visit 00,01,10,11 in
// one thread and combine four accumulators from left to right. Each value
// first adds the +0 identity, exactly as Reduce.cuh::thread_reduce_impl.
template <class T> __device__ Z<T> scale2_sum4(Z<T> a, Z<T> b, Z<T> c, Z<T> d) {
    const Z<T> zero(0, 0);
    return add(add(add(add(zero, a), add(zero, b)), add(zero, c)),
               add(zero, d));
}
template <class T> __device__ T scale2_sum4_real(T a, T b, T c, T d) {
    return add_rn(
        add_rn(add_rn(add_rn(T(0), a), add_rn(T(0), b)), add_rn(T(0), c)),
        add_rn(T(0), d));
}
template <class T>
__device__ Z<T> scale2_mean4(Z<T> a, Z<T> b, Z<T> c, Z<T> d) {
    const auto sum = scale2_sum4(a, b, c, d);
    return {mul_rn(sum.real(), T(0.25)), mul_rn(sum.imag(), T(0.25))};
}

template <class T>
__global__ void scale2_forward(const Z<T> *y, const Z<T> *p, const Z<T> *k,
                               const T *l, Z<T> *out, Z<T> *q, T *d, I n, I C,
                               I H, I W, I KB, I KC) {
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const I hw = H * W, bc = i / hw, batch = bc / C, c = bc % C,
            h = (i / W) % H, w = i % W;
    const I kbc = kc(bc, C, KB, KC),
            di = ((KB == 1 ? 0 : batch) * C + c) * hw + i % hw;
    Z<T> filters[4], priors[4], products[4];
    T powers[4];
    I offsets[4];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int j = 2 * a + b;
            offsets[j] = (h + a * H) * (2 * W) + w + b * W;
            filters[j] = k[kbc * (4 * hw) + offsets[j]];
            priors[j] = p[bc * (4 * hw) + offsets[j]];
            products[j] = product(filters[j], priors[j]);
            powers[j] = norm(filters[j]);
        }
    }
    const Z<T> prediction =
        scale2_mean4(products[0], products[1], products[2], products[3]);
    const T km = mul_rn(
        scale2_sum4_real(powers[0], powers[1], powers[2], powers[3]), T(0.25));
    const T denominator = add_rn(km, l[c]);
    const Z<T> correction = add(y[i], -prediction) / Z<T>(denominator, 0);
    q[i] = correction;
    if (KB != 1 || batch == 0)
        d[di] = denominator;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        out[bc * (4 * hw) + offsets[j]] =
            add(priors[j], product(cj(filters[j]), correction));
}

template <class T, bool FuseKernel>
__global__ void scale2_adjoint(const Z<T> *g, const Z<T> *p, const Z<T> *k,
                               const Z<T> *q, const T *d, Z<T> *gy, Z<T> *gp,
                               Z<T> *direct, Z<T> *prediction, Z<T> *gd,
                               Z<T> *gk, I n, I C, I H, I W, I KB, I KC) {
    const I i = I(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    const I hw = H * W, bc = i / hw, h = (i / W) % H, w = i % W,
            kbc = kc(bc, C, KB, KC);
    const I di = ((KB == 1 ? 0 : bc / C) * C + bc % C) * hw + i % hw;
    Z<T> filters[4], grads[4], products[4];
    I offsets[4];
#pragma unroll
    for (int a = 0; a < 2; ++a) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {
            const int j = 2 * a + b;
            offsets[j] = (h + a * H) * (2 * W) + w + b * W;
            filters[j] = k[kbc * (4 * hw) + offsets[j]];
            grads[j] = g[bc * (4 * hw) + offsets[j]];
            products[j] = product(grads[j], filters[j]);
        }
    }
    const Z<T> t =
        scale2_sum4(products[0], products[1], products[2], products[3]);
    const Z<T> den(d[di], 0);
    Z<T> gyi;
    if (gy || gp || prediction || FuseKernel)
        gyi = t / den;
    if (gy)
        gy[i] = gyi;
    // Complex storage retains the real-view stride seen by sum_to(gd).
    Z<T> gd_value;
    if (gd || FuseKernel) {
        gd_value = product(-t, cj(q[i] / den));
        if (gd)
            gd[i] = gd_value;
    }
    // ATen's CPU-scalar division multiplies by a complex reciprocal. Keep
    // the complex multiply's explicit FMA boundaries, including zero terms.
    if (gp || direct || prediction || FuseKernel) {
        const Z<T> gm = product(-gyi, Z<T>(T(0.25), T(0)));
        // With no kernel broadcast all kernel-gradient sum_to operations are
        // identities. Preserve the real power / 4 rounding separately from
        // both the complex gm / 4 and the subsequent 2*k multiplications.
        T power_scale;
        if constexpr (FuseKernel)
            power_scale = mul_rn(gd_value.real(), T(0.25));
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const I index = bc * (4 * hw) + offsets[j];
            if (gp)
                gp[index] = add(grads[j], product(gm, cj(filters[j])));
            if constexpr (FuseKernel) {
                const Z<T> direct_value = product(grads[j], cj(q[i]));
                const Z<T> prediction_value = product(gm, cj(p[index]));
                const Z<T> power_value(
                    mul_rn(power_scale, mul_rn(T(2), filters[j].real())),
                    mul_rn(power_scale, mul_rn(T(2), filters[j].imag())));
                // Match the separate conj materialization and adj_kernel's
                // nested adds, including the zero real/imaginary components.
                gk[index] = add(add(add(cj(direct_value), prediction_value),
                                    Z<T>(0, power_value.imag())),
                                Z<T>(power_value.real(), 0));
            } else {
                if (direct)
                    direct[index] = product(grads[j], cj(q[i]));
                if (prediction)
                    prediction[index] = product(gm, cj(p[index]));
            }
        }
    }
}

std::vector<Tensor> full_scale2_forward_cuda(Tensor y0, Tensor p0, Tensor k0,
                                             Tensor l0) {
    auto y = plain(y0), p = plain(p0), k = plain(k0), l = plain(l0);
    auto out = at::empty(p.sizes(), p.options()),
         q = at::empty(y.sizes(), y.options());
    auto d =
        at::empty({k.size(0), y.size(1), y.size(2), y.size(3)}, l.options());
    auto stream = c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(l.scalar_type(), "full_scale2_forward", [&] {
        scale2_forward<scalar_t><<<(y.numel() + 255) / 256, 256, 0, stream>>>(
            y.data_ptr<Z<scalar_t>>(), p.data_ptr<Z<scalar_t>>(),
            k.data_ptr<Z<scalar_t>>(), l.data_ptr<scalar_t>(),
            out.data_ptr<Z<scalar_t>>(), q.data_ptr<Z<scalar_t>>(),
            d.data_ptr<scalar_t>(), y.numel(), y.size(1), y.size(2), y.size(3),
            k.size(0), k.size(1));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {out, q, d};
}

std::vector<Tensor> full_scale2_adjoint_cuda(Tensor g0, Tensor p0, Tensor k0,
                                             Tensor q0, Tensor d0, bool need_y,
                                             bool need_p, bool need_k,
                                             bool need_l) {
    auto g = plain(g0), p = need_k ? plain(p0) : Tensor(), k = plain(k0),
         q = (need_k || need_l) ? plain(q0) : Tensor(), d = plain(d0);
    const bool fuse_kernel =
        need_k && k.size(0) == g.size(0) && k.size(1) == g.size(1);
    auto gy = need_y ? at::empty(q0.sizes(), q0.options()) : Tensor(),
         gp = need_p ? at::empty(p0.sizes(), p0.options()) : Tensor();
    auto direct =
        need_k && !fuse_kernel ? at::empty(g.sizes(), g.options()) : Tensor();
    auto prediction =
        need_k && !fuse_kernel ? at::empty(p0.sizes(), p0.options()) : Tensor();
    auto gd = (need_l || (need_k && !fuse_kernel))
                  ? at::empty(q0.sizes(), q0.options())
                  : Tensor();
    auto gk = fuse_kernel ? at::empty(k.sizes(), k.options()) : Tensor();
    auto stream = c10::cuda::getCurrentCUDAStream();
    CONVERSE_DISPATCH_FP32(d.scalar_type(), "full_scale2_adjoint", [&] {
        auto launch = [&](auto tag) {
            constexpr bool FuseKernel = decltype(tag)::value;
            scale2_adjoint<scalar_t, FuseKernel>
                <<<(q0.numel() + 255) / 256, 256, 0, stream>>>(
                    g.data_ptr<Z<scalar_t>>(), optional_data<Z<scalar_t>>(p),
                    k.data_ptr<Z<scalar_t>>(), optional_data<Z<scalar_t>>(q),
                    d.data_ptr<scalar_t>(), optional_data<Z<scalar_t>>(gy),
                    optional_data<Z<scalar_t>>(gp),
                    optional_data<Z<scalar_t>>(direct),
                    optional_data<Z<scalar_t>>(prediction),
                    optional_data<Z<scalar_t>>(gd),
                    optional_data<Z<scalar_t>>(gk), q0.numel(), q0.size(1),
                    q0.size(2), q0.size(3), k.size(0), k.size(1));
        };
        if (fuse_kernel)
            launch(std::true_type{});
        else
            launch(std::false_type{});
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {gy, gp, direct, prediction, gd, gk};
}
