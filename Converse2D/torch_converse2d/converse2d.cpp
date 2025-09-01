#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/fft_fftn.h>
#include <ATen/ops/fft_ifftn.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/real.h>
#include <ATen/ops/sigmoid.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/conj_physical.h>

using at::Tensor;

namespace
{

    static inline Tensor sfold_upsample_zero_insertion(const Tensor &x, int64_t s)
    {
        TORCH_CHECK(s >= 1, "scale must be >= 1");
        if (s == 1)
            return x;
        auto sizes = x.sizes().vec();
        sizes[sizes.size() - 2] *= s;
        sizes[sizes.size() - 1] *= s;
        Tensor z = at::zeros(sizes, x.options());
        z.index_put_(
            {at::indexing::Slice(), at::indexing::Slice(),
             at::indexing::Slice(0, z.size(-2), s),
             at::indexing::Slice(0, z.size(-1), s)},
            x);
        return z;
    }

    static inline Tensor p2o(const Tensor &psf, int64_t H, int64_t W)
    {
        TORCH_CHECK(psf.dim() == 4 && psf.size(0) == 1, "psf must be (1,C,kh,kw)");
        auto C = psf.size(1);
        auto kh = psf.size(2);
        auto kw = psf.size(3);
        Tensor otf = at::zeros({1, C, H, W}, psf.options());
        otf.index_put_({0, at::indexing::Slice(), at::indexing::Slice(0, kh), at::indexing::Slice(0, kw)}, psf);
        const int64_t sh = -static_cast<int64_t>(kh / 2);
        const int64_t sw = -static_cast<int64_t>(kw / 2);
        otf = at::roll(otf, {sh, sw}, {-2, -1});
        return at::fft_fftn(otf, c10::optional<c10::IntArrayRef>({}), c10::optional<c10::IntArrayRef>({-2, -1}), c10::nullopt);
    }

    static inline Tensor splits_mean_then_mean(const Tensor &a, int64_t s)
    {
        TORCH_CHECK(a.dim() >= 2, "tensor must have spatial dims");
        TORCH_CHECK(a.size(-2) % s == 0 && a.size(-1) % s == 0, "spatial not divisible by scale");

        const auto &sizes = a.sizes();
        const int64_t L = a.dim();
        const int64_t W = sizes[L - 2];
        const int64_t H = sizes[L - 1];
        const int64_t W_s = W / s;
        const int64_t H_s = H / s;

        std::vector<int64_t> view_shape;
        view_shape.reserve(L + 2);
        for (int64_t i = 0; i < L - 2; ++i)
            view_shape.push_back(sizes[i]);
        view_shape.push_back(s);
        view_shape.push_back(W_s);
        view_shape.push_back(s);
        view_shape.push_back(H_s);
        Tensor v = a.view(view_shape);

        std::vector<int64_t> perm;
        perm.reserve(view_shape.size());
        for (int64_t i = 0; i < L - 2; ++i)
            perm.push_back(i);
        perm.push_back(L - 2 + 1); // W_s
        perm.push_back(L - 2 + 3); // H_s
        perm.push_back(L - 2 + 0); // s
        perm.push_back(L - 2 + 2); // s
        Tensor p = v.permute(perm).contiguous();

        std::vector<int64_t> merge_shape;
        merge_shape.reserve(L + 1);
        for (int64_t i = 0; i < L - 2; ++i)
            merge_shape.push_back(p.size(i));
        merge_shape.push_back(W_s);
        merge_shape.push_back(H_s);
        merge_shape.push_back(s * s);
        Tensor r = p.view(merge_shape);

        return r.mean(-1, /*keepdim=*/false);
    }

    Tensor converse2d_forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int64_t scale, double eps)
    {
        TORCH_CHECK(x.dim() == 4, "x must be (B,C,H,W)");
        TORCH_CHECK(x0.dim() == 4, "x0 must be (B,C,Hs,Ws)");
        TORCH_CHECK(weight.dim() == 4 && weight.size(0) == 1, "weight must be (1,C,kh,kw)");
        TORCH_CHECK(bias.dim() == 4 && bias.size(0) == 1 && bias.size(2) == 1 && bias.size(3) == 1, "bias must be (1,C,1,1)");
        TORCH_CHECK(x.device() == x0.device() && x.device() == weight.device() && x.device() == bias.device(), "tensors on same device");
        TORCH_CHECK(scale >= 1, "scale must be >= 1");

        x = x.contiguous();
        x0 = x0.contiguous();
        weight = weight.contiguous();
        bias = bias.contiguous();

        const int64_t B = x.size(0);
        const int64_t C = x.size(1);
        const int64_t H = x.size(2);
        const int64_t W = x.size(3);
        const int64_t Hs = H * scale;
        const int64_t Ws = W * scale;

        Tensor lambda_ = at::sigmoid(bias - 9.0) + eps;

        Tensor STy = sfold_upsample_zero_insertion(x, scale);

        Tensor FB = p2o(weight, Hs, Ws);
        Tensor FBC = at::conj_physical(FB);
        Tensor F2B = at::abs(FB).pow(2.0);

        Tensor F_STy = at::fft_fftn(STy, c10::optional<c10::IntArrayRef>({}), c10::optional<c10::IntArrayRef>({-2, -1}), c10::nullopt);
        Tensor FBFy = FBC * F_STy;
        Tensor FR = FBFy + at::fft_fftn(lambda_ * x0, c10::optional<c10::IntArrayRef>({}), c10::optional<c10::IntArrayRef>({-2, -1}), c10::nullopt);

        Tensor x1 = FB * FR;

        Tensor FBR = splits_mean_then_mean(x1, scale);
        Tensor invW = splits_mean_then_mean(F2B, scale);

        Tensor invW_plus = invW + lambda_;
        Tensor invWBR = FBR / invW_plus;

        Tensor invWBR_rep = invWBR.repeat({1, 1, scale, scale});

        Tensor FCBinvWBR = FBC * invWBR_rep;

        Tensor FX = (FR - FCBinvWBR) / lambda_;
        Tensor out_c = at::fft_ifftn(FX, c10::optional<c10::IntArrayRef>({}), c10::optional<c10::IntArrayRef>({-2, -1}), c10::nullopt);
        Tensor out = at::real(out_c);
        (void)B;
        (void)C;
        (void)H;
        (void)W;
        return out;
    }

}

TORCH_LIBRARY(converse2d, m)
{
    m.def("forward(Tensor x, Tensor x0, Tensor weight, Tensor bias, int scale, float eps=1e-5) -> Tensor");
}
TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m)
{
    m.impl("forward", TORCH_FN(converse2d_forward));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
