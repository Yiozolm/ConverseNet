"""Isolated current-source variants for FP32 kernel-spectrum precision."""
import hashlib
from pathlib import Path

import torch
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'Converse2D/torch_converse2d'
BUILD = ROOT / '.build/kernel_precision'
NAMES = ('fp32', 'kernel_fp64', 'adaptive')


HELPER = r'''
// Experiment-only policy. Counters do not synchronize the device.
static thread_local int64_t precision_high = 0, precision_low = 0;
static thread_local int64_t precision_decisions = 0, precision_hits = 0;
static thread_local double precision_risk = 0;
std::vector<int64_t> precision_stats() {
    return {precision_high, precision_low, precision_decisions, precision_hits};
}
void precision_reset() {
    precision_high = precision_low = precision_decisions = precision_hits = 0;
    precision_risk = 0;
}
double precision_last_risk() { return precision_risk; }

static Tensor make_kernel_fb(const Tensor& weight, int64_t h, int64_t w,
                             bool real_fft, bool high) {
    auto input = high ? weight.to(at::kDouble) : weight;
    Tensor otf;
    if (input.is_cuda() && !at::GradMode::is_enabled() && real_fft)
        otf = converse_psf_cuda(input, h, w);
    else {
        otf = at::constant_pad_nd(input, {0,w-input.size(3),0,h-input.size(2)},0);
        otf = at::roll(otf, {-(input.size(2)/2),-(input.size(3)/2)}, {-2,-1});
    }
    auto result = real_fft ? at::fft_rfft2(otf) : at::fft_fft2(otf);
    return high ? result.to(at::kComplexFloat) : result;
}

static Tensor select_kernel_fb(const Tensor& weight, int64_t h, int64_t w,
                              int64_t s, bool real_fft, double eps) {
    const bool eligible = real_fft && weight.scalar_type() == at::kFloat;
    bool high = eligible && PRECISION_MODE == 1;
    bool capture = false;
    if (weight.is_cuda()) capture = c10::cuda::currentStreamCaptureStatusMayInitCtx() != c10::cuda::CaptureStatus::None;
    // Avoid a host decision in capture. Adaptive training uses differentiable
    // high-precision preparation unconditionally.
    if (eligible && PRECISION_MODE == 2 && (capture || at::GradMode::is_enabled())) high = true;
    Tensor fb;
    if (eligible && PRECISION_MODE == 2 && !high) {
        fb = make_kernel_fb(weight,h,w,real_fft,false);
        auto power = at::real(fb).square() + at::imag(fb).square();
        auto grouped = alias_mean(s > 1 ? full_spectrum(power,w) : power,s);
        auto minimum = grouped.amin({-2,-1});
        auto l1 = weight.abs().sum({-2,-1});
        // eps bounds lambda from below, so bias changes cannot stale the choice.
        // This heuristic is not a certified output-error bound.
        precision_risk = (l1.square()/(minimum+eps)).amax().item<double>();
        ++precision_decisions;
        high = precision_risk >= 1e4;
    }
    if (high || !fb.defined()) fb = make_kernel_fb(weight,h,w,real_fft,high);
    if (high) ++precision_high; else ++precision_low;
    return fb;
}
'''


def generated(mode):
    content=(SOURCE/'converse2d.cpp').read_text(encoding='utf-8')
    content=content.replace('struct CacheEntry {', HELPER+'\nstruct CacheEntry {',1)
    content=content.replace('bool real_fft, inference;', 'bool real_fft, inference;\n    double policy_eps;',1)
    old='int64_t h, int64_t w, int64_t s, bool real_fft) {'
    assert content.count(old)==1
    content=content.replace(old,'int64_t h, int64_t w, int64_t s, bool real_fft, double eps) {',1)
    content=content.replace('it->real_fft == real_fft && it->inference == inference &&',
        'it->real_fft == real_fft && it->inference == inference &&\n                    (PRECISION_MODE != 2 || it->policy_eps == eps) &&',1)
    content=content.replace('auto result = std::make_pair(it->fb, it->invw);',
        '++precision_hits;\n                    auto result = std::make_pair(it->fb, it->invw);',1)
    begin=content.index('    const auto kh = weight.size(2), kw = weight.size(3);')
    end=content.index('    Tensor invw;',begin)
    content=content[:begin]+'''    const bool fused_prepare = weight.is_cuda() && !at::GradMode::is_enabled() && real_fft;
    auto fb = select_kernel_fb(weight,h,w,s,real_fft,eps);
'''+content[end:]
    content=content.replace('s, stream, real_fft, inference, bytes}', 's, stream, real_fft, inference, eps, bytes}',1)
    content=content.replace('Hs, Ws, scale, real_fft);','Hs, Ws, scale, real_fft, eps);',1)
    content=content.replace('TORCH_LIBRARY(converse2d, m) {','''TORCH_LIBRARY(converse2d, m) {
    m.def("precision_stats() -> int[]");
    m.def("precision_reset() -> ()");
    m.def("precision_last_risk() -> float");''',1)
    content=content.replace('TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m) {','''TORCH_LIBRARY_IMPL(converse2d, CompositeImplicitAutograd, m) {
    m.impl("precision_stats", TORCH_FN(precision_stats));
    m.impl("precision_reset", TORCH_FN(precision_reset));
    m.impl("precision_last_risk", TORCH_FN(precision_last_risk));''',1)
    prefix=f'kp_{NAMES[mode]}_converse'
    return '#define PRECISION_MODE '+str(mode)+'\n'+content.replace('converse',prefix)


def load_all():
    cpp_extension.SUBPROCESS_DECODE_ARGS=('utf-8','replace')
    ops={}
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (SOURCE/'converse2d.cpp',SOURCE/'converse2d_kernels.cu',Path(__file__))}
    for mode,name in enumerate(NAMES):
        build=BUILD/name
        build.mkdir(parents=True,exist_ok=True)
        prefix=f'kp_{name}_converse'
        src={'operator.cpp':generated(mode),'kernel.cu':(SOURCE/'converse2d_kernels.cu').read_text(encoding='utf-8').replace('converse',prefix)}
        for filename,content in src.items():
            path=build/filename
            if not path.exists() or path.read_text(encoding='utf-8')!=content:path.write_text(content,encoding='utf-8')
            hashes[f'{name}/{filename}']=hashlib.sha256(content.encode()).hexdigest()
        print('BUILD',name,flush=True)
        cpp_extension.load(name='kernel_precision_'+name,sources=[str(build/n) for n in src],
            extra_cflags=['/O2','/std:c++17','-DCONVERSE2D_WITH_CUDA=1'],
            extra_cuda_cflags=['-O3','-lineinfo'],with_cuda=True,is_python_module=False,
            build_directory=str(build),verbose=False)
        ops[name]=getattr(torch.ops,f'kp_{name}_converse2d')
    return ops,hashes


if __name__=='__main__':load_all()
