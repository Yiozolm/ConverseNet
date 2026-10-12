"""Which rounding pattern do ATen's FP32 complex kernels use on this GPU?

The frozen Python FP32 baseline is a chain of ATen elementwise kernels; the fused training
kernels mirror its operation boundaries with explicit IEEE roundings (no FMA contraction). If
the ATen SASS for one architecture fuses a multiply-add, the two agree bitwise where it does not
and differ where it does. For each operation this probe computes, in FP64 with exact FP32
products, every plausible rounding pattern and reports how many elements of the ATen result
match each pattern bitwise.

    python tools/v4_release_a100/aten_rounding_probe.py --output <new json>
"""
import argparse
import json
from pathlib import Path
import time

import torch

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--n', type=int, default=1 << 20)
parser.add_argument('--seed', type=int, default=11)
args = parser.parse_args()
if args.output.exists():
    raise SystemExit(f'refusing to overwrite {args.output}')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dev = torch.device('cuda')
g = torch.Generator().manual_seed(args.seed)
f32 = lambda *shape: torch.randn(*shape, generator=g).to(dev)
r32 = lambda t64: t64.float()  # FP64 -> FP32 is the IEEE rounding of the FP64 value
match = lambda a, b: float((a.view(torch.int32) == b.view(torch.int32)).double().mean())

report = dict(gpu=torch.cuda.get_device_name(0), capability='%d.%d' % torch.cuda.get_device_capability(),
              torch=torch.__version__, cuda=torch.version.cuda, arch_list=torch.cuda.get_arch_list(),
              n=args.n, ops={})

# 1. complex * complex: real part ar*br - ai*bi, imaginary part ar*bi + ai*br.
ar, ai, br, bi = f32(args.n), f32(args.n), f32(args.n), f32(args.n)
a = torch.complex(ar, ai)
b = torch.complex(br, bi)
m = a * b
A, B, C, D = ar.double(), ai.double(), br.double(), bi.double()
patterns_re = {
    'no_fma  r(r(ar*br) - r(ai*bi))': r32(r32(A * C).double() - r32(B * D).double()),
    'fma_a   r(ar*br - r(ai*bi))': r32(A * C - r32(B * D).double()),
    'fma_b   r(r(ar*br) - ai*bi)': r32(r32(A * C).double() - B * D),
    'exact   r(ar*br - ai*bi)': r32(A * C - B * D),
}
patterns_im = {
    'no_fma  r(r(ar*bi) + r(ai*br))': r32(r32(A * D).double() + r32(B * C).double()),
    'fma_a   r(ar*bi + r(ai*br))': r32(A * D + r32(B * C).double()),
    'fma_b   r(r(ar*bi) + ai*br)': r32(r32(A * D).double() + B * C),
    'exact   r(ar*bi + ai*br)': r32(A * D + B * C),
}
report['ops']['complex_mul_real'] = {k: match(m.real.contiguous(), v) for k, v in patterns_re.items()}
report['ops']['complex_mul_imag'] = {k: match(m.imag.contiguous(), v) for k, v in patterns_im.items()}

# 1b. conj(a) * b, as in k.conj() * correction.
mc = a.conj() * b
patterns_cre = {
    'no_fma  r(r(ar*br) + r(ai*bi))': r32(r32(A * C).double() + r32(B * D).double()),
    'fma_a   r(ar*br + r(ai*bi))': r32(A * C + r32(B * D).double()),
    'fma_b   r(r(ar*br) + ai*bi)': r32(r32(A * C).double() + B * D),
    'exact   r(ar*br + ai*bi)': r32(A * C + B * D),
}
report['ops']['conj_mul_real'] = {k: match(mc.real.contiguous(), v) for k, v in patterns_cre.items()}

# 2. power = re.square() + im.square() (two ATen ops) and the fused alternative.
p = ar.square() + ai.square()
report['ops']['power'] = {
    'no_fma  r(r(re^2) + r(im^2))': match(p, r32(r32(A * A).double() + r32(B * B).double())),
    'fma     r(re^2 + r(im^2))': match(p, r32(A * A + r32(B * B).double())),
    'fma_rev r(r(re^2) + im^2)': match(p, r32(r32(A * A).double() + B * B)),
}

# 3. complex / real (correction = numerator / (real denominator)).
den = (f32(args.n).abs() + 0.5)
q = a / den
Dn = den.double()
report['ops']['complex_div_real'] = {
    'real: r(ar/den)': match(q.real.contiguous(), r32(A / Dn)),
    'real: r(ar * r(1/den))': match(q.real.contiguous(), r32(A * r32(1 / Dn).double())),
    'imag: r(ai/den)': match(q.imag.contiguous(), r32(B / Dn)),
}

# 3b. complex / complex (if the baseline divides by a complex denominator anywhere).
qc = a / b
scale = torch.maximum(br.abs(), bi.abs()).double()
report['ops']['complex_div_complex_real_part'] = {
    'naive  r((ar*br + ai*bi) / (br^2 + bi^2)) exact inside': match(qc.real.contiguous(), r32((A * C + B * D) / (C * C + D * D))),
}

# 4. real * complex broadcast (regularizer-like) and complex + complex.
s = ar * b
report['ops']['real_times_complex'] = {'r(ar*br), r(ar*bi)': match(s.real.contiguous(), r32(A * C))}
summ = a + b
report['ops']['complex_add'] = {'r(ar+br)': match(summ.real.contiguous(), r32(A + C))}

# 5. sigmoid(bias - 9) + eps and the 1/N IFFT normalization path are scalar-ish; skip.

# 6. fft2 of a real 5x7 plane: R2C+fill vs C2C of (x, 0). Same on this GPU?
x = f32(3, 5, 5, 7)
f_real = torch.fft.fft2(x)
f_c2c = torch.fft.fft2(torch.complex(x, torch.zeros_like(x)))
report['ops']['fft2_real_vs_c2c_5x7'] = {
    'bitwise equal': bool(torch.equal(f_real, f_c2c)),
    'max abs diff': float((f_real - f_c2c).abs().max()),
    'rel_l2 vs fp64 (real input)': float(((f_real.double() - torch.fft.fft2(x.double())).abs().pow(2).sum().sqrt()
                                       / torch.fft.fft2(x.double()).abs().pow(2).sum().sqrt())),
    'rel_l2 vs fp64 (c2c)': float(((f_c2c.double() - torch.fft.fft2(x.double())).abs().pow(2).sum().sqrt()
                                / torch.fft.fft2(x.double()).abs().pow(2).sum().sqrt())),
}
# Same transform, strided (transposed) layout: a different cuFFT plan?
xt = x.transpose(-1, -2).contiguous().transpose(-1, -2)
report['ops']['fft2_contiguous_vs_transposed_5x7'] = {'bitwise equal': bool(torch.equal(f_real, torch.fft.fft2(xt)))}

report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2))
print(report['gpu'], 'sm', report['capability'], 'arch list', report['arch_list'])
for op, res in report['ops'].items():
    print(op)
    for k, v in res.items():
        print(f'   {k:48s} {v}')
