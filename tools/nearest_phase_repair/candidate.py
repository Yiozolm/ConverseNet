"""Isolated nearest_spectral repair: exact, decidable axis coefficients only.

No production dispatch. FFTs, solver, reduction order, FP32 polar evaluation
at ordinary frequencies, and the original geometry-cache policy are retained.
The frozen helper implementation is shared with the prior failure reproducer.
"""
import math
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training_followup'))
from route_b_nearest import check, spectral


def exact_axis_coefficients(n, s):
    """Return only exact Gaussian-integer box-filter coefficients.

    f=0 gives s; quarter-turn roots have period <=4. A non-DC frequency
    with (f*s) % n == 0 gives a complete geometric sum of zero. Integer
    positions use Python arithmetic, never rounded float angle comparisons.
    The caller's original supported geometry is n>0, s>=2, n%s==0.
    """
    if not isinstance(n, int) or not isinstance(s, int) or n <= 0 or s < 2:
        raise ValueError('Positive integer axis and integer scale >=2 required')
    values = {}
    # Every integer through 2**24 is exactly representable in FP32. Avoid
    # advertising an exact DC coefficient outside this conservative bound.
    if s <= 2**24:
        values[0] = complex(s, 0)
    remainder = s % 4
    for quarter in (1, 2, 3):
        if (quarter * n) % 4:
            continue
        f = quarter * n // 4
        if quarter == 1:
            value = (0j, 1+0j, 1-1j, complex(0, -1))[remainder]
        elif quarter == 2:
            value = complex(s % 2, 0)
        else:
            value = (0j, 1+0j, 1+1j, 1j)[remainder]
        values[f] = value
    common = math.gcd(n, s)
    for multiple in range(1, common):
        values[multiple * (n // common)] = 0j
    return dict(sorted(values.items()))


def apply_exact_coefficients(z, n, s):
    """Ordinary z entries are not touched, recomputed, or symmetrized."""
    exact = exact_axis_coefficients(n, s)
    if exact:
        indices = torch.tensor(list(exact), device=z.device, dtype=torch.int64)
        values = torch.tensor(list(exact.values()), device=z.device, dtype=torch.complex64)
        z[indices] = values
    return z


_geometry = {}


def nearest_spectral(x, p, k, b, s, eps=1e-5):
    """Caller declares p=nearest(x); this is never inferred from shape alone."""
    check(x, p, k, b, s, eps)
    if s < 2:
        raise ValueError('Upsampling required')
    hs, ws = p.shape[-2:]
    key = (hs, ws, s, x.device)
    if key not in _geometry:
        with torch.no_grad():
            phases = []
            for n in (hs, ws):
                f = torch.arange(n, device=x.device, dtype=torch.float32)
                z = torch.zeros(n, device=x.device, dtype=torch.complex64)
                for a in range(s):
                    angle = f*(-2*math.pi*a/n)
                    z = z+torch.polar(torch.ones_like(angle), angle)
                z = apply_exact_coefficients(z, n, s)
                phases.append(z)
            if len(_geometry) >= 16:
                _geometry.clear()
            _geometry[key] = phases
    row, col = _geometry[key]
    Y = torch.fft.fft2(x)
    P = Y.repeat(1, 1, s, s)*row[:, None]*col[None, :]
    return spectral(x, p, k, b, s, eps, P=P, Y=Y)
