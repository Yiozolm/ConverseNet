"""Explicit isolated FP32 candidates, excluded from the release build."""
import math
import torch


def check(x, p, k, b, s, eps):
    if any(v.dtype != torch.float32 for v in (x, p, k, b)):
        raise ValueError('FP32 candidates only; FP64 belongs to independent references')
    if p.shape != (*x.shape[:-2], x.shape[-2]*s, x.shape[-1]*s):
        raise ValueError('Invalid prior dimensions')


def aliases(t, s):
    return t if s == 1 else t.reshape(*t.shape[:-2], s, t.shape[-2]//s, s, t.shape[-1]//s).mean((-4, -2))


def kernel_fft(k, shape, real=False):
    kh, kw = k.shape[-2:]
    p = torch.nn.functional.pad(k, (0, shape[1]-kw, 0, shape[0]-kh))
    p = torch.roll(p, (-(kh//2), -(kw//2)), (-2, -1))
    return torch.fft.rfft2(p) if real else torch.fft.fft2(p)


def spectral(x, p, k, b, s, eps, *, K=None, P=None, Y=None):
    check(x, p, k, b, s, eps)
    K = kernel_fft(k, p.shape[-2:]) if K is None else K
    Y = torch.fft.fft2(x) if Y is None else Y
    P = (Y if p is x else torch.fft.fft2(p)) if P is None else P
    power = K.real.square()+K.imag.square()
    lam = torch.sigmoid(b-9.0)+eps
    q = (Y-aliases(K*P, s))/(aliases(power, s)+lam)
    if s > 1:
        q = q.repeat(1, 1, s, s)
    return torch.fft.ifft2(P+K.conj()*q).real


def disjoint_k2_s2(x, p, k, b, s, eps=1e-5):
    check(x, p, k, b, s, eps)
    if s != 2 or k.shape[-2:] != (2, 2):
        raise ValueError('Only k2/s2 is admitted to this prototype')
    predicted = torch.zeros_like(x)
    for a in range(2):
        for c in range(2):
            predicted = predicted+k[..., a, c, None, None]*p[..., 1-a::2, 1-c::2]
    power = k.square().sum((-2, -1), keepdim=True)
    lam = torch.sigmoid(b-9.0)+eps
    residual = (x-predicted)/(power+lam)
    out = torch.empty_like(p, memory_format=torch.contiguous_format)
    for a in range(2):
        for c in range(2):
            out[..., 1-a::2, 1-c::2] = p[..., 1-a::2, 1-c::2]+k[..., a, c, None, None]*residual
    return out


def transfer_shared(x, p, k, b, s, eps=1e-5):
    check(x, p, k, b, s, eps)
    if s != 1 or p is not x:
        raise ValueError('Explicit shared s1 prior required')
    K = kernel_fft(k, x.shape[-2:])
    lam = torch.sigmoid(b-9.0)+eps
    T = (K.conj()+lam)/(K.real.square()+K.imag.square()+lam)
    return torch.fft.ifft2(torch.fft.fft2(x)*T).real


class FixedTransferInference:
    def __init__(self):
        self.entries = {}

    def clear(self):
        self.entries.clear()

    def __call__(self, x, p, k, b, s, eps=1e-5):
        check(x, p, k, b, s, eps)
        if s != 1 or p is not x or torch.is_grad_enabled():
            raise ValueError('no_grad shared s1 required')
        def state(t):
            return (id(t), t.data_ptr(), t._version, tuple(t.shape), t.stride(), t.storage_offset(), t.device)
        capture = x.is_cuda and torch.cuda.is_current_stream_capturing()
        cacheable = not capture and all(t.is_leaf and not t.is_inference() for t in (k, b))
        stream = torch.cuda.current_stream(x.device).cuda_stream if x.is_cuda else 0
        key = (state(k), state(b), tuple(x.shape[-2:]), eps, stream, torch.is_inference_mode_enabled()) if cacheable else None
        T = self.entries.get(key, (None,))[0] if cacheable else None
        if T is None:
            K = kernel_fft(k, x.shape[-2:], real=True)
            lam = torch.sigmoid(b-9.0)+eps
            T = (K.conj()+lam)/(K.real.square()+K.imag.square()+lam)
            if cacheable:
                if len(self.entries) >= 16:
                    self.entries.clear()
                self.entries[key] = (T, k, b)
        return torch.fft.irfft2(torch.fft.rfft2(x)*T, s=x.shape[-2:])


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
                phases.append(z)
            if len(_geometry) >= 16:
                _geometry.clear()
            _geometry[key] = phases
    row, col = _geometry[key]
    Y = torch.fft.fft2(x)
    P = Y.repeat(1, 1, s, s)*row[:, None]*col[None, :]
    return spectral(x, p, k, b, s, eps, P=P, Y=Y)


def direct_dft(x, p, k, b, s, eps=1e-5):
    check(x, p, k, b, s, eps)
    if max(k.shape[-2:]) > 7:
        raise ValueError('Only small kernels are included')
    axes = []
    for n, support in zip(p.shape[-2:], k.shape[-2:]):
        f = torch.arange(n, device=x.device, dtype=torch.float32)
        a = torch.arange(support, device=x.device, dtype=torch.float32)-support//2
        angle = f[:, None]*a[None, :]*(-2*math.pi/n)
        axes.append(torch.polar(torch.ones_like(angle), angle))
    K = torch.einsum('ha,...ab,wb->...hw', axes[0], k.to(torch.complex64), axes[1])
    return spectral(x, p, k, b, s, eps, K=K)
