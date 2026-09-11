import os
import torch
import torch.nn as nn
from models.util_converse import sequential, Converse2D, ConverseBlock, ConverseBlockAlphaVariant
from models import util_converse as converse_utils
from models.converse_core import converse2d_reference
# from utils import utils_image as util
import torch.fft
import torch.nn.init as init
import torch.nn.functional as F


class MultiConverseBlock(nn.Module):
    def __init__(self, in_channels=64, num_blocks=7, kernel_size=3, scale=1, padding=2, padding_mode="circular"):
        super(MultiConverseBlock, self).__init__()
        self.m_body = sequential(*[ConverseBlock(in_channels, in_channels, kernel_size, scale, padding, padding_mode) for _ in range(num_blocks)])
    def forward(self, x):
        x = self.m_body(x)
        return x


class MultiConverseBlockAlphaVariant(nn.Module):
    def __init__(self, in_channels=64, num_blocks=7, kernel_size=3, scale=1, padding=2, padding_mode="circular"):
        super(MultiConverseBlockAlphaVariant, self).__init__()
        self.m_body = sequential(*[ConverseBlockAlphaVariant(in_channels, in_channels, kernel_size, scale, padding, padding_mode) for _ in range(num_blocks)])
    def forward(self, x):
        x = self.m_body(x)
        return x


"""
# --------------------------------------------
# Data module, for condition the kernel
# --------------------------------------------
"""
class ConvReverseDataNet(nn.Module):
    def __init__(self,eps=1e-3,backend="auto",variant="v7"):
        super(ConvReverseDataNet, self).__init__()
        '''
        Converse2d operator for condition the kernel
        '''
        self.alpha = nn.Parameter(torch.zeros(1, 64, 1, 1))
        self.eps = eps
        self.backend = backend.lower()
        self.variant = variant.lower()
        if self.backend not in ("auto", "cuda", "pytorch"):
            raise ValueError("backend must be auto, cuda or pytorch")
        if self.variant not in ("v2", "v3", "v4", "v5", "v6", "v7"):
            raise ValueError("variant must be v2-v7")
    def forward(self, x, k, sf, padding = 0, padding_mode = 'circular'):

        output_dtype = x.dtype
        # Mixed precision from the surrounding network may leave alpha/kernel
        # in float32. Promote before dispatch while preserving each gradient.
        if k.dtype != x.dtype or self.alpha.dtype != x.dtype:
            dtype = torch.promote_types(torch.promote_types(x.dtype, k.dtype), self.alpha.dtype)
            if dtype in (torch.float16, torch.bfloat16):
                dtype = torch.float32
            x, k, alpha = x.to(dtype), k.to(dtype), self.alpha.to(dtype)
        else:
            alpha = self.alpha
        if padding > 0:
            x = nn.functional.pad(x, pad=[padding, padding, padding, padding], mode=padding_mode, value=0)
        x0 = x if sf == 1 else F.interpolate(x, scale_factor=sf, mode='nearest')
        backend = (os.environ.get("CONVERSE2D_BACKEND", "") or self.backend).lower()
        if backend not in ("auto", "cuda", "pytorch"):
            raise ValueError("backend must be auto, cuda or pytorch")
        available = False
        if backend != "pytorch" and x.is_cuda:
            converse_utils._try_import_converse2d_ext()
            available = hasattr(torch.ops.converse2d, "forward")
        if backend == "cuda" and not available:
            raise RuntimeError("ConvReverseDataNet backend='cuda' but CUDA extension is unavailable")
        if available:
            out = torch.ops.converse2d.forward(x,x0,k,alpha,sf,float(self.eps),self.variant)
        else:
            out = converse2d_reference(x,x0,k,alpha,sf,self.eps)

        if padding > 0:
            out = out[..., padding*sf:-padding*sf, padding*sf:-padding*sf]

        return out.to(output_dtype)
    def splits(self, a, scale):
        '''
        Split tensor `a` into `scale x scale` distinct blocks.
        Args:
            a: Tensor of shape (..., W, H)
            scale: Split factor
        Returns:
            b: Tensor of shape (..., W/scale, H/scale, scale^2)
        '''
        *leading_dims, W, H = a.size()
        W_s, H_s = W // scale, H // scale

        # Reshape to separate the scale factors
        b = a.view(*leading_dims, scale, W_s, scale, H_s)

        # Generate the permutation order
        permute_order = list(range(len(leading_dims))) + [len(leading_dims) + 1, len(leading_dims) + 3, len(leading_dims), len(leading_dims) + 2]
        b = b.permute(*permute_order).contiguous()

        # Combine the scale dimensions
        b = b.view(*leading_dims, W_s, H_s, scale * scale)
        return b
    def p2o(self, psf, shape):
        '''
        Convert point-spread function to optical transfer function.
        otf = p2o(psf) computes the Fast Fourier Transform (FFT) of the
        point-spread function (PSF) array and creates the optical transfer
        function (OTF) array that is not influenced by the PSF off-centering.
        Args:
            psf: NxCxhxw
            shape: [H, W]
        Returns:
            otf: NxCxHxWx2
        '''
        otf = torch.zeros(psf.shape[:-2] + shape).type_as(psf)
        otf[...,:psf.shape[-2],:psf.shape[-1]].copy_(psf)
        otf = torch.roll(otf, (-int(psf.shape[-2]/2), -int(psf.shape[-1]/2)), dims=(-2, -1))
        otf = torch.fft.fftn(otf, dim=(-2,-1))

        return otf
    def upsample(self, x, scale=3):
        '''s-fold upsampler
        Upsampling the spatial size by filling the new entries with zeros
        x: tensor image, NxCxWxH
        '''
        st = 0
        z = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*scale, x.shape[3]*scale)).type_as(x)
        z[..., st::scale, st::scale].copy_(x)
        return z


"""
# --------------------------------------------
# kernelnet module
# --------------------------------------------
"""
class KernelNet(nn.Module):
    def __init__(self, kernel_size=7):
        super(KernelNet, self).__init__()
        self.kernel_size = kernel_size
        self.fc1 = nn.Linear(kernel_size**2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 16 * (kernel_size**2))
        self.gelu = nn.GELU()
    def forward(self, k):
        k = k.to(self.fc1.weight.dtype)
        b,_,_,_ = k.shape
        k = k.reshape(b, -1) # accept noncontiguous caller kernels
        # fully connected
        k = self.gelu(self.fc1(k))
        k = self.gelu(self.fc2(k))
        k = self.fc3(k)
        k = k.view(b, 16, self.kernel_size, self.kernel_size)
        return k

"""
# --------------------------------------------
# ConverseNet
# for blind deblurring
# --------------------------------------------
"""
class ConverseNet(nn.Module):
    def __init__(self, num_iterations=5, in_channels=64, num_blocks=7):
        """
        totally denoiser for blind deblur
        """
        super(ConverseNet, self).__init__()
        self.p = MultiConverseBlock(in_channels=in_channels, num_blocks=num_blocks)
        self.conv1 = nn.Conv2d(3, in_channels, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channels, 3, 1, 1, 0)
        self.num_iterations = num_iterations
        
    def forward(self, x):
        '''
        x: tensor, NxCxWxH
        '''
        x = self.conv1(x)
        for i in range(self.num_iterations):
            x = self.p(x)
        x = self.conv2(x)
        return x


"""
# --------------------------------------------
# ConverseUSRNet
# for non-blind deblurring, condition the kernel
# --------------------------------------------
"""
class ConverseUSRNet(nn.Module):
    def __init__(self, num_iterations=5, in_channels=64, num_blocks=7, backend="auto", variant="v7"):
        super(ConverseUSRNet, self).__init__()

        self.d = ConvReverseDataNet(backend=backend, variant=variant)
        self.p = MultiConverseBlockAlphaVariant(in_channels=in_channels, num_blocks=num_blocks)
        for layer in self.p.modules():
            if isinstance(layer, Converse2D):
                layer.backend = self.d.backend
                layer.variant = self.d.variant
        self.conv1 = nn.Conv2d(3, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(64, 3, 1, 1, 0)
        self.kernelnet = KernelNet()
        self.num_iterations = num_iterations
        
        self.convs = nn.ModuleList([nn.Conv2d(16, 64, 1, 1, 0) for _ in range(num_iterations)])


    def forward(self, x, k, sf):
        '''
        x: tensor, NxCxWxH
        k: tensor, Nx(1,3)xwxh
        sf: integer, 1
        sigma: tensor, Nx1x1x1
        '''
        b,c,h,w = k.shape
        k = self.kernelnet(k)
        x = self.conv1(x)

        k_1 = self.convs[0](k)
        k_1 = k_1.view(b, 64, h, w)
        x = self.d(x, k_1, sf)
        x = self.p(x)
        for i in range(1, self.num_iterations):
            k_ = k
            k_ = self.convs[i](k_)
            k_ = k_.view(b, 64, h, w)
            x = self.d(x, k_, 1)
            x = self.p(x)
        x = self.conv2(x)
        return x
