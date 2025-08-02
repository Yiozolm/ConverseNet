import torch
import torch.nn as nn
from models.basicblock import sequential, Converse_Block, Converse_Block_alpha
# from utils import utils_image as util
import torch.fft
import torch.nn.init as init
import torch.nn.functional as F

"""
# --------------------------------------------
# Prior module; denoiser
# x_k = P(z_k)
# --------------------------------------------
"""
class ConverseNet(nn.Module):
    def __init__(self, in_nc=64, nb=7, kernel=3, scale=1, padding=2, padding_mode="circular"):
        super(ConverseNet, self).__init__()
        self.m_body  = sequential(*[Converse_Block(in_nc, in_nc, kernel, scale, padding, padding_mode) for _ in range(nb)])
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.m_body(x)
        return x
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


class ConverseNet_alpha(nn.Module):
    def __init__(self, in_nc=64, nb=7, kernel=3, scale=1, padding=2, padding_mode="circular"):
        super(ConverseNet_alpha, self).__init__()
        self.m_body  = sequential(*[Converse_Block_alpha(in_nc, in_nc, kernel, scale, padding, padding_mode) for _ in range(nb)])
        self.apply(self._init_weights)
    def forward(self, x):
        x = self.m_body(x)
        return x
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)



"""
# --------------------------------------------
# Data module, closed-form solution
# --------------------------------------------
"""
class ConvReverse2d_DataNet(nn.Module):
    def __init__(self,eps=1e-3):
        super(ConvReverse2d_DataNet, self).__init__()
        '''
        padding_mode: default:reflect, {'reflect', 'replicate', 'circular'}
        '''
        self.alpha = nn.Parameter(torch.zeros(1, 64, 1, 1))
        self.eps = eps
    def forward(self, x, k, sf, padding = 0, padding_mode = 'circular'):

        if padding > 0:
            x = nn.functional.pad(x, pad=[padding, padding, padding, padding], mode=padding_mode, value=0)
        self.biaseps = torch.sigmoid(self.alpha-9.0) + self.eps
        _, _, h, w = x.shape
        STy = self.upsample(x, sf)
        if sf != 1:
            x = nn.functional.interpolate(x, scale_factor=sf, mode='nearest')
        
        FB = self.p2o(k, (h*sf, w*sf))
        FBC = torch.conj(FB)
        F2B = torch.pow(torch.abs(FB), 2)

        FBFy = FBC*torch.fft.fftn(STy, dim=(-2, -1))
        FR = FBFy + torch.fft.fftn(self.biaseps*x, dim=(-2,-1))
        x1 = FB.mul(FR)
        FBR = torch.mean(self.splits(x1, sf), dim=-1, keepdim=False)
        invW = torch.mean(self.splits(F2B, sf), dim=-1, keepdim=False)
        invWBR = FBR.div(invW + self.biaseps)
        FCBinvWBR = FBC*invWBR.repeat(1, 1, sf, sf)
        FX = (FR-FCBinvWBR)/self.biaseps
        out = torch.real(torch.fft.ifftn(FX, dim=(-2, -1)))

        if padding > 0:
            out = out[..., padding*sf:-padding*sf, padding*sf:-padding*sf]

        return out
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
        k = k.to(torch.float)
        b,_,_,_ = k.shape
        k = k.view(b, -1) # flatten
        # fully connected
        k = self.gelu(self.fc1(k))
        k = self.gelu(self.fc2(k))
        k = self.fc3(k)
        k = k.view(b, 16, self.kernel_size, self.kernel_size)
        return k


"""
# --------------------------------------------
# main USRNet
# deep unfolding super-resolution network
# --------------------------------------------
"""
class USRNet(nn.Module):
    def __init__(self, n_iter=5, in_nc=64, nb=7):
        super(USRNet, self).__init__()

        self.d = ConvReverse2d_DataNet()
        self.p = ConverseNet_alpha(in_nc=in_nc, nb=nb)
        self.conv1 = nn.Conv2d(3, 64, 1, 1, 0)
        self.conv2 = nn.Conv2d(64, 3, 1, 1, 0)
        self.kernelnet = KernelNet()
        self.n = n_iter
        
        self.convs = nn.ModuleList([nn.Conv2d(16, 64, 1, 1, 0) for _ in range(n_iter)])


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
        for i in range(1, self.n):
            k_ = k
            k_ = self.convs[i](k_)
            k_ = k_.view(b, 64, h, w)
            x = self.d(x, k_, 1)
            x = self.p(x)
        x = self.conv2(x)
        return x
