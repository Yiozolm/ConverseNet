import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from models.basicblock import ConvReverse2d, Converse_Block, ResidualBlock


class MSRResNet(nn.Module):
    '''
    modified SRResNet
    '''
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_body = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(nf, nf * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hres = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv_first(x))
        x1 = self.conv_body(x1)

        if self.upscale == 4:
            x1 = self.lrelu(self.pixel_shuffle(self.upconv1(x1)))
            x1 = self.lrelu(self.pixel_shuffle(self.upconv2(x1)))
        elif self.upscale in [2, 3]:
            x1 = self.lrelu(self.pixel_shuffle(self.upconv1(x1)))

        x1 = self.conv_last(self.lrelu(self.conv_hres(x1)))
        x = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        return x1 + x



class Converse_Block_MSRResNet(nn.Module):
    '''
    modified MSRResNet
       replace the residualblock with reverse block
    '''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(Converse_Block_MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 1, 1, 0, bias=True)
        self.conv_body = nn.Sequential(*[Converse_Block(nf, nf, 5, 1, 4, padding_mode='replicate', eps=1e-5) for _ in range(nb)])

        self.upconv1 = Converse_Block(nf, nf, 5, 1, 4, padding_mode='replicate', eps=1e-5)
        self.up1 = ConvReverse2d(nf, nf, 2, 2, 2, padding_mode='replicate', eps=1e-3)
        self.upconv2 = Converse_Block(nf, nf, 5, 1, 4, padding_mode='replicate', eps=1e-5)
        self.up2 = ConvReverse2d(nf, nf, 2, 2, 2, padding_mode='replicate', eps=1e-3)
        self.conv_hres = Converse_Block(nf, nf, 5, 1, 4, padding_mode='replicate', eps=1e-5)
        self.conv_last = nn.Conv2d(nf, out_nc, 1, 1, 0, bias=False)
        # activation function
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.gelu(self.conv_first(x))
        x1 = self.conv_body(x1)
        x1 = self.gelu(self.up1(self.upconv1(x1)))
        x1 = self.gelu(self.up2(self.upconv2(x1)))
        x1 = self.conv_last(self.gelu(self.conv_hres(x1)))
        x = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        return x1 + x




class Converse_MSRResNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(Converse_MSRResNet, self).__init__()
        '''
            upsample with Converse2D operator
        '''      

        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv_body = nn.Sequential(*[ResidualBlock(nf) for _ in range(nb)])
       
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1) 
        self.reverse1 = ConvReverse2d(nf, nf, 2, scale=2)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.reverse2 = ConvReverse2d(nf, nf, 2, scale=2)

        self.conv_hres = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.gelu(self.conv_first(x))
        x1 = self.conv_body(x1)
        x1 = self.gelu(self.reverse1(self.conv1(x1)))
        x1 = self.gelu(self.reverse2(self.conv2(x1)))
        x1 = self.conv_last(self.gelu(self.conv_hres(x1)))
        x = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        return x1 + x
    



