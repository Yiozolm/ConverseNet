import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from models.util_converse import Converse2D, ConverseBlock, ResidualBlock


class MSRResNet(nn.Module):
    '''
    modified SRResNet
    '''
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True)
        self.conv_body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])

        # upsampling
        if self.upscale in [2, 3]:
            self.upconv1 = nn.Conv2d(num_features, num_features * (upscale ** 2), 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(upscale)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(num_features, num_features * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(num_features, num_features * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hres = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1, bias=True)

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



class ConverseMSRResNet(nn.Module):
    '''
    modified MSRResNet
       replace the residualblock with reverse block
    '''

    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=16, upscale=4):
        super(ConverseMSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_channels, num_features, 1, 1, 0, bias=True)
        self.conv_body = nn.Sequential(*[ConverseBlock(num_features, num_features, 5, 1, 4, padding_mode='replicate', eps=1e-5) for _ in range(num_blocks)])

        self.upconv1 = ConverseBlock(num_features, num_features, 5, 1, 4, padding_mode='replicate', eps=1e-5)
        self.up1 = Converse2D(num_features, num_features, 2, 2, 2, padding_mode='replicate', eps=1e-3)
        self.upconv2 = ConverseBlock(num_features, num_features, 5, 1, 4, padding_mode='replicate', eps=1e-5)
        self.up2 = Converse2D(num_features, num_features, 2, 2, 2, padding_mode='replicate', eps=1e-3)
        self.conv_hres = ConverseBlock(num_features, num_features, 5, 1, 4, padding_mode='replicate', eps=1e-5)
        self.conv_last = nn.Conv2d(num_features, out_channels, 1, 1, 0, bias=False)
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




class ConverseUpMSRResNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, num_blocks=16, upscale=4):
        super(ConverseUpMSRResNet, self).__init__()
        '''
            upsample with Converse2D operator
        '''      

        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=True)
        self.conv_body = nn.Sequential(*[ResidualBlock(num_features) for _ in range(num_blocks)])
       
        self.conv1 = nn.Conv2d(num_features, num_features, 3, 1, 1) 
        self.reverse1 = Converse2D(num_features, num_features, 2, scale=2)
        self.conv2 = nn.Conv2d(num_features, num_features, 3, 1, 1)
        self.reverse2 = Converse2D(num_features, num_features, 2, scale=2)

        self.conv_hres = nn.Conv2d(num_features, num_features, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(num_features, out_channels, 3, 1, 1, bias=True)

        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.gelu(self.conv_first(x))
        x1 = self.conv_body(x1)
        x1 = self.gelu(self.reverse1(self.conv1(x1)))
        x1 = self.gelu(self.reverse2(self.conv2(x1)))
        x1 = self.conv_last(self.gelu(self.conv_hres(x1)))
        x = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)

        return x1 + x
    



