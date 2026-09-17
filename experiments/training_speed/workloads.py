"""Per-instance experimental routing; repository model implementations stay intact."""
import sys
import types

import torch
from torch import nn
from torch.nn import functional as F

from extension import ROOT
sys.path.insert(0, str(ROOT))
from models.util_converse import Converse2D, ConverseBlock
from models.converse_usrnet import ConverseUSRNet, ConvReverseDataNet


def backend(name):
    def call(x, prior, weight, bias, scale, eps):
        if name == 'checkout':
            return torch.ops.training_checkout_converse2d.forward(x, prior, weight, bias, scale, eps, 'v7')
        return torch.ops.training_speed.forward(x, prior, weight, bias, scale, eps, False, name == 's1')
    return call


def route(model, name):
    op = backend(name)
    def layer_forward(self, x):
        if self.padding:
            x = F.pad(x, (self.padding,) * 4, mode=self.padding_mode)
        prior = x if self.scale == 1 else F.interpolate(x, scale_factor=self.scale, mode='nearest')
        y = op(x, prior, self.weight, self.bias, self.scale, self.eps)
        if self.padding:
            p = self.padding * self.scale
            y = y[..., p:-p, p:-p]
        return y

    def data_forward(self, x, k, sf, padding=0, padding_mode='circular'):
        if padding:
            x = F.pad(x, (padding,) * 4, mode=padding_mode)
        prior = x if sf == 1 else F.interpolate(x, scale_factor=sf, mode='nearest')
        y = op(x, prior, k, self.alpha, sf, self.eps)
        if padding:
            p = padding * sf
            y = y[..., p:-p, p:-p]
        return y

    for layer in model.modules():
        if isinstance(layer, Converse2D):
            layer.forward = types.MethodType(layer_forward, layer)
        elif isinstance(layer, ConvReverseDataNet):
            layer.forward = types.MethodType(data_forward, layer)
    if isinstance(model, OperatorTrain):
        model.op = op
    return model


class OperatorTrain(nn.Module):
    """A 1x1 producer ensures the solver must propagate input gradients."""
    def __init__(self, channels, scale):
        super().__init__()
        self.stem = nn.Conv2d(channels, channels, 1)
        self.weight = nn.Parameter(torch.randn(1, channels, 3, 3).flatten(2).softmax(-1).reshape(1, channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = scale
        self.op = backend('fused')

    def forward(self, x):
        x = self.stem(x)
        prior = x if self.scale == 1 else F.interpolate(x, scale_factor=self.scale, mode='nearest')
        return self.op(x, prior, self.weight, self.bias, self.scale, 1e-3)


class USRTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = ConverseUSRNet(num_iterations=2, num_blocks=1, backend='pytorch')
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name.endswith(('alpha1', 'alpha2')):
                    parameter.fill_(.1)

    def forward(self, x, kernel):
        return self.model(x, kernel, 2)


def workload(kind, shape, scale=1, accumulation=1):
    B, C, H, W = shape
    if kind == 'operator':
        model = OperatorTrain(C, scale)
    elif kind == 'block':
        model = ConverseBlock(C, C, scale=1)
        assert scale == 1
    elif kind == 'usrnet':
        model = USRTrain()
        assert C == 3 and scale == 2
    else:
        raise ValueError(kind)
    batches, targets = [], []
    for _ in range(accumulation):
        x = torch.rand(shape, device='cuda') * .2
        if kind == 'usrnet':
            kernel = torch.rand(B, 1, 7, 7, device='cuda')
            kernel = kernel / kernel.sum((-2,-1), keepdim=True)
            batches.append((x, kernel))
        else:
            batches.append((x,))
        targets.append(torch.rand(B, C, H * scale, W * scale, device='cuda') * .2)
    return model.cuda(), batches, targets
