"""One warmed full-spectrum training call for independent NCU diagnostics."""
import argparse
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, required=True)
args = parser.parse_args()
sys.path[:0] = [str(args.root.resolve() / 'test'), str(args.root.resolve())]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
import torch
from extension_loader import load_extension

load_extension()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(41001)
x = torch.randn(4, 128, 100, 100, device='cuda', requires_grad=True)
weight = torch.rand(1, 128, 3, 3, device='cuda', requires_grad=True)
bias = torch.zeros(1, 128, 1, 1, device='cuda', requires_grad=True)
upstream = torch.randn_like(x)

def run():
    out = torch.ops.converse2d.forward(x, x, weight, bias, 1, 1e-5)
    return torch.autograd.grad(out, (x, weight, bias), upstream)

for _ in range(5):
    run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStart()
run()
torch.cuda.synchronize()
torch.cuda.cudart().cudaProfilerStop()
