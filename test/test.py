import os, sys, time, json, argparse, pathlib, statistics, subprocess, csv, re
import torch
import torch.nn.functional as F

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from models.util_converse import Converse2D  

Model = Converse2D(in_channels=3, out_channels=3, kernel_size=3, padding=1, scale=1,backend='pytorch')
x = torch.randn(1, 3, 256, 256)
y = Model(x)

