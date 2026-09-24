"""Capture deterministic pre/post cleanup evidence using either checkout.

This verifies preservation of an existing implementation, not convergence.
Run baseline and candidate in separate processes (TORCH_LIBRARY is global).
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--compare', type=Path)
args = parser.parse_args()
sys.path[:0] = [str(args.root / 'test'), str(args.root)]
import torch
from extension_loader import load_extension, production_source_hashes
load_extension(verbose=True)
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_num_threads(4)

def digest(value):
    value = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
    assert torch.isfinite(value).all(), 'Nonfinite evidence'
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()

records = {}
for s in (1, 2, 3, 4):
    for kb, kc in ((1, 1), (1, 3), (2, 1), (2, 3)):
        torch.manual_seed(193 + s + kb + kc)
        x = torch.randn(2, 3, 7, 9, device='cuda')
        p = torch.randn(2, 3, 7*s, 9*s, device='cuda')
        k = torch.rand(kb, kc, 3, 3, device='cuda') / 9
        b = torch.zeros(1, 3, 1, 1, device='cuda')
        for dynamic in (False, True):
            weight = k.detach().requires_grad_().clone() if dynamic else k
            assert weight.is_leaf != dynamic
            for mode in (torch.no_grad, torch.inference_mode):
                with mode():
                    for call in range(2):
                        out = torch.ops.converse2d.forward(x, p, weight, b, s)
                        records[f'inference/{s}/{kb}/{kc}/{dynamic}/{mode.__name__}/{call}'] = digest(out)
        torch.ops.converse2d.clear_cache()
print('Captured inference fixtures', flush=True)

from models.converse_usrnet import ConverseUSRNet
from PIL import Image
import numpy as np
image = torch.from_numpy(np.array(Image.open(args.root/'utils/test.png').convert('RGB'), copy=True)).permute(2, 0, 1).float()/255
for seed in (17, 29, 43):
    torch.manual_seed(seed)
    model = ConverseUSRNet(backend='cuda').cuda()
    model.load_state_dict(torch.load(args.root/'model_zoo/converse_usrnet.pth', weights_only=True, map_location='cuda'), strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    gen = torch.Generator().manual_seed(seed)
    for step in range(3):
        h = int(torch.randint(0, image.shape[1]-24+1, (1,), generator=gen))
        w = int(torch.randint(0, image.shape[2]-24+1, (1,), generator=gen))
        target = image[:, h:h+24, w:w+24].unsqueeze(0).cuda()
        x = torch.nn.functional.interpolate(target, scale_factor=1/3, mode='bicubic', align_corners=False)
        k = torch.rand(1, 1, 7, 7, generator=gen).cuda()
        k /= k.sum()
        optimizer.zero_grad(set_to_none=True)
        output = model(x, k, 3)
        loss = (output-target).square().mean()
        loss.backward()
        key = f'train/{seed}/{step}'
        records[key+'/output'] = digest(output)
        records[key+'/loss'] = digest(loss)
        for name, p in model.named_parameters():
            assert p.grad is not None, name
            records[key+'/grad/'+name] = digest(p.grad)
        optimizer.step()
        for name, p in model.named_parameters():
            records[key+'/param/'+name] = digest(p)
            for slot, value in optimizer.state[p].items():
                records[key+'/adam/'+name+'/'+slot] = digest(value)
        with torch.inference_mode():
            records[key+'/eval'] = digest(model(x, k, 3))
        print(f'Full pretrained USRNet seed={seed} step={step} loss={loss.item():.8g}', flush=True)
    del optimizer, model
    torch.ops.converse2d.clear_cache()
    torch.cuda.empty_cache()

data = dict(torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            sources=production_source_hashes(), records=records,
            scope='Full pretrained USRNet, 3 seeds x 3 Adam updates, real-image HR24 crops; no convergence claim')
if args.compare:
    baseline = json.loads(args.compare.read_text())
    failures = [k for k in set(records) | set(baseline['records']) if records.get(k) != baseline['records'].get(k)]
    data['comparison'] = dict(baseline=str(args.compare), checked=len(records), failures=failures)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(data, indent=2), encoding='utf-8')
if args.compare:
    assert not failures, f'{len(failures)} mismatches: {failures[:10]}'
print(f'Saved {len(records)} tensor hashes to {args.output}', flush=True)
