"""Preserve repeated native/GEMM/FP64 results without changing any budget."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from test_pointwise_wgrad import fixture, evaluate
from numerical_policy import comparison

path = Path(sys.argv[1])
if path.exists():
    raise RuntimeError('New report path required')
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
raw = fixture(4101)
rows = []
for deterministic in (False, True):
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic)
    for repeat in range(8):
        high, _ = evaluate(raw, (True, True, True), dtype=torch.float64)
        base, _ = evaluate(raw, (True, True, True))
        candidate, _ = evaluate(raw, (True, True, True), production=True)
        result = comparison(candidate['dweight'], base['dweight'], high['dweight'])
        hashes = {name: hashlib.sha256(values['dweight'].detach().cpu().numpy().tobytes()).hexdigest()
                  for name, values in [('baseline', base), ('candidate', candidate), ('fp64', high)]}
        row = dict(deterministic=deterministic, repeat=repeat, hashes=hashes, check=result)
        rows.append(row)
        print(json.dumps(row), flush=True)
path.write_text(json.dumps(dict(rows=rows, torch=str(torch.__version__), cuda=torch.version.cuda,
                               gpu=torch.cuda.get_device_name()), indent=2))
