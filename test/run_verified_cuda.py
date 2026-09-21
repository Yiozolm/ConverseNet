"""Run a repository experiment with a source-verified existing CUDA extension.

No compiler/version probe is necessary for a warm build. Stale or missing
source/header/PyTorch/binary hashes fail normally; build extension_loader first.
"""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
if len(sys.argv) < 2:
    raise SystemExit("Usage: python test/run_verified_cuda.py path/to/experiment.py [args ...]")
script = Path(sys.argv[1]).resolve()
if not script.is_relative_to(ROOT) or not script.is_file() or script.suffix != ".py":
    raise SystemExit("The experiment must be an existing Python file inside this repository")
sys.path[:0] = [str(script.parent), str(ROOT / "test"), str(ROOT)]
from extension_loader import load_extension
previous = os.environ.get("CONVERSE2D_SKIP_BUILD")
os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
try:
    load_extension()
finally:
    if previous is None:
        os.environ.pop("CONVERSE2D_SKIP_BUILD", None)
    else:
        os.environ["CONVERSE2D_SKIP_BUILD"] = previous
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name="__main__")
