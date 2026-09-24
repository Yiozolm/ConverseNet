"""Run the opt-in full-spectrum spectral contracts with a fresh result name."""
from pathlib import Path
import runpy
import sys

if __name__ == '__main__':
    package = Path(__file__).resolve().parents[1] / 'experiments' / 'training_full_spectrum_fusion'
    sys.path.insert(0, str(package))
    runpy.run_path(str(package / 'contracts.py'), run_name='__main__')
