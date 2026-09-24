from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(os.environ.get('CONVERSE_S2_ARTIFACTS',ROOT/'artifacts/training_scale2_20260921')).resolve()
HERE.mkdir(parents=True,exist_ok=True)
