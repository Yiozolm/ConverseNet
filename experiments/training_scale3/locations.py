from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(os.environ.get('CONVERSE_S3_ARTIFACTS',ROOT/'artifacts/training_scale3_20260922')).resolve()
HERE.mkdir(parents=True,exist_ok=True)
