"""Run existing first/higher-order CUDA contracts against frozen forced-s1.

The alias used by the tests is replaced only inside this process. No production
sources, test tolerances, compiler settings or saved historical evidence change.
"""
import argparse
import importlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import study_training_small_s1 as study


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--warm-builds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite evidence")
    saved = json.loads(args.warm_builds.read_text())
    args.snapshot = Path(saved["settings"]["snapshot"])
    ops, identity = study.load_ops(args)
    import torch
    import extension_loader
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    names = ("test_training_fusion", "test_training_refinements")
    with patch.object(extension_loader, "load_extension"), patch.object(torch.ops, "converse2d", ops["candidate"]):
        modules = [importlib.import_module(name) for name in names]
        suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromModule(m) for m in modules)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
        failures=[(str(t), msg) for t, msg in result.failures], errors=[(str(t), msg) for t, msg in result.errors],
        skipped=[(str(t), msg) for t, msg in result.skipped], identity=identity,
        tests_sha256={m.__name__: study.sha(Path(m.__file__)) for m in modules},
        script_sha256=study.sha(Path(__file__)),
        scope="Actual forced s1 kernels, inherited test budgets; not Python noninferiority or training quality proof")
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
