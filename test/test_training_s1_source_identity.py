"""CPU-only regression for the refactored s1 ablation's setup identity check.

Exercise the CLI through its real source verification, replacing only builds
and GPU dispatch. No CUDA initialization, compiler or timed workload is needed.
"""
import copy
import hashlib
import os
import sys
import unittest
from unittest.mock import patch

import training_s1_ablation as ablation


class ReachedDispatch(Exception):
    pass


class AblationSourceIdentity(unittest.TestCase):
    def setUp(self):
        self.exported = ablation.amalgamation_hashes()
        self.production = ablation.production_source_hashes()
        self.baseline = dict(source_sha256=self.exported.copy(),
                             production_source_sha256=self.production.copy())

    def invoke(self, baseline=None, production_after=None):
        production_after = self.production if production_after is None else production_after
        dispatch_calls = 0
        def verified_half_then_stop(*args, **kwargs):
            nonlocal dispatch_calls
            self.assertEqual(kwargs.get("expected"), "half")
            dispatch_calls += 1
            if dispatch_calls == 1:
                return {str(scale): ["SpectralSolve"] for scale in (1, 2, 3)}
            raise ReachedDispatch
        with (patch.object(sys, "argv", ["training_s1_ablation.py", "--quick"]),
              patch.dict(os.environ, {"CONVERSE2D_CPU_ONLY": "0", "CONVERSE2D_BACKEND": "cuda"}),
              patch.object(ablation.torch.cuda, "is_available", return_value=True),
              patch.object(ablation, "load_extension"),
              patch.object(ablation, "load_scale1_disabled",
                           return_value=(object(), baseline or self.baseline)),
              patch.object(ablation, "production_source_hashes",
                           side_effect=[self.production, production_after]),
              patch.object(ablation.common, "verify_fused_dispatch",
                           side_effect=verified_half_then_stop)):
            ablation.main()

    def test_full_default_is_rejected_before_building_half_ablation(self):
        with (patch.object(sys, "argv", ["training_s1_ablation.py", "--quick"]),
              patch.dict(os.environ, {"CONVERSE2D_CPU_ONLY": "0", "CONVERSE2D_BACKEND": "cuda"}),
              patch.object(ablation.torch.cuda, "is_available", return_value=True),
              patch.object(ablation, "load_extension"),
              patch.object(ablation, "load_scale1_disabled") as isolated_build,
              patch.object(ablation.common, "verify_fused_dispatch",
                           side_effect=RuntimeError("expected half spectrum fused training, observed FullSolve"))):
            with self.assertRaisesRegex(RuntimeError, "expected half"):
                ablation.main()
            isolated_build.assert_not_called()

    def test_cli_accepts_unchanged_exports_that_differ_from_facades(self):
        # This is the actual refactored layout: a facade hash must NOT be used
        # as the identity of the self-contained source compiled by the loader.
        for name in ("converse2d.cpp", "converse2d_training.cu", "converse2d_kernels.cu"):
            path = ablation.ROOT / ablation.SOURCE / name
            self.assertNotEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
                                self.exported[f"{ablation.SOURCE}/{name}"])
        with self.assertRaises(ReachedDispatch):
            self.invoke()

    def test_cli_rejects_changed_transitive_production_header(self):
        changed = self.production.copy()
        header = next(name for name in changed if name.endswith("training/detail/math.cuh"))
        changed[header] = "changed"
        with self.assertRaisesRegex(RuntimeError, "Source changed during ablation setup"):
            self.invoke(production_after=changed)

    def test_cli_rejects_different_derived_build_source(self):
        baseline = copy.deepcopy(self.baseline)
        baseline["source_sha256"][f"{ablation.SOURCE}/converse2d_training.cu"] = "different build"
        with self.assertRaisesRegex(RuntimeError, "Source changed during ablation setup"):
            self.invoke(baseline=baseline)


if __name__ == "__main__":
    unittest.main(verbosity=2)
