"""Thin full-model accuracy audit of four full-C2C shared-s1 candidates.

    ./experiments/training_speed/run.ps1 test/compare_full_fft_shared_s1_model.py

Reuse compare_shared_s1_accuracy_to_python unchanged: the archived seed9214
fixture, one fixed FP64 diagnostic oracle, original Python FP32 error baseline,
all 136 output/gradient tensors, raw max_abs/relative-L2 ratios and rounding
floors. Prior pointwise failures remain diagnostic, never a collection veto.

Replace all 39 shared-input s1 calls only; the initial s2 call stays production.
Candidates are full_h32/full_h64_cast/full_h64_product/full_h64_lambda64 from
probe_shared_s1_full_fft. The residual-C route is explicitly excluded. No
production/training dependency edits, timing, parameter cache or quality claim.
"""
import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare_shared_s1_accuracy_to_python as audit

BASE_SHA256 = "a7b907c919ee8c218c590e1f2f81cdb1cd541b067847a8872f3f40a349830145"
FULL_MODES = ("full_h32", "full_h64_cast", "full_h64_product", "full_h64_lambda64")


@contextmanager
def full_fft_context(mode):
    import torch
    from models import util_converse
    from probe_shared_s1_full_fft import spatial

    original = torch.ops.converse2d.forward
    stats = dict(total_calls=0, eligible_shared_s1=0, transfer_calls=0,
                 original_calls=0, scales={}, transferred_kernel_shapes={}, full_fft_variant=mode)

    def forward(x, prior, weight, bias, scale, eps, variant="v7"):
        stats["total_calls"] += 1
        stats["scales"][str(scale)] = stats["scales"].get(str(scale), 0) + 1
        eligible = (scale == 1 and prior is x and x.is_cuda and x.dtype == torch.float32 and variant == "v7"
                    and not torch.is_autocast_enabled(x.device.type))
        stats["eligible_shared_s1"] += int(eligible)
        if not eligible:
            stats["original_calls"] += 1
            return original(x, prior, weight, bias, scale, eps, variant)
        result = spatial(x, prior, weight, bias, scale, eps, mode)
        if result.dtype != torch.float32:
            raise RuntimeError("The full-FFT candidate did not return FP32")
        stats["transfer_calls"] += 1
        key = "x".join(str(value) for value in weight.shape[-2:])
        stats["transferred_kernel_shapes"][key] = stats["transferred_kernel_shapes"].get(key, 0) + 1
        return result

    with ExitStack() as stack:
        stack.enter_context(patch.object(torch.ops.converse2d, "forward", forward))
        stack.enter_context(patch.object(util_converse, "converse2d_CUDA", forward))
        yield stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", action="append", choices=FULL_MODES)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/shared_s1_full_fft_model.json")
    args = parser.parse_args()
    base_path = ROOT / "test/compare_shared_s1_accuracy_to_python.py"
    if audit.frozen.sha256(base_path) != BASE_SHA256:
        raise RuntimeError("The original full-model accuracy audit changed")
    selected = tuple(dict.fromkeys(args.mode or FULL_MODES))
    argv = [str(Path(__file__).resolve()), "--output", str(args.output)]
    for mode in selected:
        argv.extend(("--mode", mode))
    original_context, original_run = audit.frozen.route_context, audit.frozen.run

    @contextmanager
    def selected_context(mode):
        context = full_fft_context(mode) if mode in FULL_MODES else original_context(mode)
        with context as stats:
            yield stats

    def annotated_run(report, save):
        helper = ROOT / "test/probe_shared_s1_full_fft.py"
        report["scope"] = __doc__
        report["experiment_adapter"] = dict(path=str(Path(__file__).resolve()),
            sha256=audit.frozen.sha256(__file__), base_audit_sha256=BASE_SHA256,
            helper_sha256=audit.frozen.sha256(helper), candidates=list(selected),
            excluded_candidates=["full_c64_residual"], shared_s1_replacements=39, original_s2_calls=1)
        original_run(report, save)
        for name in ("test/compare_full_fft_shared_s1_model.py", "test/probe_shared_s1_full_fft.py"):
            report["source_sha256"][name] = audit.frozen.sha256(ROOT / name)
        if audit.frozen.sha256(base_path) != BASE_SHA256:
            raise RuntimeError("The reused audit source changed during this run")

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(audit, "CANDIDATES", FULL_MODES))
        stack.enter_context(patch.object(audit.frozen, "route_context", selected_context))
        stack.enter_context(patch.object(audit.frozen, "run", annotated_run))
        return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
