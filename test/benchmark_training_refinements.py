"""Paired before/after refinement benchmark; frozen models preserve forward scope.

The compatible result schema calls frozen-before 'dev' and working-tree-after
'current'. These labels do not refer to the original dev commit in this study.
Profile in a separate process with profile_training_refinements.py.
"""
import argparse
import datetime
import json
import os
import sys

import torch
from torch import nn

import benchmark_fp32_training as common
from extension_loader import load_extension
from fp32_training_baseline import current_manifest
from training_refinement_baseline import (
    ROOT, DEFAULT_SNAPSHOT, load_baseline, load_frozen_models, sha256,
)


class USRWrapper(nn.Module):
    def __init__(self, model_class, scale, iterations=2, blocks=1):
        super().__init__()
        self.model = model_class(num_iterations=iterations, num_blocks=blocks, backend="cuda")
        self.scale = scale
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name.endswith(("alpha1", "alpha2")):
                    parameter.fill_(0.1)

    def forward(self, x, kernel):
        return self.model(x, kernel, self.scale)


def load_variants(snapshot, verbose=False):
    if os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        raise RuntimeError("Unset CONVERSE2D_CPU_ONLY for CUDA training measurements")
    if os.environ.get("CONVERSE2D_BACKEND", "").lower() not in ("", "auto", "cuda"):
        raise RuntimeError("CONVERSE2D_BACKEND must not select the reference backend")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    skip_build = os.environ.pop("CONVERSE2D_SKIP_BUILD", None)
    try:
        load_extension(verbose=verbose)
    finally:
        if skip_build is not None:
            os.environ["CONVERSE2D_SKIP_BUILD"] = skip_build
    current = torch.ops.converse2d
    before, baseline = load_baseline(snapshot, verbose=verbose)
    dispatch = {"current": common.verify_fused_dispatch(current),
                "dev": common.verify_fused_dispatch(before)}
    common.release_cuda()
    sys.path.insert(0, str(ROOT))
    frozen = load_frozen_models(snapshot)
    torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    return dict(dev=before, current=current), baseline, frozen, dispatch


def model_templates(config, frozen, reuse_spectra=False):
    kind, shape, scale, accumulation = config
    if kind == "usrnet_full":
        from models.converse_usrnet import ConverseUSRNet
        current = USRWrapper(ConverseUSRNet, scale, iterations=5, blocks=7)
        # Reuse only the CPU input factory; the reduced model is discarded.
        _, batches, targets = common.workload("usrnet", shape, scale, accumulation)
    else:
        current, batches, targets = common.workload(kind, shape, scale, accumulation)
    if kind == "operator":
        before = common.OperatorTrain(shape[1], scale)
    elif kind == "block":
        before = frozen.util_converse.ConverseBlock(shape[1], shape[1])
        for layer in before.modules():
            if isinstance(layer, frozen.util_converse.Converse2D):
                layer.backend = "cuda"
    elif kind in ("usrnet", "usrnet_full"):
        before = USRWrapper(frozen.converse_usrnet.ConverseUSRNet, scale,
                            iterations=5 if kind == "usrnet_full" else 2,
                            blocks=7 if kind == "usrnet_full" else 1)
    else:
        raise ValueError(kind)
    before.load_state_dict(current.state_dict(), strict=True)
    for layer in current.modules():
        if hasattr(layer, "reuse_training_spectra"):
            layer.reuse_training_spectra = reuse_spectra
    return dict(dev=before, current=current), batches, targets


def training_case(config, variants, frozen, args):
    from models.util_converse import Converse2D
    kind, shape, scale, accumulation = config
    templates, batches, targets = model_templates(config, frozen, getattr(args, "reuse_spectra", False))
    states, activity = {}, {}
    for name, ops in variants.items():
        template = templates[name]
        solver_weights = (["weight"] if kind == "operator" else
                          [f"{module_name}.weight" for module_name, layer in template.named_modules()
                           if isinstance(layer, (Converse2D, frozen.util_converse.Converse2D))])
        with common.production_namespace(ops):
            runner = common.TrainingRunner(template, batches, targets)
            states[name] = []
            for _ in range(4):
                runner.step()
                states[name].append(runner.snapshot())
                runner.perturb_batch()
            final = states[name][-1]
            updated = [key for key, value in template.named_parameters()
                       if not torch.equal(final[f"param/{key}"], value.detach())]
            nonzero = [key[5:] for key, value in final.items()
                       if key.startswith("grad/") and torch.count_nonzero(value).item()]
            if not set(solver_weights).intersection(updated) or not set(solver_weights).intersection(nonzero):
                raise AssertionError(f"{name}: solver weights did not train")
            if kind.startswith("usrnet"):
                if "model.d.alpha" not in set(updated).intersection(nonzero):
                    raise AssertionError(f"{name}: data regularization did not train")
                if not any("kernelnet." in key for key in set(updated).intersection(nonzero)):
                    raise AssertionError(f"{name}: kernel producer did not train")
            activity[name] = dict(updated_parameters=updated, nonzero_gradients=nonzero)
            del runner
        common.release_cuda()
    checks = [common.compare_tensors(current, before)
              for current, before in zip(states["current"], states["dev"])]

    def benchmark(name):
        with common.production_namespace(variants[name]):
            runner = common.TrainingRunner(templates[name], batches, targets)
            for _ in range(args.warmup):
                runner.step()
            runner.reset()
            result = common.sample(runner.step, args.iters)
            if not all(torch.isfinite(value).all() for value in runner.snapshot().values()):
                raise AssertionError(f"{name}: nonfinite timed trajectory")
            return result

    result = dict(kind=kind, shape=shape, scale=scale, accumulation=accumulation,
                  trajectory_checks=checks, activity=activity,
                  scope="FP32 eager full SGD step; frozen-before/current-after model forwards; identical initial state and device inputs",
                  timing=common.paired(benchmark, args))
    print(json.dumps(dict(training=config, **result["timing"]["medians"])), flush=True)
    return result


def cases():
    result = []
    for batch in (1, 4):
        for scale in (1, 2, 3):
            result.append((f"op-b{batch}-256-s{scale}",
                           ("operator", (batch, 32, 256, 256), scale, 1)))
    result += [("usrnet-b8-s3", ("usrnet", (8, 3, 16, 20), 3, 1)),
               ("usrnet-b16-s3", ("usrnet", (16, 3, 16, 20), 3, 1)),
               ("usrnet-b1-s2", ("usrnet", (1, 3, 16, 20), 2, 1)),
               ("block-b16", ("block", (16, 16, 32, 40), 1, 1))]
    return result


EXPLICIT_CASES = {
    "usrnet-default-tiny": ("usrnet_full", (1, 3, 8, 8), 2, 1),
}


def report_metadata(args, baseline, dispatch):
    hashes = current_manifest()
    hashes["models/converse_core.py"] = sha256(ROOT / "models/converse_core.py")
    return dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                current_source_sha256=hashes, baseline=baseline, current_fused_dispatch=dispatch,
                variant_labels=dict(dev="frozen refinement before", current="working-tree refinement after"),
                model_scope="Before uses manifest-verified frozen Python model forwards; after uses current Python models; strict shared state_dict",
                benchmark_sha256={name: sha256(ROOT / "test" / name) for name in
                                  ("benchmark_fp32_training.py", "benchmark_training_refinements.py",
                                   "training_refinement_baseline.py", "profile_training_refinements.py")},
                settings=vars(args), operators=[], training=[],
                memory_scope="Single variant PyTorch allocator totals including model, optimizer and inputs; driver/library allocations excluded",
                timing_scope="Warm FP32 eager complete steps; no input transfer, AMP, graphs or profiler",
                validation_scope="Original paired numerical and four-step SGD checks; independent FP64 and convergence evidence required separately")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", choices=("all", "operators", "training"), default="all")
    parser.add_argument("--case", action="append",
                        choices=[*[name for name, _ in cases()], *EXPLICIT_CASES])
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--reuse-spectra", action="store_true",
                        help="Evaluate explicit per-forward reuse candidate; default production leaves it disabled")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", default="artifacts/training_refinements/benchmark.json")
    args = parser.parse_args()
    if min(args.iters, args.rounds, args.warmup) < 1:
        parser.error("iters, rounds and warmup must be positive")
    variants, baseline, frozen, dispatch = load_variants(args.snapshot, args.verbose_build)
    report = report_metadata(args, baseline, dispatch)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = [(name, config) for name, config in cases() if not args.case or name in args.case]
    selected += [(name, config) for name, config in EXPLICIT_CASES.items()
                 if args.case and name in args.case]
    if args.quick:
        selected = selected[:1]
    for name, config in selected:
        kind, shape, scale, _ = config
        if kind == "operator" and args.mode in ("all", "operators"):
            row = common.operator_case(shape, scale, variants, args)
            row["case"] = name
            report["operators"].append(row)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if args.mode in ("all", "training"):
            row = training_case(config, variants, frozen, args)
            row["case"] = name
            if kind == "usrnet_full":
                row["model_size"] = dict(iterations=5, blocks=7)
            report["training"].append(row)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
