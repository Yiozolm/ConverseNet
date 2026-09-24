"""Compare production eager FP32 training with fixed dev b850e38.

Run independent FP64 correctness tests before interpreting speed results. This
benchmark adds paired output/VJP and short SGD trajectory checks, not convergence
or task-quality evidence. Inputs are device resident; compilation, model setup,
warmup, state reset and validation are excluded from timed steps. No CUDA Graphs.
"""
import argparse
from contextlib import contextmanager
import copy
import datetime
import gc
import json
import os
import statistics
import subprocess
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

from extension_loader import load_extension
from fp32_training_baseline import ROOT, current_manifest, load_baseline, sha256


@contextmanager
def production_namespace(ops):
    """Route unchanged Python model forwards to one public C++ implementation.

    Namespace selection happens outside the timed region. This benchmark is
    single-threaded and restores the production namespace on every exit.
    """
    previous = torch.ops.converse2d
    torch.ops.converse2d = ops
    try:
        yield
    finally:
        torch.ops.converse2d = previous


def release_cuda():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def verify_fused_dispatch(ops, expected="any"):
    """Verify the requested spectral route and return its actual autograd names.

    General training comparisons accept either fused solve. Historical ablation
    callers must request their specific route so a default change cannot turn
    an irrelevant selector change into an apparent optimization experiment.
    """
    accepted = {"any": ("SpectralSolve", "FullSolve"),
                "half": ("SpectralSolve",), "full": ("FullSolve",)}
    if expected not in accepted:
        raise ValueError(f"expected must be one of {tuple(accepted)}, got {expected!r}")
    checks = {}
    for scale in (1, 2, 3):
        x = torch.ones(1, 2, 3, 4, device="cuda", requires_grad=True)
        weight = torch.full((1, 2, 3, 3), 1 / 9, device="cuda", requires_grad=True)
        bias = torch.zeros(1, 2, 1, 1, device="cuda", requires_grad=True)
        prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
        output = ops.forward(x, prior, weight, bias, scale, 1e-3, "v7")
        pending, visited, names = [output.grad_fn], set(), set()
        while pending:
            node = pending.pop()
            if node is None or node in visited:
                continue
            visited.add(node)
            names.add(node.name())
            pending.extend(child for child, _ in node.next_functions)
        matches = sorted(name for name in names
                         if "SpectralSolve" in name or "FullSolve" in name)
        if not matches or any(not any(kind in name for kind in accepted[expected])
                              for name in matches):
            raise RuntimeError(
                f"Current production scale={scale} expected {expected} spectrum fused training, "
                f"observed {matches or sorted(names)}")
        checks[str(scale)] = matches
    return checks


def errors(actual, expected):
    actual, expected = actual.double(), expected.double()
    return dict(max_abs=(actual - expected).abs().max().item(),
                relative_l2=((actual - expected).norm() /
                             expected.norm().clamp_min(1e-30)).item())


def compare_tensors(actual, expected, *, atol=3e-5, rtol=3e-4):
    if actual.keys() != expected.keys():
        raise AssertionError("Mismatching state/gradient keys")
    checks = []
    for key, value in actual.items():
        reference = expected[key]
        if not torch.isfinite(value).all() or not torch.isfinite(reference).all():
            raise AssertionError(f"Nonfinite value: {key}")
        torch.testing.assert_close(value, reference, atol=atol, rtol=rtol,
                                   msg=lambda message: f"{key}: {message}")
        checks.append(errors(value, reference))
    return dict(max_abs=max((r["max_abs"] for r in checks), default=0),
                max_relative_l2=max((r["relative_l2"] for r in checks), default=0))


def sample(fn, iters):
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    initial_allocated = torch.cuda.memory_allocated()
    initial_reserved = torch.cuda.memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - began) * 1000 / iters
    return dict(wall_ms=wall_ms, event_ms=start.elapsed_time(end) / iters,
                initial_allocated_bytes=initial_allocated,
                initial_reserved_bytes=initial_reserved,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                incremental_peak_allocated_bytes=(torch.cuda.max_memory_allocated()
                                                  - initial_allocated))


def paired(run, args):
    rows = {"dev": [], "current": []}
    orders = []
    for round_index in range(args.rounds):
        order = ["dev", "current"] if round_index % 2 == 0 else ["current", "dev"]
        orders.append(order)
        for name in order:
            # run() owns exactly one GPU fixture; it is destroyed before the
            # other variant is created. CPU templates never affect CUDA peaks.
            release_cuda()
            rows[name].append(run(name))
            release_cuda()
    medians = {name: {key: statistics.median(row[key] for row in values)
                      for key in values[0]} for name, values in rows.items()}
    ratios = {key: [rows["dev"][i][key] / rows["current"][i][key]
                    for i in range(args.rounds)] for key in ("wall_ms", "event_ms")}
    return dict(round_order=orders, rounds=rows, medians=medians,
                paired_speedups=ratios,
                median_paired_speedup={key: statistics.median(value)
                                      for key, value in ratios.items()})


class OperatorTrain(nn.Module):
    """A trainable producer makes propagation through the solver input necessary."""
    def __init__(self, channels, scale):
        super().__init__()
        self.stem = nn.Conv2d(channels, channels, 1)
        self.weight = nn.Parameter(torch.randn(1, channels, 3, 3).flatten(2)
                                   .softmax(-1).reshape(1, channels, 3, 3))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.scale = scale

    def forward(self, x):
        x = self.stem(x)
        prior = x if self.scale == 1 else F.interpolate(x, scale_factor=self.scale, mode="nearest")
        return torch.ops.converse2d.forward(x, prior, self.weight, self.bias,
                                           self.scale, 1e-3, "v7")


class USRTrain(nn.Module):
    def __init__(self, scale=2):
        super().__init__()
        from models.converse_usrnet import ConverseUSRNet
        self.scale = scale
        self.model = ConverseUSRNet(num_iterations=2, num_blocks=1, backend="cuda")
        # Exercise the solver gradients from step one rather than relying on
        # the residual gates' first update to enable them.
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                if name.endswith(("alpha1", "alpha2")):
                    parameter.fill_(0.1)

    def forward(self, x, kernel):
        return self.model(x, kernel, self.scale)


def workload(kind, shape, scale, accumulation):
    from models.util_converse import Converse2D, ConverseBlock
    batch, channels, height, width = shape
    if kind == "operator":
        model = OperatorTrain(channels, scale)
    elif kind == "block":
        model = ConverseBlock(channels, channels)
        for layer in model.modules():
            if isinstance(layer, Converse2D):
                layer.backend = "cuda"
    elif kind == "usrnet":
        model = USRTrain(scale)
    else:
        raise ValueError(kind)
    batches, targets = [], []
    for _ in range(accumulation):
        x = torch.rand(shape) * 0.2
        if kind == "usrnet":
            kernel = torch.rand(batch, 1, 7, 7)
            batches.append((x, kernel / kernel.sum((-2, -1), keepdim=True)))
        else:
            batches.append((x,))
        targets.append(torch.rand(batch, channels, height * scale, width * scale) * 0.2)
    return model, batches, targets


class TrainingRunner:
    def __init__(self, template, batches, targets):
        self.model = copy.deepcopy(template).cuda().train()
        self.initial = template.state_dict()  # CPU only.
        self.batches = [tuple(t.cuda() for t in batch) for batch in batches]
        self.targets = [target.cuda() for target in targets]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4,
                                         momentum=0.9, foreach=False, fused=False)
        for parameter in self.model.parameters():
            self.optimizer.state[parameter]["momentum_buffer"] = torch.zeros_like(parameter)
        self.loss = None

    @torch.no_grad()
    def reset(self):
        self.model.load_state_dict(self.initial)
        self.optimizer.zero_grad(set_to_none=True)
        for parameter in self.model.parameters():
            self.optimizer.state[parameter]["momentum_buffer"].zero_()
        self.loss = None

    def step(self):
        self.optimizer.zero_grad(set_to_none=True)
        losses = []
        for batch, target in zip(self.batches, self.targets):
            output = self.model(*batch)
            if output.shape != target.shape:
                raise AssertionError((output.shape, target.shape))
            loss = (output - target).square().mean() / len(self.batches)
            loss.backward()
            losses.append(loss.detach())
            del output, loss
        self.optimizer.step()
        self.loss = losses[0] if len(losses) == 1 else torch.stack(losses).sum()

    @torch.no_grad()
    def perturb_batch(self):
        for batch in self.batches:
            batch[0].add_(0.003)
        for target in self.targets:
            target.mul_(0.99)

    @torch.no_grad()
    def snapshot(self):
        result = {"loss": self.loss.detach().cpu().clone()}
        for name, parameter in self.model.named_parameters():
            result[f"param/{name}"] = parameter.detach().cpu().clone()
            if parameter.grad is not None:
                result[f"grad/{name}"] = parameter.grad.detach().cpu().clone()
            result[f"momentum/{name}"] = self.optimizer.state[parameter]["momentum_buffer"].cpu().clone()
        result.update({f"buffer/{name}": value.cpu().clone()
                       for name, value in self.model.named_buffers()})
        return result


def operator_case(shape, scale, variants, args):
    batch, channels, height, width = shape
    cpu = (torch.randn(shape),
           torch.randn(1, channels, 3, 3).flatten(2).softmax(-1).reshape(1, channels, 3, 3),
           torch.zeros(1, channels, 1, 1),
           torch.randn(batch, channels, height * scale, width * scale) * 0.001)

    def fixture():
        x, weight, bias = [t.cuda().requires_grad_() for t in cpu[:3]]
        upstream = cpu[3].cuda()

        def run():
            prior = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
            output = torch.ops.converse2d.forward(x, prior, weight, bias, scale, 1e-3, "v7")
            grads = torch.autograd.grad(output, (x, weight, bias), upstream)
            return dict(output=output.detach(), **dict(zip(("dx", "dw", "db"), grads)))
        return run

    states = {}
    for name, ops in variants.items():
        with production_namespace(ops):
            run = fixture()
            states[name] = {key: value.cpu() for key, value in run().items()}
            del run
        release_cuda()
    check = compare_tensors(states["current"], states["dev"], atol=5e-5, rtol=3e-4)

    def benchmark(name):
        with production_namespace(variants[name]):
            run = fixture()
            for _ in range(args.warmup):
                run()
            return sample(run, args.iters)

    result = dict(shape=shape, scale=scale, numerical_check=check,
                  scope="Operator forward and x/kernel/bias VJP; no loss or optimizer",
                  timing=paired(benchmark, args))
    print(json.dumps(dict(operator=shape, scale=scale, **result["timing"]["medians"])), flush=True)
    return result


def training_case(config, variants, args):
    from models.util_converse import Converse2D

    kind, shape, scale, accumulation = config
    template, batches, targets = workload(kind, shape, scale, accumulation)
    solver_weights = (["weight"] if kind == "operator" else
                      [f"{module_name}.weight" if module_name else "weight"
                       for module_name, module in template.named_modules()
                       if isinstance(module, Converse2D)])
    states, activity = {}, {}
    for name, ops in variants.items():
        with production_namespace(ops):
            runner = TrainingRunner(template, batches, targets)
            states[name] = []
            for step in range(4):
                runner.step()
                states[name].append(runner.snapshot())
                # A changing input and target accompanies repeated parameter
                # updates; both implementations receive identical changes.
                runner.perturb_batch()
            final = states[name][-1]
            updated = [key for key, value in template.named_parameters()
                       if not torch.equal(final[f"param/{key}"], value.detach())]
            nonzero = [key[5:] for key, value in final.items()
                       if key.startswith("grad/") and torch.count_nonzero(value).item()]
            if not set(solver_weights).intersection(updated) or not set(solver_weights).intersection(nonzero):
                raise AssertionError(f"{name}: solver weights were not trained")
            if kind == "usrnet":
                if "model.d.alpha" not in updated or "model.d.alpha" not in nonzero:
                    raise AssertionError(f"{name}: data regularization did not train")
                if not any("kernelnet." in key for key in set(updated).intersection(nonzero)):
                    raise AssertionError(f"{name}: kernel producer did not train")
            activity[name] = dict(updated_parameters=updated, nonzero_gradients=nonzero)
            del runner
        release_cuda()
    checks = [compare_tensors(current, dev)
              for current, dev in zip(states["current"], states["dev"])]

    def benchmark(name):
        with production_namespace(variants[name]):
            runner = TrainingRunner(template, batches, targets)
            for _ in range(args.warmup):
                runner.step()
            runner.reset()
            result = sample(runner.step, args.iters)
            snapshot = runner.snapshot()
            if not all(torch.isfinite(value).all() for value in snapshot.values()):
                raise AssertionError(f"{name}: nonfinite timed trajectory")
            return result

    result = dict(kind=kind, shape=shape, scale=scale, accumulation=accumulation,
                  trajectory_checks=checks, activity=activity,
                  scope="Eager FP32 zero_grad(set_to_none=True), forward, MSE, backward, SGD; lr=1e-4, momentum=.9",
                  timing=paired(benchmark, args))
    print(json.dumps(dict(training=config, **result["timing"]["medians"])), flush=True)
    return result


def suite_configs(suite):
    if suite == "default":
        operators = [((1, 32, 64, 80), 1), ((1, 32, 256, 256), 1),
                     ((1, 32, 64, 80), 2), ((1, 32, 256, 256), 2)]
        models = [("block", (1, 16, 32, 40), 1, 1),
                  ("block", (1, 16, 32, 40), 1, 2),
                  ("usrnet", (1, 3, 16, 20), 2, 1)]
    elif suite == "scale3-batch":
        operators = [((1, 32, 64, 80), 3), ((1, 32, 256, 256), 3)]
        operators += [((batch, 32, 64, 80), scale)
                      for batch in (8, 32) for scale in (1, 2, 3)]
        operators += [((4, 32, 256, 256), scale) for scale in (1, 2, 3)]
        models = [("block", (16, 16, 32, 40), 1, 1),
                  ("usrnet", (8, 3, 16, 20), 3, 1),
                  ("usrnet", (16, 3, 16, 20), 3, 1),
                  ("usrnet", (16, 3, 16, 20), 2, 1)]
    else:
        raise ValueError(suite)
    training = [("operator", shape, scale, 1) for shape, scale in operators]
    return operators, training + models


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--mode", choices=("all", "operators", "training"), default="all")
    parser.add_argument("--suite", choices=("default", "scale3-batch"), default="default")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", help="Defaults to a separate JSON path for each suite")
    args = parser.parse_args()
    if args.output is None:
        filename = "benchmark.json" if args.suite == "default" else "benchmark_scale3_batch.json"
        args.output = "artifacts/training_operator_optimization/" + filename
    if min(args.iters, args.rounds, args.warmup) < 1:
        parser.error("iters, rounds and warmup must be positive")
    if os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
        parser.error("Unset CONVERSE2D_CPU_ONLY to benchmark the production CUDA training backend")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if os.environ.get("CONVERSE2D_BACKEND", "").lower() not in ("", "auto", "cuda"):
        parser.error("CONVERSE2D_BACKEND must not override CUDA with a reference backend")
    # Always ask the production loader to validate/rebuild the current sources.
    skip_build = os.environ.pop("CONVERSE2D_SKIP_BUILD", None)
    try:
        load_extension(verbose=args.verbose_build)
    finally:
        if skip_build is not None:
            os.environ["CONVERSE2D_SKIP_BUILD"] = skip_build
    current_ops = torch.ops.converse2d
    dispatch_check = verify_fused_dispatch(current_ops)
    release_cuda()
    dev_ops, baseline = load_baseline(verbose=args.verbose_build)
    variants = dict(dev=dev_ops, current=current_ops)
    sys.path.insert(0, str(ROOT))
    torch.manual_seed(20260917)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
                  gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  current_source_sha256=current_manifest(), baseline=baseline,
                  current_fused_dispatch=dispatch_check,
                  benchmark_sha256={name: sha256(ROOT / "test" / name) for name in
                                    ("benchmark_fp32_training.py", "fp32_training_baseline.py")},
                  settings=vars(args), operators=[], training=[],
                  memory_scope="PyTorch CUDA allocator totals with one variant resident; includes model, optimizer and input tensors; excludes non-PyTorch driver/library allocations",
                  timing_scope="Warm eager allocator/FFT state, dynamic kernel preparation included every step; no input transfer, AMP or CUDA Graphs",
                  validation_scope="Paired FP32 numerical and four-step SGD checks; independent FP64 tests and real-data convergence required separately")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    operators, training = suite_configs(args.suite)
    if args.mode in ("all", "operators"):
        for shape, scale in operators[:1] if args.quick else operators:
            report["operators"].append(operator_case(shape, scale, variants, args))
            save()
    if args.mode in ("all", "training"):
        for config in training[:1] if args.quick else training:
            report["training"].append(training_case(config, variants, args))
            save()
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
