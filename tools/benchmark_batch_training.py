"""Checked B2/B4 s1 reduction and full B4 USRNet training comparison.

Run each --root in a fresh process. The operator includes circular pad2/crop2
around an FFT100x100 shared-input solve. The model uses real batch4 (no gradient
accumulation or microbatch split), LR32/s3, MSE and complete Adam updates. Each
timed model round restores the same CPU snapshot of a single *Python FP32*
preparatory Adam step, so baseline/candidate start from the same prepared state.
State restoration, input transfer, snapshots and numerical checks are untimed.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import benchmark_fp32_p0 as common
import benchmark_fp32_psf as psf


def cpu_tree(value):
    import torch
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def model_snapshot(torch, model, optimizer, output=None, loss=None):
    result = common.records({"output": output, "loss": loss}) if output is not None else {}
    for name, parameter in model.named_parameters():
        if output is not None and parameter.grad is not None:
            result[f"gradient/{name}"] = common.tensor_record(parameter.grad)
        result[f"parameter/{name}"] = common.tensor_record(parameter)
        for key, value in optimizer.state.get(parameter, {}).items():
            if torch.is_tensor(value):
                result[f"adam/{name}/{key}"] = common.tensor_record(value)
    return result


def run_operators(torch, reference, args, checkpoint):
    cases = {}
    key = "p.m_body.0.conv1.3."
    masks = {"all": (True, True, True), "input_only": (True, False, False), "kernel_only": (False, True, False)}
    for batch in (2, 4):
        shape = (batch, 128, 96, 96)
        generator = torch.Generator().manual_seed(args.seed + batch)
        x = torch.randn(shape, generator=generator)
        raw = (x, checkpoint[key + "weight"].clone(), checkpoint[key + "bias"].clone())
        upstream_cpu = torch.randn(shape, generator=generator) / x.numel() ** .5
        spec = dict(scale=1, padding=2, padding_mode="circular", eps=1e-5)
        for mask, needs in masks.items():
            data = common.prepare(torch, raw, needs, torch.float32)
            refdata = common.prepare(torch, raw, needs, torch.float64)
            upstream = upstream_cpu.cuda()
            expected = psf.forward_vjp(torch, reference, refdata, upstream.double(), spec)
            control = psf.forward_vjp(torch, reference, data, upstream, spec)
            def call():
                return psf.forward_vjp(torch, torch.ops.converse2d.forward, data, upstream, spec)
            actual = call()
            snapshot, control_snapshot = common.records(actual, expected), common.records(control, expected)
            del actual, expected, control, refdata
            name = f"operator/shared_s1_B{batch}_C128_FFT100/{mask}"
            item = {
                "scope": "Complete pad2 -> shared-prior s1 solve -> crop2 forward and selected VJPs; differentiable kernel FFT per call",
                "caller_shape": shape, "fft_shape": [100, 100], "kernel_shape": [1, 128, 3, 3],
                "kernel_checkpoint_keys": [key + "weight", key + "bias"],
                "needs_grad": dict(zip(("x", "weight", "bias"), needs)),
                "fixture": common.records(dict(zip(("x", "weight", "bias", "upstream"), (*data, upstream)))),
                "snapshot": snapshot, "python_fp32": control_snapshot,
                "timing": common.measure(torch, call, args),
            }
            item["python_fp32_noninferiority"] = common.noninferiority(item)
            cases[name] = item
            print(name, item["timing"]["median"], flush=True)
            del data, upstream
    torch.ops.converse2d.clear_cache()
    torch.cuda.empty_cache()
    return cases


def run_model(torch, args, initial):
    from models.converse_usrnet import ConverseUSRNet
    model = ConverseUSRNet(backend="cuda").cuda()
    model.load_state_dict(initial, strict=True)
    model.train()
    generator = torch.Generator().manual_seed(args.seed + 100)
    x = torch.rand(4, 3, 32, 32, generator=generator).cuda()
    k = torch.rand(4, 1, 7, 7, generator=generator)
    k = (k / k.sum((-2, -1), keepdim=True)).cuda()
    target = torch.rand(4, 3, 96, 96, generator=generator).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)
    def step(snapshot=False):
        optimizer.zero_grad(set_to_none=True)
        output = model(x, k, 3)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        return model_snapshot(torch, model, optimizer, output, loss) if snapshot else None
    # Candidate's first step is kept independently of the common timing state.
    first_step = step(snapshot=True)
    model.load_state_dict(initial, strict=True)
    optimizer.state.clear()
    backends = [(module, module.backend) for module in model.modules() if hasattr(module, "backend")]
    try:
        for module, _ in backends:
            module.backend = "pytorch"
        step()
    finally:
        for module, backend in backends:
            module.backend = backend
    prepared_snapshot = model_snapshot(torch, model, optimizer)
    prepared_parameters = cpu_tree(model.state_dict())
    prepared_optimizer = cpu_tree(optimizer.state_dict())
    prepared_sha = hashlib.sha256(json.dumps(prepared_snapshot, sort_keys=True).encode()).hexdigest()
    def restore():
        model.load_state_dict(prepared_parameters, strict=True)
        # Adam's CPU step tensor can otherwise alias the template across rounds.
        optimizer.load_state_dict(copy.deepcopy(prepared_optimizer))
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
    restore()
    timing = common.measure(torch, step, args, before_round=restore)
    name = "usrnet/B4_LR32_s3/adam_training"
    item = {
        "scope": "Complete real batch4 model: zero_grad(set_to_none), forward, MSE, all parameter backward, Adam(foreach=False,fused=False,lr=1e-5); no microbatch split",
        "fixture": common.records({"x": x, "kernel": k, "target": target}), "snapshot": first_step,
        "timing": timing, "fft_shapes": {"DataNet": [96, 96], "35_shared_prior_calls": [100, 100]},
        "prepared_state": prepared_snapshot, "prepared_state_sha256": prepared_sha,
        "restore_policy": "Before each timed round load one common Python FP32 warm-step parameter/Adam snapshot; deep-copy CPU step scalars; all restoration transfers excluded",
    }
    print(name, timing["median"], flush=True)
    return {name: item}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, type=Path)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--seed", type=int, default=823)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--deterministic-algorithms", action="store_true")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output filename")
    if args.compare:
        result = common.compare(args.compare)
        before, after = [json.loads(path.read_text(encoding="utf-8")) for path in args.compare]
        prepared = {name: before["cases"][name]["prepared_state_sha256"] == after["cases"][name]["prepared_state_sha256"]
                    for name in before["cases"].keys() & after["cases"].keys() if "prepared_state_sha256" in before["cases"][name]}
        result.update(kind="batch_training_comparison", common_prepared_states_equal=prepared,
                      helpers_match=before["helper_sha256"] == after["helper_sha256"],
                      harness_match=before["harness_sha256"] == after["harness_sha256"],
                      note="Whole operator and real batch4 training spans. No inference, dataset quality or convergence claim. Prepared-state hashes must match for paired timing eligibility.")
    else:
        if args.warmup < 0 or min(args.rounds, args.iters) < 1:
            parser.error("warmup>=0, rounds/iters>0 required")
        if os.environ.get("CONVERSE2D_BACKEND") or os.environ.get("CONVERSE2D_CPU_ONLY") == "1":
            raise RuntimeError("Unset backend and CPU-only overrides")
        args.root = args.root.resolve()
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        os.environ["CONVERSE2D_SKIP_BUILD"] = "0" if args.build else "1"
        sys.path.insert(0, str(args.root))
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(args.deterministic_algorithms)
        torch.manual_seed(args.seed)
        specification = importlib.util.spec_from_file_location("batch_checked_loader", args.root / "test/extension_loader.py")
        loader = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(loader)
        loader.load_extension()
        from models.converse_core import converse2d_reference
        manifest = json.loads((args.root / ".build/cuda/source_manifest.json").read_text(encoding="utf-8"))
        sources = loader.production_source_hashes()
        model_paths = ("models/util_converse.py", "models/converse_core.py", "models/converse_usrnet.py", "test/extension_loader.py")
        python_hashes = {name: common.file_sha(args.root / name) for name in model_paths}
        helpers = {"p0": common.file_sha(common.__file__), "psf": common.file_sha(psf.__file__)}
        harness = common.file_sha(__file__)
        checkpoint_path = args.root / "model_zoo/converse_usrnet.pth"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        result = {"kind": "batch_training_benchmark", "root": str(args.root), "harness_sha256": harness,
                  "helper_sha256": helpers, "source_sha256": sources, "python_sha256": python_hashes,
                  "checked_build_manifest": manifest, "checkpoint_sha256": common.file_sha(checkpoint_path),
                  "settings": {key: getattr(args, key) for key in ("warmup", "rounds", "iters", "seed", "skip_model", "deterministic_algorithms")},
                  "environment": {"torch": str(torch.__version__), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "tf32": False, "amp": False,
                                  "cudnn_deterministic": True, "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
                  "memory_scope": "Additional PyTorch allocated peak over live tensors/state before timing; not whole-process VRAM",
                  "cases": run_operators(torch, converse2d_reference, args, checkpoint)}
        if not args.skip_model:
            result["cases"].update(run_model(torch, args, checkpoint))
        if sources != loader.production_source_hashes() or python_hashes != {name: common.file_sha(args.root / name) for name in model_paths}:
            raise RuntimeError("Sources changed during benchmark")
        if common.file_sha(args.root / ".build/cuda" / manifest["library"]) != manifest["binary_sha256"]:
            raise RuntimeError("Checked binary changed during benchmark")
        if harness != common.file_sha(__file__) or helpers != {"p0": common.file_sha(common.__file__), "psf": common.file_sha(psf.__file__)}:
            raise RuntimeError("Harness or helper changed during benchmark")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
