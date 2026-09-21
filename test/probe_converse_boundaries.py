"""Isolated autograd-only circular padding/cropping candidates for Converse2D.

No production mutation. Full operator forward + dx/dw/db, actual pretrained
prior weights, B4/C128/96x96, scale1/pad2. Fixed independent FP64 budgets from
the existing operator suite: output3e-5/3e-5, gradients5e-5/5e-5.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pad_cat(x, p):
    import torch
    x = torch.cat((x[..., -p:], x, x[..., :p]), dim=-1)
    return torch.cat((x[..., -p:, :], x, x[..., :p, :]), dim=-2)


def crop_view(x, p):
    return x.as_strided((*x.shape[:-2], x.shape[-2]-2*p, x.shape[-1]-2*p),
                        x.stride(), x.storage_offset()+p*x.stride(-2)+p*x.stride(-1))


def make_run(tensors, variant, reference=False):
    import torch
    import torch.nn.functional as F
    from models.converse_core import converse2d_reference
    dtype = torch.float64 if reference else torch.float32
    values = [value.to(device="cuda", dtype=dtype).requires_grad_() for value in tensors[:3]]
    upstream = tensors[3].to(device="cuda", dtype=dtype)

    def run():
        x, weight, bias = values
        x = pad_cat(x, 2) if variant.startswith("cat") else F.pad(x, (2, 2, 2, 2), mode="circular")
        output = (converse2d_reference(x, x, weight, bias, 1, 1e-5) if reference else
                  torch.ops.converse2d.forward(x, x, weight, bias, 1, 1e-5, "v7"))
        output = crop_view(output, 2) if variant.endswith("view") else output[..., 2:-2, 2:-2]
        gradients = torch.autograd.grad(output, values, upstream)
        return dict(output=output, dx=gradients[0], dw=gradients[1], db=gradients[2])
    return run


def compare(actual, expected, atol, rtol):
    import torch
    a, e = actual.double(), expected.double()
    error = (a-e).abs()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(e).all())
    violations = int((error > atol+rtol*e.abs()).sum())
    return dict(passed=finite and violations == 0, finite=finite, failed_elements=violations,
                max_abs=error.max().item(), relative_l2=(error.norm()/e.norm().clamp_min(1e-30)).item(),
                atol=atol, rtol=rtol)


def capture(tensors, variant, reference=False):
    from probe_pointwise_training import clear_cuda
    clear_cuda()
    run = make_run(tensors, variant, reference)
    result = {name: value.detach().cpu() for name, value in run().items()}
    del run
    clear_cuda()
    return result


def time_one(tensors, variant, args):
    import torch
    from probe_pointwise_training import clear_cuda
    clear_cuda()
    run = make_run(tensors, variant)
    for _ in range(args.warmup):
        run()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    start.record()
    end.record()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    began = time.perf_counter()
    start.record()
    for _ in range(args.iters):
        run()
    end.record()
    torch.cuda.synchronize()
    result = dict(wall_ms=(time.perf_counter()-began)*1000/args.iters,
                  cuda_ms=start.elapsed_time(end)/args.iters,
                  allocated_bytes=torch.cuda.max_memory_allocated(),
                  reserved_bytes=torch.cuda.max_memory_reserved())
    del run
    clear_cuda()
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=4)
    args = parser.parse_args()
    if args.output.exists() or min(args.warmup, args.iters, args.rounds) < 1:
        parser.error("Require new output path and positive iteration counts")
    import torch
    import train_usrnet_dataset as worker
    from extension_loader import load_extension
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    load_extension()  # Source/header/PyTorch/binary hash validation; stale fails.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator().manual_seed(20260918)
    checkpoint = ROOT / "model_zoo/converse_usrnet.pth"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    x = torch.randn(4, 128, 96, 96, generator=generator)
    upstream = torch.randn(x.shape, generator=generator)/x.numel()**.5
    weight = state["p.m_body.0.conv1.3.weight"].clone()
    bias = state["p.m_body.0.conv1.3.bias"].clone()
    tensors = (x, weight, bias, upstream)
    variants = ["native_slice", "cat_slice", "native_view", "cat_view"]
    reference = capture(tensors, "native_slice", reference=True)
    report = dict(status="running", shape=list(x.shape), scale=1, padding=2, eps=1e-5,
                  numerical_reference="Independent FP64 full FFT reference with native pad and slices",
                  source_sha256=worker.source_hashes(), script_sha256=worker.file_hash(__file__),
                  checkpoint_sha256=worker.file_hash(checkpoint),
                  input_sha256=worker.tensor_hash(dict(zip(("x", "weight", "bias", "upstream"), tensors))),
                  config=vars(args), torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                  finite_precision="FP32, TF32 off; identical mathematical boundary, ATen autograd only",
                  validation={}, rounds=[])
    baseline = None
    for variant in variants:
        actual = capture(tensors, variant)
        comparisons = {name: compare(actual[name], ref, *( (3e-5, 3e-5) if name == "output" else (5e-5, 5e-5)))
                       for name, ref in reference.items()}
        if baseline is None:
            baseline = actual
        report["validation"][variant] = dict(passed=all(row["passed"] for row in comparisons.values()),
                                             reference=comparisons,
                                             same_as_current={name: torch.equal(value, baseline[name])
                                                              for name, value in actual.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    if not all(row["passed"] for row in report["validation"].values()):
        report["status"] = "numerical_gate_failed_no_timing"
        worker.write_json(args.output, report)
        raise SystemExit(1)
    for index in range(args.rounds):
        order = variants[index % len(variants):]+variants[:index % len(variants)]
        report["rounds"].append(dict(order=order, variants={name: time_one(tensors, name, args) for name in order}))
    report["summary"] = {}
    for variant in variants:
        ratios = [r["variants"]["native_slice"]["wall_ms"]/r["variants"][variant]["wall_ms"]
                  for r in report["rounds"]]
        report["summary"][variant] = dict(wall_ms=statistics.median(r["variants"][variant]["wall_ms"] for r in report["rounds"]),
                                          paired_speedup_median=statistics.median(ratios), ratios=ratios)
    report["status"] = "complete"
    worker.write_json(args.output, report)
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
