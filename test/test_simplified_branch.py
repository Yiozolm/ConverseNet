"""Characterize a simplified-branch revision or worktree alongside current main.

Snapshots source with git show; only the C++ operator namespace is renamed.
The original pytest test bodies are replayed directly (decorators/import wiring
removed by AST), so pytest is not required. Failures are recorded, benchmarks
still finish, and the process exits nonzero when correctness checks fail.
"""
import ast
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from torch.utils.cpp_extension import load, CUDA_HOME

from extension_loader import ROOT, load_extension
from benchmark_corrected import load_legacy, measure
from test_correctness import dense_spatial
from models.converse_core import converse2d_reference

REF = "codex/numerically-stable-closed-form"
OUT = ROOT / "analysis" / "simplified_branch_results.json"


def snapshot(source_tree=None):
    sha = subprocess.check_output(["git", "rev-parse", "HEAD" if source_tree else REF],
                                  cwd=source_tree or ROOT).decode().strip()
    paths = ["Converse2D/torch_converse2d/converse2d.cpp", "models/util_converse.py",
             "models/converse_usrnet.py", "test/test_closed_form.py"]
    contents = {path: (source_tree/path).read_bytes() if source_tree else
                subprocess.check_output(["git", "show", f"{sha}:{path}"], cwd=ROOT) for path in paths}
    suffix = "-working-" + hashlib.sha256(b"".join(contents.values())).hexdigest()[:12] if source_tree else ""
    folder = ROOT / ".build" / "simplified_branch" / (sha[:8]+suffix)
    for path in paths:
        target = folder / "source" / path
        target.parent.mkdir(parents=True, exist_ok=True)
        data = contents[path]
        if not target.exists() or target.read_bytes() != data:
            target.write_bytes(data)
    build = folder / "build"
    build.mkdir(exist_ok=True)
    raw = (folder / "source" / paths[0]).read_text(encoding="utf-8")
    raw = raw.replace("TORCH_LIBRARY(converse2d,", "TORCH_LIBRARY(converse2d_simplified,")
    raw = raw.replace("TORCH_LIBRARY_IMPL(converse2d,", "TORCH_LIBRARY_IMPL(converse2d_simplified,")
    cpp = build / "simplified.cpp"
    if not cpp.exists() or cpp.read_text(encoding="utf-8") != raw:
        cpp.write_text(raw, encoding="utf-8")
    cuda = "CONVERSE2D_WITH_CUDA" in raw and torch.version.cuda is not None and CUDA_HOME is not None
    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3"]
    if cuda: flags.append("-DCONVERSE2D_WITH_CUDA=1")
    load(name="converse2d_simplified_ext", sources=[str(cpp)], build_directory=str(build),
         extra_cflags=flags, with_cuda=cuda, verbose=False)
    return sha, folder / "source"


def import_source(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def diff(a, b):
    return (a-b).abs().max().item()


def save(results):
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")


def args_for(s, device, dtype=torch.float64, h=5, w=7, c=2, b=1, grad=False):
    x = torch.randn(b,c,h,w,device=device,dtype=dtype)
    prior = torch.randn(b,c,h*s,w*s,device=device,dtype=dtype)
    weight = torch.randn(1,c,3,3,device=device,dtype=dtype).flatten(2).softmax(-1).reshape(1,c,3,3)
    bias = torch.zeros(1,c,1,1,device=device,dtype=dtype)
    return tuple(t.requires_grad_(grad) for t in (x,prior,weight,bias))


def replay_original(source, results):
    util = import_source("simplified_util", source / "models/util_converse.py")
    util._HAS_CONVERSE2D_EXT = True
    util.converse2d_CUDA = torch.ops.converse2d_simplified.forward
    usrnet = import_source("simplified_usrnet", source / "models/converse_usrnet.py")
    path = source / "test/test_closed_form.py"
    # Mathematical test bodies stay unchanged; direct extension calls select
    # the isolated branch operator instead of the main operator.
    code = path.read_text(encoding="utf-8").replace("torch.ops.converse2d.", "torch.ops.converse2d_simplified.")
    tree = ast.parse(code)
    tree.body = [node for node in tree.body if not (
        isinstance(node, ast.Import) and any(a.name == "pytest" for a in node.names)
        or isinstance(node, ast.ImportFrom) and node.module in ("models.util_converse", "models.converse_usrnet"))]
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    namespace = {"__file__":str(path), "__name__":"simplified_native_tests",
                 "Converse2D":util.Converse2D, "ConvReverseDataNet":usrnet.ConvReverseDataNet}
    exec(compile(tree, str(path), "exec"), namespace)
    cases = [("test_converse2d_residual_form_matches_legacy_forward_and_gradients", s) for s in (1,2,3,4)]
    cases += [("test_data_net_residual_form_matches_legacy_kernel_and_lambda_gradients", s) for s in (1,2,3)]
    cases += [("test_residual_form_reduces_small_lambda_cancellation_error", None)]
    cases += [("test_extension_residual_form_matches_pytorch_forward_and_gradients", s) for s in (1,2,3,4)]
    rows = results["branch_original_tests"] = []
    for device in ("cpu", "cuda"):
        for name, scale in cases:
            torch.ops.converse2d_simplified.clear_cache()
            row = dict(device=device,test=name,scale=scale)
            try:
                with torch.device(device):
                    namespace[name]() if scale is None else namespace[name](scale)
                row["passed"] = True
            except Exception as error:
                row.update(passed=False,error=str(error))
            rows.append(row)
    print("Original tests:",sum(r["passed"] for r in rows),"/",len(rows),flush=True)


def boundaries(results):
    branch = torch.ops.converse2d_simplified
    current = torch.ops.converse2d
    rows = results["edge_cases"] = []
    torch.manual_seed(271)
    for device in ("cpu", "cuda"):
        for s in (1,2,3,4):
            data = args_for(s,device,grad=True,h=3,w=4,c=1)
            branch.clear_cache()
            reference = dense_spatial(data,s,1e-3)
            expected_grad = torch.autograd.grad(reference.sum(),data)
            for label, fn in (("simplified",lambda:branch.forward(*data,s,1e-3)),
                              ("current_v7",lambda:current.forward(*data,s,1e-3,"v7"))):
                out = fn()
                grads = torch.autograd.grad(out.sum(),data,allow_unused=True)
                error = diff(out,reference)
                row = dict(test="independent_x0_dense_and_gradients",device=device,scale=s,implementation=label,
                           max_abs=error,missing_gradients=[k for k,g in zip(("x","x0","weight","bias"),grads) if g is None],
                           gradient_errors={k:None if g is None else diff(g,e) for k,g,e in zip(("x","x0","weight","bias"),grads,expected_grad)})
                row["passed"] = error < 1e-9 and not row["missing_gradients"] and all(v < 2e-8 for v in row["gradient_errors"].values())
                rows.append(row)
        for label, ops in (("simplified",branch),("current_v7",current)):
            data = args_for(2,device)
            ops.clear_cache()
            with torch.no_grad():
                ops.forward(*data,2,1e-5)
                data[2].mul_(0.7).add_(0.03)
                stale = ops.forward(*data,2,1e-5)
                ops.clear_cache()
                fresh = ops.forward(*data,2,1e-5)
            error = diff(stale,fresh)
            rows.append(dict(test="inplace_weight_update",device=device,implementation=label,max_abs=error,passed=error<1e-10))
            # Frozen weights may still be needed for the input's backward.
            data = args_for(2,device)
            ops.clear_cache()
            with torch.inference_mode():
                ops.forward(*data,2,1e-5)
            data[0].requires_grad_(True)
            row = dict(test="inference_cache_to_input_grad",device=device,implementation=label)
            try:
                out = ops.forward(*data,2,1e-5)
                torch.autograd.grad(out.sum(),data[0])
                row["passed"] = True
            except Exception as error:
                row.update(passed=False,error=str(error))
            rows.append(row)
            ops.clear_cache()
        for dtype in (torch.float16,torch.bfloat16):
            data = args_for(2,device,dtype=dtype)
            for label, ops in (("simplified",branch),("current_v7",current)):
                ops.clear_cache()
                row = dict(test="low_precision_non_power_of_two",device=device,dtype=str(dtype),implementation=label)
                try:
                    with torch.no_grad():
                        out=ops.forward(*data,2,1e-5)
                    row["passed"] = bool(torch.isfinite(out).all())
                except Exception as error:
                    row.update(passed=False,error=str(error))
                rows.append(row)
    print("Edge failures:",sum(not r["passed"] for r in rows),"/",len(rows),flush=True)


def benchmark(results):
    load_legacy()
    rows = results["benchmarks"] = []
    torch.manual_seed(14)
    for training in (False,True):
        for c,h,w,s in ((32,128,128,1),(32,128,128,2),(32,128,128,3),(64,256,256,2)):
            x=torch.randn(1,c,h,w,device="cuda")
            x0=x if s==1 else torch.nn.functional.interpolate(x,scale_factor=s,mode="nearest")
            weight=torch.randn(1,c,3,3,device="cuda").flatten(2).softmax(-1).reshape(1,c,3,3)
            bias=torch.zeros(1,c,1,1,device="cuda")
            if training:
                # The simplified branch's scale=1 API assumes x0=x. Keep that
                # valid contract for timings; boundary failures are tested above.
                x0 = x if s==1 else x0.detach().clone()
                for tensor in (x,x0,weight,bias): tensor.requires_grad_(True)
            data=(x,x0,weight,bias,s,1e-5)
            with torch.no_grad():
                xd=x.double()
                ref=converse2d_reference(xd,xd if s==1 else x0.double(),weight.double(),bias.double(),s).float()
            targets={"legacy":lambda:torch.ops.converse2d_legacy.forward(*data),
                     "simplified":lambda:torch.ops.converse2d_simplified.forward(*data),
                     "current_v2":lambda:torch.ops.converse2d.forward(*data,"v2"),
                     "current_v7":lambda:torch.ops.converse2d.forward(*data,"v7")}
            for ops in (torch.ops.converse2d,torch.ops.converse2d_simplified,torch.ops.converse2d_legacy): ops.clear_cache()
            with (torch.enable_grad() if training else torch.no_grad()):
                for label,forward in targets.items():
                    output=forward().detach()
                    error=diff(output,ref)
                    if label != "legacy": torch.testing.assert_close(output,ref,atol=1e-4,rtol=5e-5)
                    grad_inputs=(x,weight,bias) if s==1 else (x,x0,weight,bias)
                    fn=(lambda:torch.autograd.grad(forward().square().mean(),grad_inputs)) if training else forward
                    row=dict(training=training,C=c,H=h,W=w,scale=s,implementation=label,max_abs_vs_float64=error,
                             **measure(fn,5,10))
                    rows.append(row)
                    print(json.dumps(row),flush=True)
            for ops in (torch.ops.converse2d,torch.ops.converse2d_simplified,torch.ops.converse2d_legacy): ops.clear_cache()
            save(results)


def main():
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-tree", type=Path, help="Test uncommitted branch worktree sources")
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    OUT = args.output.resolve()
    load_extension()
    source_tree = args.source_tree.resolve() if args.source_tree else None
    sha,source=snapshot(source_tree)
    results=dict(branch=REF,commit=sha,current_main=subprocess.check_output(["git","rev-parse","main"],cwd=ROOT).decode().strip(),
                 gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,
                 source_tree=str(source_tree) if source_tree else None, source_snapshot=str(source))
    replay_original(source,results)
    save(results)
    boundaries(results)
    save(results)
    if not args.skip_benchmark:
        benchmark(results)
    save(results)
    print("Results:",OUT)
    sys.exit(0 if all(row["passed"] for group in ("branch_original_tests", "edge_cases")
                     for row in results[group]) else 1)


if __name__ == "__main__":
    main()
