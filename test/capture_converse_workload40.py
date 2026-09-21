"""Freeze one real Python-FP32 full-USRNet step for a separate 40-op replay.

Pretrained full5/7, seed17 step0, HR96/LR32, batch4, scale3, unclipped RGB MSE.
No optimizer update, gate change or spectrum cache. Forward hooks observe whole
Converse modules; tensor hooks capture their true loss grad_outputs. Capture,
CPU copies, hashes and serialization are never benchmark measurements.

The resulting replay deliberately cuts model dependencies at module inputs and
generated DataNet kernels. It retains seven prior weights/biases used five times
and one DataNet bias used five times: 40 input leaves + 20 unique parameter/kernel
leaves = 60 VJP targets. This is not an alternative full-model training graph.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FORMAT = "converse-module-workload40-v1"
DEFAULT_CAPTURE = ROOT / "artifacts/training_research/operator_workload40/python_seed17_step0"


def source_hashes():
    import train_usrnet_dataset as worker
    result = worker.source_hashes()
    for name in ("capture_converse_workload40.py", "benchmark_converse_workload40.py",
                 "benchmark_shared_s1_cuda.py", "diagnose_boundary_precision.py",
                 "probe_converse_boundaries.py", "validate_shared_s1_python_fp32.py"):
        result["test/" + name] = worker.file_hash(ROOT / "test" / name)
    return result


def configure_cuda():
    import torch
    if os.environ.get("CONVERSE2D_CPU_ONLY") == "1" or not torch.cuda.is_available():
        raise RuntimeError("This capture/replay requires CUDA; no CPU substitute")
    if os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE") == "1":
        raise RuntimeError("TF32 override conflicts with FP32 protocol")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(False)
    return dict(torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                tf32=False, cudnn_benchmark=False, cudnn_deterministic=True,
                deterministic_algorithms=False, amp=False, graphs=False)


def tensor_layout(value):
    # Reject unsupported overlapping layouts instead of silently changing them.
    span = 1
    for stride, size in sorted(zip(value.stride(), value.shape)):
        if size > 1:
            if stride < span:
                raise RuntimeError(f"Overlapping/broadcast input layout unsupported: {value.shape}/{value.stride()}")
            span += (size - 1) * stride
    return dict(shape=list(value.shape), stride=list(value.stride()),
                original_storage_offset=value.storage_offset(), dtype=str(value.dtype),
                original_requires_grad=value.requires_grad)


def payload_hash(payload):
    import train_usrnet_dataset as worker
    return worker.tensor_hash({group + "/" + name: value for group in
                              ("leaves", "grad_outputs", "captured_parameter_vjps")
                              for name, value in payload[group].items()})


def validate_payload(payload):
    import torch
    calls, leaves, upstreams = payload["calls"], payload["leaves"], payload["grad_outputs"]
    if payload["format"] != FORMAT or len(calls) != 40:
        raise RuntimeError("Expected the frozen 40-module format")
    if len(leaves) != 60 or len(payload["target_keys"]) != 60 or set(payload["target_keys"]) != set(leaves):
        raise RuntimeError("Expected exactly 60 unique VJP targets")
    if len(set(payload["target_keys"])) != 60 or set(upstreams) != {str(index) for index in range(40)}:
        raise RuntimeError("Repeated target or missing upstream")
    counts = Counter(row["kind"] for row in calls)
    if counts != {"data": 5, "prior": 35}:
        raise RuntimeError(f"Incorrect module counts: {counts}")
    weights, biases = Counter(), Counter()
    for index, row in enumerate(calls):
        iteration, position = divmod(index, 8)
        kind = "data" if position == 0 else "prior"
        expected_name = "d" if kind == "data" else f"p.m_body.{position - 1}.conv1.3"
        expected_scale = 3 if index == 0 else 1
        if (row["index"] != index or row["iteration"] != iteration or row["kind"] != kind
                or row["module"] != expected_name or row["scale"] != expected_scale):
            raise RuntimeError(f"Unexpected execution order at call {index}")
        expected_shape = [4, 64, 32, 32] if index == 0 else [4, 64 if kind == "data" else 128, 96, 96]
        expected_kernel = [4, 64, 7, 7] if kind == "data" else [1, 128, 3, 3]
        expected_bias = [1, 64 if kind == "data" else 128, 1, 1]
        if (row["input_key"] != f"input/{index:02d}" or list(leaves[row["input_key"]].shape) != expected_shape
                or list(leaves[row["weight_key"]].shape) != expected_kernel
                or list(leaves[row["bias_key"]].shape) != expected_bias
                or list(upstreams[str(index)].shape) != [4, expected_shape[1], 96, 96]):
            raise RuntimeError(f"Incorrect captured shapes/keys at call {index}")
        if row["padding"] != (0 if kind == "data" else 2) or row["padding_mode"] != "circular":
            raise RuntimeError("Unexpected module boundary")
        if row["eps"] != (1e-3 if kind == "data" else 1e-5) or row["variant"] != "v7":
            raise RuntimeError("Epsilon/variant differs from the original full model")
        expected_weight = f"dynamic_kernel/{iteration}" if kind == "data" else f"parameter/{expected_name}.weight"
        expected_bias_key = "parameter/d.alpha" if kind == "data" else f"parameter/{expected_name}.bias"
        if (row["weight_key"], row["bias_key"]) != (expected_weight, expected_bias_key):
            raise RuntimeError("Shared parameter identities were lost")
        weights[row["weight_key"]] += 1
        biases[row["bias_key"]] += 1
        if row["grad_output_hook_calls"] != 1:
            raise RuntimeError("Each output must have exactly one captured total upstream")
    if sorted(weights.values()) != [1] * 5 + [5] * 7 or list(sorted(biases.values())) != [5] * 8:
        raise RuntimeError("Expected 7 shared prior weights and 8 shared bias targets")
    noninputs = set(leaves) - {row["input_key"] for row in calls}
    if set(payload["captured_parameter_vjps"]) != noninputs:
        raise RuntimeError("Missing original model/shared-parameter or dynamic-kernel VJP")
    for group in (leaves, upstreams, payload["captured_parameter_vjps"]):
        if any(value.device.type != "cpu" or value.dtype != torch.float32 or value.requires_grad
               or not bool(torch.isfinite(value).all()) for value in group.values()):
            raise RuntimeError("Capture must contain finite detached CPU FP32 tensors")
    return dict(calls=40, s3_data=1, s1_data=4, prior=35, targets=60, output_plus_vjp_tensors=100,
                unique_weights=12, unique_biases=8, prior_weight_references=[5] * 7,
                shared_data_bias_references=5, original_layouts_recorded=True)


def capture(args, report):
    import numpy as np
    import random
    import torch
    import torch.nn.functional as F
    import train_usrnet_dataset as worker
    from usrnet_training_data import DatasetProtocol
    from models.converse_usrnet import ConverseUSRNet
    from models.util_converse import Converse2D
    if os.environ.get("CONVERSE2D_BACKEND", "") not in ("", "pytorch"):
        raise RuntimeError("Unset a backend override: capture must use original Python FP32")
    report["environment"] = configure_cuda()
    random.seed(17); np.random.seed(17); torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    frozen_path = ROOT / "artifacts/native_deconv_target/source_before/manifest.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if worker.file_hash(ROOT / "models/converse_core.py") != frozen["models/converse_core.py"]:
        raise RuntimeError("Original Python full-FFT reference source changed")
    report["reference_manifest_sha256"] = worker.file_hash(frozen_path)
    protocol = DatasetProtocol(args.manifest, patch_size=96, scale=3, seed=17, noise_std=.01)
    recipe_path = ROOT / "artifacts/dataset_training/protocol.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if protocol.metadata["manifest_sha256"] != recipe["manifest_sha256"]:
        raise RuntimeError("The declared real-data split differs from the original recipe")
    report["dataset_recipe_sha256"] = worker.file_hash(recipe_path)
    batch = protocol.train_batch(0, 4)
    report["data_protocol"] = protocol.metadata
    report["batch_sha256"] = worker.tensor_hash(dict(zip(("lr", "kernel", "hr"), batch)))
    report["checkpoint_sha256"] = worker.file_hash(args.checkpoint)
    if report["checkpoint_sha256"] != worker.file_hash(ROOT / "model_zoo/converse_usrnet.pth"):
        raise RuntimeError("Capture must use the original pretrained checkpoint")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model = ConverseUSRNet(num_iterations=5, num_blocks=7, in_channels=64, backend="pytorch")
    model.load_state_dict(state, strict=True)
    model = model.float().cuda().train()
    model.reuse_training_spectra = False
    params = dict(model.named_parameters())
    if len(params) != 133 or sum(value.numel() for value in params.values()) != 307987:
        raise RuntimeError("Expected the full pretrained 133-parameter-tensor architecture")
    initial_hash = worker.tensor_hash(model.state_dict())
    if initial_hash != worker.tensor_hash(state):
        raise RuntimeError("Strict loading altered the original state")
    prior = {name: module for name, module in model.named_modules() if isinstance(module, Converse2D)}
    if set(prior) != {f"p.m_body.{index}.conv1.3" for index in range(7)}:
        raise RuntimeError("Expected exactly seven prior solver modules")
    payload = dict(format=FORMAT, calls=[], leaves={}, layouts={}, grad_outputs={}, grad_output_layouts={},
                   captured_parameter_vjps={}, target_keys=[], output_hashes={}, source_sha256=report["source_sha256"])
    handles, pending, parameter_ids = [], {}, {}

    def snapshot(value):
        return value.detach().to(device="cpu").contiguous().clone()

    def save_leaf(key, value, parameter=False):
        if not value.requires_grad or value.dtype != torch.float32:
            raise RuntimeError("All captured operator inputs/weights/biases must need FP32 gradients")
        if key in payload["leaves"]:
            if not parameter or parameter_ids[key] != (id(value), value._version):
                raise RuntimeError("A shared parameter changed identity/version")
            return
        payload["leaves"][key] = snapshot(value)
        payload["layouts"][key] = tensor_layout(value)
        if parameter:
            parameter_ids[key] = (id(value), value._version)

    def before(name, kind):
        def hook(module, inputs, kwargs):
            index = len(payload["calls"])
            x = inputs[0] if inputs else kwargs["x"]
            if kind == "prior":
                weight, bias = module.weight, module.bias
                scale, padding, mode = module.scale, module.padding, module.padding_mode
                wk, bk = f"parameter/{name}.weight", f"parameter/{name}.bias"
            else:
                weight = inputs[1] if len(inputs) > 1 else kwargs["k"]
                bias = module.alpha
                scale = inputs[2] if len(inputs) > 2 else kwargs["sf"]
                padding = inputs[3] if len(inputs) > 3 else kwargs.get("padding", 0)
                mode = inputs[4] if len(inputs) > 4 else kwargs.get("padding_mode", "circular")
                wk, bk = f"dynamic_kernel/{index // 8}", "parameter/d.alpha"
            ik = f"input/{index:02d}"
            save_leaf(ik, x)
            save_leaf(wk, weight, parameter=kind == "prior")
            save_leaf(bk, bias, parameter=True)
            row = dict(index=index, iteration=index // 8, module=name, kind=kind, input_key=ik,
                       weight_key=wk, bias_key=bk, scale=int(scale), padding=int(padding),
                       padding_mode=mode, eps=float(module.eps), variant=module.variant,
                       grad_output_hook_calls=0)
            payload["calls"].append(row)
            pending.setdefault(name, []).append(row)
            if kind == "data":
                def save_kernel_grad(gradient, key=wk):
                    if key in payload["captured_parameter_vjps"]:
                        raise RuntimeError("Dynamic kernel gradient captured more than once")
                    payload["captured_parameter_vjps"][key] = snapshot(gradient)
                handles.append(weight.register_hook(save_kernel_grad))
        return hook

    def after(name):
        def hook(_module, _inputs, output):
            row = pending[name].pop()
            key = str(row["index"])
            if not output.requires_grad or output.dtype != torch.float32:
                raise RuntimeError("Captured module output lost FP32 autograd")
            payload["output_hashes"][key] = worker.tensor_hash(dict(value=snapshot(output)))
            row["output_shape"] = list(output.shape)
            def save_upstream(gradient):
                row["grad_output_hook_calls"] += 1
                if key in payload["grad_outputs"]:
                    raise RuntimeError("Output hook fired more than once")
                payload["grad_outputs"][key] = snapshot(gradient)
                payload["grad_output_layouts"][key] = tensor_layout(gradient)
            handles.append(output.register_hook(save_upstream))
        return hook

    try:
        for name, module, kind in [("d", model.d, "data"), *[(name, module, "prior") for name, module in prior.items()]]:
            handles.append(module.register_forward_pre_hook(before(name, kind), with_kwargs=True))
            handles.append(module.register_forward_hook(after(name)))
        gpu = tuple(value.cuda() for value in batch)
        output = model(gpu[0], gpu[1], 3)
        if list(output.shape) != [4, 3, 96, 96]:
            raise RuntimeError("Unexpected full model output shape")
        loss = F.mse_loss(output, gpu[2])
        loss.backward()
        if any(value.grad is None for value in params.values()) or not bool(torch.stack(
                [torch.isfinite(loss), *[torch.isfinite(value.grad).all() for value in params.values()]]).all()):
            raise RuntimeError("Full Python capture has missing/nonfinite parameter gradients")
        for key in parameter_ids:
            payload["captured_parameter_vjps"][key] = snapshot(params[key.removeprefix("parameter/")].grad)
        payload["target_keys"] = [row["input_key"] for row in payload["calls"]] + [
            key for key in payload["leaves"] if not key.startswith("input/")]
        report["structure"] = validate_payload(payload)
        report["capture_loss"] = loss.item()
        report["initial_state_tensor_sha256"] = initial_hash
        report["final_state_tensor_sha256"] = worker.tensor_hash(model.state_dict())
        if report["final_state_tensor_sha256"] != initial_hash:
            raise RuntimeError("Capture changed pretrained parameters or buffers")
        if any((id(params[key.removeprefix("parameter/")]), params[key.removeprefix("parameter/")]._version) != identity
               for key, identity in parameter_ids.items()):
            raise RuntimeError("Capture changed a shared parameter identity/version")
        report["payload_tensor_sha256"] = payload_hash(payload)
        report["parameter_identities"] = {key: dict(object_id=value[0], version=value[1]) for key, value in parameter_ids.items()}
        report["full_model_gradient_tensors"] = 133
        report["optimizer_steps"] = 0
        report["source_unchanged"] = source_hashes() == report["source_sha256"]
        if not report["source_unchanged"]:
            raise RuntimeError("Capture sources changed during execution")
        return payload
    finally:
        for handle in reversed(handles):
            handle.remove()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CAPTURE)
    parser.add_argument("--manifest", type=Path, default=ROOT / "artifacts/dataset_training/split_900_100.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "model_zoo/converse_usrnet.pth")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new capture directory; previous evidence is immutable")
    import torch
    import train_usrnet_dataset as worker
    args.output_dir.mkdir(parents=True)
    report_path = args.output_dir / "capture.json"
    report = dict(status="capturing", scope=__doc__, format=FORMAT, seed=17, step=0, batch_size=4,
                  patch_size=96, scale=3, backend="pytorch", dtype="float32", reuse_training_spectra=False,
                  settings=worker.json_safe(vars(args)), source_sha256=source_hashes(), measured=False,
                  serialization="CPU contiguous logical values plus original shape/stride; replay allocates fresh storage with original strides and offset zero")
    worker.write_json(report_path, report)
    try:
        payload = capture(args, report)
        fixture_path = args.output_dir / "fixture.pt"
        torch.save(payload, fixture_path)
        report.update(status="complete", fixture_file="fixture.pt", fixture_file_sha256=worker.file_hash(fixture_path))
        worker.write_json(report_path, report)
        print(json.dumps(dict(status="complete", capture=str(args.output_dir), structure=report["structure"])), flush=True)
        return 0
    except Exception as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        worker.write_json(report_path, report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
