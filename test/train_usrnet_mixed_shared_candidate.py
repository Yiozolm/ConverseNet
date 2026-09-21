"""Research training adapter: mixed DataNet plus shared-s1 prior, or current.

    python test/train_usrnet_mixed_shared_candidate.py --candidate mixed_shared \
        --seed 17 --run-dir artifacts/native_deconv_target/mixed_shared_seed17

The original train_usrnet_dataset worker owns data, pretrained initialization,
loss, Adam, finite checks, evaluation, saving and .05dB/.001 quality gates.
Only gradient-enabled CUDA FP32 forwards without autocast use the candidate:
FP64 KernelNet and five kernel projections keep their generated kernels double;
five DataNet calls prepare those kernels in FP64 -> c64 and retain the original
FP32 private spectral solver/activation FFT/lambda/IFFT. The 35 prior calls use
SharedTransfer; cat-pad/view-crop keeps the measured C128/96x96/pad2 guard.

No-grad validation uses the original model methods, production public solver
and boundaries throughout. Model parameters and final parameter gradients stay
FP32. Casts remain differentiable and are recomputed on every call. No parameter
or trainable-spectrum cache, custom VJP, worker edit or release claim is added.
This adapter does not edit the earlier shared-s1 training adapter or reports.
"""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = Path(__file__).resolve()
sys.path.insert(0, str(ROOT))
CANDIDATE_ID = "kernel64_mixeddata_sharedprior_training_boundary_v1"
HELPERS = ("test/probe_nearest_training.py", "test/probe_converse_boundaries.py",
           "experiments/training_shared_s1/loader.py", "experiments/training_shared_s1/bindings.cpp",
           "experiments/training_shared_s1/kernels.cu")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_contains(output, name):
    pending, seen = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if name in node.name():
            return True
        pending.extend(child for child, _ in node.next_functions)
    return False


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", choices=("mixed_shared", "current"), default="mixed_shared")
    parser.add_argument("--variant", choices=("current",), default="current")
    selection, worker_args = parser.parse_known_args()
    import train_usrnet_dataset as worker
    original_load, original_execute, original_step = worker.load_backend, worker.execute, worker.train_step
    original_namespace, original_hashes = worker.production_namespace, worker.source_hashes
    shared_files = ("test/train_usrnet_dataset.py", "test/usrnet_training_data.py", "test/evaluate_usrnet_quality.py")
    hashes = {name: file_hash(ROOT / name) for name in (*shared_files, *HELPERS)}
    selected_id = CANDIDATE_ID if selection.candidate == "mixed_shared" else "unchanged_current_control"
    shared_ops, installed = None, False
    active_frames = []
    stats = dict(candidate=selection.candidate, candidate_id=selected_id, model_forwards=0,
        grad_enabled_forwards=0, no_grad_forwards=0, mixed_grad_forwards=0,
        optimizer_gradient_dtype_checks=0, parameter_tensor_count=133,
        first_training_graph_checked=False, first_training_graph_has_shared_transfer=None,
        first_training_graph_has_spectral_solve=None, forward_checks=[])

    def frame():
        if not active_frames:
            raise RuntimeError("A solver or prior module executed outside the counted full-model forward")
        return active_frames[-1]

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", [str(ADAPTER), *worker_args, "--variant", "current"]))

        @contextmanager
        def counted_namespace(ops):
            import torch
            from models import util_converse
            from probe_nearest_training import prepare_kernel
            with original_namespace(ops):
                original_public = torch.ops.converse2d.forward

                def forward(x, prior, weight, bias, scale, eps, variant="v7"):
                    record = frame()
                    record["public_solver_calls"] += 1
                    if any(value.dtype != torch.float32 for value in (x, prior, weight, bias)):
                        raise RuntimeError("Public solver inputs must retain the FP32 model contract")
                    if not record["candidate_active"]:
                        result = original_public(x, prior, weight, bias, scale, eps, variant)
                        record["original_public_calls"] += 1
                    else:
                        eligible = (torch.is_grad_enabled() and not torch.is_autocast_enabled(x.device.type)
                            and (x.requires_grad or weight.requires_grad or bias.requires_grad)
                            and x.is_cuda and prior is x and scale == 1 and variant == "v7"
                            and x.shape[1] == 128 and tuple(weight.shape[-2:]) == (3, 3))
                        if not eligible or shared_ops is None:
                            raise RuntimeError("Only the 35 eligible prior calls may enter SharedTransfer")
                        value = x.contiguous()
                        y = torch.fft.rfft2(value)
                        kernel = prepare_kernel(weight, *value.shape[-2:])
                        regularizer = torch.sigmoid(bias.contiguous() - 9.) + eps
                        spectrum = shared_ops.shared_s1_transfer(y, kernel, regularizer)
                        result = torch.fft.irfft2(spectrum, s=value.shape[-2:])
                        record["shared_prior_calls"] += 1
                    if result.dtype != torch.float32:
                        raise RuntimeError("A public solver returned a non-FP32 activation")
                    record["fp32_solver_outputs"] += 1
                    return result

                with ExitStack() as local:
                    local.enter_context(patch.object(torch.ops.converse2d, "forward", forward))
                    local.enter_context(patch.object(util_converse, "converse2d_CUDA", forward))
                    yield

        def install_model_adapter():
            nonlocal installed
            if installed:
                return
            import torch
            import torch.nn.functional as F
            from models import converse_usrnet, util_converse
            from probe_nearest_training import prepare_kernel
            from probe_converse_boundaries import pad_cat, crop_view
            original_model = converse_usrnet.ConverseUSRNet.forward
            original_layer = util_converse.Converse2D.forward

            def matching_boundary(layer, x):
                backend = (os.environ.get("CONVERSE2D_BACKEND", "") or layer.backend).lower()
                return (x.is_cuda and x.dtype == torch.float32 and x.ndim == 4
                    and x.shape[1] == 128 and tuple(x.shape[-2:]) == (96, 96)
                    and layer.in_channels == 128 and layer.scale == 1 and layer.padding == 2
                    and layer.padding_mode == "circular" and layer.variant == "v7" and backend in ("auto", "cuda"))

            def prior_forward(layer, x):
                record = frame()
                record["prior_module_calls"] += 1
                eligible = (record["candidate_active"] and matching_boundary(layer, x)
                    and torch.is_grad_enabled() and not torch.is_autocast_enabled(x.device.type)
                    and (x.requires_grad or layer.weight.requires_grad or layer.bias.requires_grad))
                if not eligible:
                    return original_layer(layer, x)
                value = pad_cat(x, 2)
                # This must pass through the public dispatcher/counters above.
                result = torch.ops.converse2d.forward(value, value, layer.weight, layer.bias,
                                                     1, float(layer.eps), layer.variant)
                record["boundary_calls"] += 1
                return crop_view(result, 2)

            def model_forward(model, x, kernel, scale):
                parameters = tuple(model.parameters())
                if (model.num_iterations != 5 or len(model.p.m_body) != 7 or len(model.convs) != 5
                        or len(parameters) != 133 or any(value.dtype != torch.float32 for value in parameters)):
                    raise RuntimeError("The unchanged worker must construct the original FP32 full5/7 model")
                if getattr(model, "reuse_training_spectra", False):
                    raise RuntimeError("This candidate does not use a trainable-spectrum cache")
                grad_enabled = torch.is_grad_enabled()
                eligible = (grad_enabled and not torch.is_autocast_enabled(x.device.type)
                    and x.is_cuda and kernel.is_cuda and x.dtype == kernel.dtype == torch.float32
                    and (x.requires_grad or kernel.requires_grad or any(value.requires_grad for value in parameters)))
                enabled = selection.candidate == "mixed_shared" and eligible
                record = dict(index=len(stats["forward_checks"]) + 1, grad_enabled=grad_enabled,
                    model_training=model.training, candidate_active=enabled, scale=int(scale),
                    public_solver_calls=0, original_public_calls=0, mixed_data_calls=0, shared_prior_calls=0,
                    private_spectral_calls=0, logical_solver_calls=0, parameter_storage_dtype="torch.float32",
                    prior_module_calls=0, boundary_calls=0, kernelnet_fp64_calls=0,
                    projection_fp64_calls=[0] * 5, generated_kernel_gradient_calls=0,
                    generated_kernel_gradient_dtypes={}, shared_data_fft_calls=0, fp32_solver_outputs=0,
                    output_dtype=None, forward_verified=False, backward_verified=False)
                stats["forward_checks"].append(record)
                active_frames.append(record)
                try:
                    with ExitStack() as local:
                        if enabled:
                            kernelnet = model.kernelnet

                            def kernel_forward(value):
                                if value.dtype != torch.float32:
                                    raise RuntimeError("The original blur input must enter KernelNet in FP32")
                                batch = value.shape[0]
                                value = value.double().reshape(batch, -1)
                                for index, layer in enumerate((kernelnet.fc1, kernelnet.fc2, kernelnet.fc3)):
                                    value = F.linear(value, layer.weight.double(), None if layer.bias is None else layer.bias.double())
                                    if index < 2:
                                        value = F.gelu(value, approximate=kernelnet.gelu.approximate)
                                record["kernelnet_fp64_calls"] += 1
                                return value.view(batch, 16, kernelnet.kernel_size, kernelnet.kernel_size)

                            def projection(index, layer):
                                if (layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride,
                                        layer.padding, layer.dilation, layer.groups, layer.padding_mode) != (
                                        16, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, "zeros"):
                                    raise RuntimeError("The existing kernel projection geometry changed")

                                def forward(value):
                                    if value.dtype != torch.float64:
                                        raise RuntimeError("Kernel features were rounded before projection")
                                    result = F.conv2d(value, layer.weight.double(), None if layer.bias is None else layer.bias.double(),
                                                      layer.stride, layer.padding, layer.dilation, layer.groups)
                                    record["projection_fp64_calls"][index] += 1
                                    return result
                                return forward

                            def data_forward(value, generated, sf, padding=0, padding_mode="circular"):
                                if (value.dtype != torch.float32 or generated.dtype != torch.float64
                                        or model.d.alpha.dtype != torch.float32 or not generated.requires_grad
                                        or not torch.is_grad_enabled() or torch.is_autocast_enabled(value.device.type)):
                                    raise RuntimeError("Mixed DataNet dtype/gradient contract changed")

                                def kernel_vjp(gradient):
                                    if gradient.dtype != torch.float64:
                                        raise RuntimeError("The generated kernel VJP must stay FP64")
                                    record["generated_kernel_gradient_calls"] += 1
                                    key = str(gradient.dtype)
                                    record["generated_kernel_gradient_dtypes"][key] = record["generated_kernel_gradient_dtypes"].get(key, 0) + 1
                                # The tensor's autograd graph owns this hook; it
                                # captures only a small statistics dict, no tensor.
                                generated.register_hook(kernel_vjp)
                                value = F.pad(value, (padding,) * 4, mode=padding_mode, value=0) if padding else value
                                value = value.contiguous()
                                height, width = value.shape[-2:]
                                regularizer = torch.sigmoid(model.d.alpha.contiguous() - 9.) + model.d.eps
                                prepared = prepare_kernel(generated, height * sf, width * sf)
                                y = torch.fft.rfft2(value)
                                p = y if sf == 1 else torch.fft.rfft2(F.interpolate(value, scale_factor=sf, mode="nearest"))
                                if (y.dtype != torch.complex64 or p.dtype != torch.complex64
                                        or prepared.dtype != torch.complex64 or not prepared.is_contiguous()
                                        or regularizer.dtype != torch.float32):
                                    raise RuntimeError("DataNet did not retain its FP32 spectral boundary")
                                if sf == 1:
                                    if p is not y:
                                        raise RuntimeError("Shared DataNet s1 acquired a duplicate activation FFT")
                                    record["shared_data_fft_calls"] += 1
                                corrected = torch.ops.converse2d._training_spectral(y, p, prepared, regularizer, height, width, sf)
                                record["private_spectral_calls"] += 1
                                result = torch.fft.irfft2(corrected, s=(height * sf, width * sf))
                                if corrected.dtype != torch.complex64 or result.dtype != torch.float32:
                                    raise RuntimeError("Mixed DataNet returned a non-FP32 activation")
                                record["mixed_data_calls"] += 1
                                record["fp32_solver_outputs"] += 1
                                crop = padding * sf
                                return result[..., crop:-crop, crop:-crop] if crop else result

                            local.enter_context(patch.object(kernelnet, "forward", kernel_forward))
                            for index, layer in enumerate(model.convs):
                                local.enter_context(patch.object(layer, "forward", projection(index, layer)))
                            local.enter_context(patch.object(model.d, "forward", data_forward))
                        result = original_model(model, x, kernel, scale)
                    if result.dtype != torch.float32:
                        raise RuntimeError("The full model must retain FP32 output")
                    record["output_dtype"] = str(result.dtype)
                    record["logical_solver_calls"] = record["public_solver_calls"] + record["private_spectral_calls"]
                    expected_boundary = 35 if enabled and (x.shape[-2] * scale, x.shape[-1] * scale) == (96, 96) else 0
                    expected = dict(public_solver_calls=35 if enabled else 40,
                        original_public_calls=0 if enabled else 40, mixed_data_calls=5 if enabled else 0,
                        private_spectral_calls=5 if enabled else 0, logical_solver_calls=40,
                        shared_prior_calls=35 if enabled else 0, prior_module_calls=35,
                        boundary_calls=expected_boundary, kernelnet_fp64_calls=1 if enabled else 0,
                        projection_fp64_calls=[1 if enabled else 0] * 5,
                        shared_data_fft_calls=(5 if scale == 1 else 4) if enabled else 0,
                        fp32_solver_outputs=40)
                    if any(record[key] != value for key, value in expected.items()):
                        raise RuntimeError(f"Full-forward route coverage mismatch: {record}; expected {expected}")
                    record["forward_verified"] = True
                    stats["model_forwards"] += 1
                    stats["grad_enabled_forwards" if grad_enabled else "no_grad_forwards"] += 1
                    stats["mixed_grad_forwards"] += int(enabled)
                    if grad_enabled and not stats["first_training_graph_checked"]:
                        shared = graph_contains(result, "SharedTransfer")
                        spectral = graph_contains(result, "SpectralSolve")
                        if not result.requires_grad or shared != enabled or not spectral:
                            raise RuntimeError("The training graph lost SharedTransfer or the five unchanged private DataNet solvers")
                        stats["first_training_graph_checked"] = True
                        stats["first_training_graph_has_shared_transfer"] = shared
                        stats["first_training_graph_has_spectral_solve"] = spectral
                    return result
                finally:
                    if active_frames.pop() is not record:
                        raise RuntimeError("Full-model frame stack became inconsistent")

            stack.enter_context(patch.object(util_converse.Converse2D, "forward", prior_forward))
            stack.enter_context(patch.object(converse_usrnet.ConverseUSRNet, "forward", model_forward))
            installed = True

        def load_backend(args):
            nonlocal shared_ops
            current_ops, manifest = original_load(args)
            if selection.candidate == "mixed_shared":
                from experiments.training_shared_s1.loader import load
                shared_ops, shared_build = load(verbose=args.verbose_build)
                manifest = dict(kind="original inference plus isolated mixed-DataNet/shared-prior training",
                    production_backend=manifest, shared_training_build=shared_build,
                    candidate_id=selected_id, no_grad_inference="unchanged production")
            install_model_adapter()
            return current_ops, manifest

        def train_step(model, optimizer, batch, args):
            import torch
            start = len(stats["forward_checks"])

            def before_optimizer(_optimizer, _args, _kwargs):
                parameters = tuple(model.parameters())
                if len(parameters) != 133 or any(value.dtype != torch.float32 or value.grad is None
                        or value.grad.dtype != torch.float32 for value in parameters):
                    raise RuntimeError("All 133 parameter storages and final gradients must be FP32 before Adam")
                frames = stats["forward_checks"][start:]
                if len(frames) != args.batch_size // args.microbatch_size:
                    raise RuntimeError("Unexpected number of microbatch forwards before Adam")
                for record in frames:
                    expected = 5 if record["candidate_active"] else 0
                    if record["generated_kernel_gradient_calls"] != expected or record["generated_kernel_gradient_dtypes"] != (
                            {"torch.float64": 5} if expected else {}):
                        raise RuntimeError("Generated-kernel FP64 gradients were missing or cast at the wrong boundary")
                    record["backward_verified"] = True
                    record["parameter_gradient_dtype"] = "torch.float32"
                    record["parameter_gradient_tensors"] = 133
                stats["optimizer_gradient_dtype_checks"] += 1

            handle = optimizer.register_step_pre_hook(before_optimizer)
            try:
                result = original_step(model, optimizer, batch, args)
            finally:
                handle.remove()
            if result["gradient_tensor_count"] != 133:
                raise RuntimeError("The unchanged worker did not collect all 133 parameter gradients")
            return result

        def source_hashes():
            result = original_hashes()
            result[ADAPTER.relative_to(ROOT).as_posix()] = file_hash(ADAPTER)
            result.update({name: file_hash(ROOT / name) for name in HELPERS})
            return result

        def execute(args, protocol, report):
            report["comparison_candidate"] = selection.candidate
            report["candidate_id"] = selected_id
            report["candidate_verification"] = stats
            report["candidate_adapter"] = dict(path=str(ADAPTER), sha256=file_hash(ADAPTER),
                shared_worker_files_sha256={name: hashes[name] for name in shared_files},
                helper_source_sha256={name: hashes[name] for name in HELPERS},
                precision_scope="FP64 only for differentiable KernelNet/projections and kernel preparation; activations/solvers/parameters/final gradients FP32",
                inference_policy="All no_grad validation uses unchanged production model methods, public solver and boundaries",
                quality_policy="Original paired RGB/Y PSNR .05dB and SSIM .001 gates unchanged; numerical baseline is Python FP32 error",
                instrumentation_scope="Per-forward routing/dtype counts and pre-Adam metadata checks; included in observed research-run time",
                production_eligible=False)
            original_execute(args, protocol, report)
            if not stats["first_training_graph_checked"] or stats["optimizer_gradient_dtype_checks"] != args.steps:
                raise RuntimeError("Training graph/dtype verification did not cover every update")
            if selection.candidate == "mixed_shared" and stats["mixed_grad_forwards"] < 1:
                raise RuntimeError("The selected mixed candidate was never used")
            for record in stats["forward_checks"]:
                if not record["forward_verified"] or (record["grad_enabled"] and not record["backward_verified"]):
                    raise RuntimeError("A full forward/backward was not verified")
                if not record["grad_enabled"] and (record["mixed_data_calls"] or record["shared_prior_calls"] or record["boundary_calls"]):
                    raise RuntimeError("A candidate contaminated no_grad evaluation")
            if any(file_hash(ROOT / name) != digest for name, digest in hashes.items()):
                raise RuntimeError("An unchanged worker/helper source changed during training")

        stack.enter_context(patch.object(worker, "load_backend", load_backend))
        stack.enter_context(patch.object(worker, "production_namespace", counted_namespace))
        stack.enter_context(patch.object(worker, "train_step", train_step))
        stack.enter_context(patch.object(worker, "source_hashes", source_hashes))
        stack.enter_context(patch.object(worker, "execute", execute))
        worker.__doc__ = __doc__ + "\n\nUnchanged worker options:\n" + worker.__doc__
        worker.main()


if __name__ == "__main__":
    main()
