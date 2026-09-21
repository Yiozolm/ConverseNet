"""Research adapter: unchanged real-data worker, current vs shared-s1 CUDA.

    python test/train_usrnet_shared_s1_candidate.py --candidate shared_cuda \
        --seed 17 --run-dir artifacts/native_deconv_target/shared_cuda_seed17

All remaining arguments go to train_usrnet_dataset unchanged. Its checkpoint,
full5/7 model, data order, loss/optimizer, finite checks and .05dB/.001 quality
gates remain intact. This entry does not declare global numerical/quality or
production eligibility, and does not use the old FP64 pointwise release gate.

SharedTransfer applies only to CUDA FP32/v7, scale1, x0 IS x, grad enabled,
and at least one of x/weight/bias requiring gradients. FFT/IFFT, FP64 kernel
preparation -> complex64 and lambda retain automatic differentiation. No
trainable spectra or parameter values are cached across calls.

cat pad / view crop applies only inside that same training regime and the
previously measured C128/96x96/scale1/circular-pad2 module geometry. Its solver
call goes through the counted public dispatcher. eval()+grad still qualifies;
no_grad validation uses original production solver AND boundary code. Thus
final parameters are evaluated through the same unchanged inference route.

Each full forward is checked: 40 solver calls, with 39 training replacements
and 35 boundary replacements for the intended full5/7 HR96/s3 recipe; no_grad
gets zero replacements. Different legal worker geometries retain the explicit
boundary guard and record their actual coverage instead of silently expanding it.
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
CANDIDATE_ID = "shared_s1_cuda_training_catpad_cropview_c128_96_v1"
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
    parser.add_argument("--candidate", choices=("current", "shared_cuda"), default="current")
    parser.add_argument("--variant", choices=("current",), default="current")
    selection, worker_args = parser.parse_known_args()
    import train_usrnet_dataset as worker
    original_load, original_execute = worker.load_backend, worker.execute
    original_hashes, original_namespace = worker.source_hashes, worker.production_namespace
    shared_ops = None
    installed = False
    selected_id = CANDIDATE_ID if selection.candidate == "shared_cuda" else "unchanged_current_control"
    shared_files = ("test/train_usrnet_dataset.py", "test/usrnet_training_data.py", "test/evaluate_usrnet_quality.py")
    hashes = {name: file_hash(ROOT / name) for name in (*shared_files, *HELPERS)}
    stats = dict(candidate=selection.candidate, candidate_id=selected_id,
        solver_calls=0, grad_solver_calls=0, no_grad_solver_calls=0,
        eligible_grad_solver_calls=0, candidate_solver_calls=0,
        candidate_no_grad_solver_calls=0, original_solver_calls=0,
        prior_module_calls=0, boundary_geometry_calls=0, eligible_grad_boundary_calls=0,
        candidate_boundary_calls=0, candidate_no_grad_boundary_calls=0,
        model_forwards=0, grad_enabled_forwards=0, no_grad_forwards=0,
        first_training_graph_checked=False, first_training_graph_has_shared_transfer=None,
        first_training_graph_has_spectral_solve=None, forward_checks=[])

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", [str(ADAPTER), *worker_args, "--variant", "current"]))

        @contextmanager
        def counted_namespace(ops):
            import torch
            from models import util_converse
            from probe_nearest_training import prepare_kernel
            with original_namespace(ops):
                original_forward = torch.ops.converse2d.forward

                def forward(x, prior, weight, bias, scale, eps, variant="v7"):
                    grad_mode = torch.is_grad_enabled()
                    stats["solver_calls"] += 1
                    stats["grad_solver_calls" if grad_mode else "no_grad_solver_calls"] += 1
                    eligible = (grad_mode and (x.requires_grad or weight.requires_grad or bias.requires_grad)
                        and x.is_cuda and not torch.is_autocast_enabled(x.device.type)
                        and x.dtype == torch.float32 and weight.dtype == torch.float32
                        and bias.dtype == torch.float32 and scale == 1 and prior is x and variant == "v7")
                    stats["eligible_grad_solver_calls"] += int(eligible)
                    if selection.candidate != "shared_cuda" or not eligible:
                        stats["original_solver_calls"] += 1
                        return original_forward(x, prior, weight, bias, scale, eps, variant)
                    if shared_ops is None:
                        raise RuntimeError("Shared CUDA was selected but its isolated extension is unavailable")
                    value = x.contiguous()
                    y = torch.fft.rfft2(value)
                    kernel = prepare_kernel(weight, *value.shape[-2:])
                    regularizer = torch.sigmoid(bias.contiguous() - 9.) + eps
                    spectrum = shared_ops.shared_s1_transfer(y, kernel, regularizer)
                    output = torch.fft.irfft2(spectrum, s=value.shape[-2:])
                    stats["candidate_solver_calls"] += 1
                    stats["candidate_no_grad_solver_calls"] += int(not grad_mode)
                    return output

                with ExitStack() as namespace_stack:
                    namespace_stack.enter_context(patch.object(torch.ops.converse2d, "forward", forward))
                    namespace_stack.enter_context(patch.object(util_converse, "converse2d_CUDA", forward))
                    yield

        def install_model_adapter():
            nonlocal installed
            if installed:
                return
            import torch
            from models import converse_usrnet, util_converse
            from probe_converse_boundaries import pad_cat, crop_view
            layer_class, model_class = util_converse.Converse2D, converse_usrnet.ConverseUSRNet
            original_layer_forward, original_init = layer_class.forward, model_class.__init__

            def layer_matches(layer):
                backend = (os.environ.get("CONVERSE2D_BACKEND", "") or layer.backend).lower()
                return (layer.scale == 1 and layer.padding == 2 and layer.padding_mode == "circular"
                        and layer.in_channels == 128 and layer.variant == "v7" and backend in ("auto", "cuda"))

            def guarded_layer_forward(layer, x):
                stats["prior_module_calls"] += 1
                geometry = (x.is_cuda and x.dtype == torch.float32 and x.ndim == 4
                            and x.shape[1] == 128 and tuple(x.shape[-2:]) == (96, 96) and layer_matches(layer))
                stats["boundary_geometry_calls"] += int(geometry)
                grad_mode = torch.is_grad_enabled()
                eligible = (geometry and grad_mode and not torch.is_autocast_enabled(x.device.type)
                            and (x.requires_grad or layer.weight.requires_grad or layer.bias.requires_grad))
                stats["eligible_grad_boundary_calls"] += int(eligible)
                if selection.candidate != "shared_cuda" or not eligible:
                    return original_layer_forward(layer, x)
                padded = pad_cat(x, 2)
                # Intentionally use the patched real call site: never bypass
                # the solver dispatcher/counters by calling the private op here.
                output = torch.ops.converse2d.forward(padded, padded, layer.weight, layer.bias,
                                                     1, float(layer.eps), layer.variant)
                stats["candidate_boundary_calls"] += 1
                stats["candidate_no_grad_boundary_calls"] += int(not grad_mode)
                return crop_view(output, 2)

            def configured_init(model, *args, **kwargs):
                original_init(model, *args, **kwargs)
                model.reuse_training_spectra = False
                prior_layers = [layer for layer in model.p.modules() if isinstance(layer, layer_class)]
                if model.num_iterations != 5 or len(prior_layers) != 7:
                    raise RuntimeError("The unchanged worker must construct the full5/7 model")
                active = []
                keys = ("solver_calls", "eligible_grad_solver_calls", "candidate_solver_calls",
                        "prior_module_calls", "boundary_geometry_calls", "eligible_grad_boundary_calls", "candidate_boundary_calls")

                def before_forward(module, inputs):
                    if getattr(module, "reuse_training_spectra", False):
                        raise RuntimeError("No spectrum/parameter cache is enabled by this adapter")
                    x, _kernel, scale = inputs
                    matching_boundary = (x.is_cuda and x.dtype == torch.float32
                                         and (x.shape[-2] * scale, x.shape[-1] * scale) == (96, 96))
                    active.append(dict(before={key: stats[key] for key in keys},
                        grad_enabled=torch.is_grad_enabled(), model_training=module.training,
                        all_parameters_trainable=all(value.requires_grad for value in module.parameters()),
                        scale=int(scale), expected_shared_calls=40 if scale == 1 else 39,
                        expected_boundary_geometry=35 if matching_boundary and all(layer_matches(layer) for layer in prior_layers) else 0))

                def after_forward(_module, _inputs, output):
                    if not active:
                        return
                    frame = active.pop()
                    if output is None:
                        return
                    delta = {key: stats[key] - frame["before"][key] for key in keys}
                    if delta["solver_calls"] != 40 or delta["prior_module_calls"] != 35:
                        raise RuntimeError(f"Full forward did not use exactly 40 solvers/35 prior modules: {delta}")
                    if delta["boundary_geometry_calls"] != frame["expected_boundary_geometry"]:
                        raise RuntimeError(f"Boundary geometry coverage changed: {delta}")
                    if frame["grad_enabled"] and frame["all_parameters_trainable"]:
                        if delta["eligible_grad_solver_calls"] != frame["expected_shared_calls"]:
                            raise RuntimeError(f"Shared-input training solver coverage changed: {delta}")
                        if delta["eligible_grad_boundary_calls"] != frame["expected_boundary_geometry"]:
                            raise RuntimeError(f"Training boundary guard coverage changed: {delta}")
                    wanted_solver = delta["eligible_grad_solver_calls"] if selection.candidate == "shared_cuda" else 0
                    wanted_boundary = delta["eligible_grad_boundary_calls"] if selection.candidate == "shared_cuda" else 0
                    if delta["candidate_solver_calls"] != wanted_solver or delta["candidate_boundary_calls"] != wanted_boundary:
                        raise RuntimeError(f"A candidate was skipped or escaped its guard: {delta}")
                    if not frame["grad_enabled"] and (wanted_solver or wanted_boundary):
                        raise RuntimeError("no_grad evaluation invoked a training candidate")
                    stats["model_forwards"] += 1
                    stats["grad_enabled_forwards" if frame["grad_enabled"] else "no_grad_forwards"] += 1
                    stats["forward_checks"].append(dict(index=stats["model_forwards"],
                        grad_enabled=frame["grad_enabled"], model_training=frame["model_training"], scale=frame["scale"],
                        solver_calls=delta["solver_calls"], candidate_solver_calls=delta["candidate_solver_calls"],
                        boundary_calls=delta["candidate_boundary_calls"]))
                    if frame["grad_enabled"] and not stats["first_training_graph_checked"]:
                        shared = graph_contains(output, "SharedTransfer")
                        spectral = graph_contains(output, "SpectralSolve")
                        expected_shared = selection.candidate == "shared_cuda" and wanted_solver > 0
                        expected_spectral = selection.candidate != "shared_cuda" or frame["scale"] != 1
                        if not output.requires_grad or shared != expected_shared or spectral != expected_spectral:
                            raise RuntimeError("The first training graph did not contain the expected CUDA solver routes")
                        stats["first_training_graph_checked"] = True
                        stats["first_training_graph_has_shared_transfer"] = shared
                        stats["first_training_graph_has_spectral_solve"] = spectral

                model.register_forward_pre_hook(before_forward)
                model.register_forward_hook(after_forward, always_call=True)

            stack.enter_context(patch.object(layer_class, "forward", guarded_layer_forward))
            stack.enter_context(patch.object(model_class, "__init__", configured_init))
            installed = True

        def load_backend(args):
            nonlocal shared_ops
            current_ops, manifest = original_load(args)
            if selection.candidate == "shared_cuda":
                from experiments.training_shared_s1.loader import load
                shared_ops, shared_build = load(verbose=args.verbose_build)
                manifest = dict(kind="current inference plus isolated shared-s1 CUDA training candidate",
                                production_backend=manifest, shared_training_build=shared_build,
                                candidate_id=selected_id, no_grad_inference="unchanged production")
            install_model_adapter()
            return current_ops, manifest

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
                numerical_baseline="Original Python full-spectrum FP32 error/quality; prior absolute FP64 gate is diagnostic only",
                inference_policy="All no_grad evaluation uses original production solver and boundaries for both candidates",
                quality_policy="Original worker's paired final RGB/Y PSNR drop<=.05dB and SSIM drop<=.001 remain unchanged",
                production_eligible=False)
            original_execute(args, protocol, report)
            if not stats["first_training_graph_checked"] or stats["grad_enabled_forwards"] < 1:
                raise RuntimeError("No verified gradient-enabled full forward was observed")
            if stats["candidate_no_grad_solver_calls"] or stats["candidate_no_grad_boundary_calls"]:
                raise RuntimeError("Validation was contaminated by a training-only candidate")
            if any(file_hash(ROOT / name) != digest for name, digest in hashes.items()):
                raise RuntimeError("A frozen worker/helper source changed during the run")

        stack.enter_context(patch.object(worker, "load_backend", load_backend))
        stack.enter_context(patch.object(worker, "production_namespace", counted_namespace))
        stack.enter_context(patch.object(worker, "source_hashes", source_hashes))
        stack.enter_context(patch.object(worker, "execute", execute))
        worker.__doc__ = __doc__ + "\n\nUnchanged worker arguments and recipe:\n" + worker.__doc__
        worker.main()


if __name__ == "__main__":
    main()
