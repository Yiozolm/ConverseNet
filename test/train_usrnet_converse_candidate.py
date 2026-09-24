"""Thin full-model quality adapter: unchanged current versus isolated combined.

All remaining arguments are passed to train_usrnet_dataset.main. Its strict
pretrained state, Adam/MSE, real-data order, validation and quality gates remain
unchanged. This adapter does not declare a candidate production-ready.

    python test/train_usrnet_converse_candidate.py --candidate combined --seed 17 \
        --run-dir artifacts/native_deconv_target/combined_seed17

Combined uses the isolated forced-s1 namespace. Only matching CUDA FP32 v7
Converse2D calls with C128/96x96, scale1, circular padding2 replace pad/slice with
pad_cat/crop_view. Other module forwards remain original. The namespace's s1
kernel selection also applies to eligible DataNet s1 calls; this is recorded
separately from the guarded boundary transformation. Optional spectrum reuse
stays disabled. Class symbols and all production files remain unchanged.
"""
import argparse
from contextlib import ExitStack
import hashlib
import os
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = Path(__file__).resolve()
CANDIDATE_ID = "isolated_forced_s1_catpad_cropview_c128_96_v1"
HELPERS = ("probe_training_s1_shapes.py", "probe_converse_boundaries.py", "probe_pointwise_training.py")


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def graph_has_spectral(output):
    pending, seen = [output.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        if any(kind in node.name() for kind in ("SpectralSolve", "FullSolve")):
            return True
        pending.extend(child for child, _ in node.next_functions)
    return False


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", choices=("current", "combined"), default="current")
    parser.add_argument("--variant", choices=("current",), default="current")
    adapter_args, worker_args = parser.parse_known_args()
    sys.path.insert(0, str(ROOT))
    import train_usrnet_dataset as worker
    original_load, original_execute, original_hashes = worker.load_backend, worker.execute, worker.source_hashes
    selected_id = CANDIDATE_ID if adapter_args.candidate == "combined" else "unchanged_current_control"
    shared_hashes = {name: file_hash(ROOT / "test" / name) for name in
                     ("train_usrnet_dataset.py", "usrnet_training_data.py", "evaluate_usrnet_quality.py")}
    helper_hashes = {name: file_hash(ROOT / "test" / name) for name in HELPERS}
    stats = dict(candidate=adapter_args.candidate, candidate_id=selected_id,
                 model_forwards=0, grad_enabled_forwards=0, inference_forwards=0,
                 prior_calls=0, prior_training_calls=0, prior_inference_calls=0,
                 guarded_prior_calls=0, guarded_prior_training_calls=0, guarded_prior_inference_calls=0,
                 candidate_boundary_calls=0, candidate_boundary_training_calls=0,
                 candidate_boundary_inference_calls=0, expected_prior_calls_per_model_forward=35,
                 first_training_graph_checked=False, first_training_graph_has_spectral_solve=None)

    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", [str(ADAPTER), *worker_args, "--variant", "current"]))
        installed = False

        def install_model_adapter():
            nonlocal installed
            if installed:
                return
            import torch
            from models import converse_usrnet, util_converse
            from probe_converse_boundaries import pad_cat, crop_view
            layer_class, model_class = util_converse.Converse2D, converse_usrnet.ConverseUSRNet
            original_forward, original_init = layer_class.forward, model_class.__init__

            def layer_matches(layer):
                backend = (os.environ.get("CONVERSE2D_BACKEND", "") or layer.backend).lower()
                return (layer.scale == 1 and layer.padding == 2 and layer.padding_mode == "circular"
                        and layer.in_channels == 128 and layer.variant == "v7" and backend in ("auto", "cuda"))

            def guarded_forward(layer, x):
                mode = "training" if torch.is_grad_enabled() else "inference"
                stats["prior_calls"] += 1
                stats[f"prior_{mode}_calls"] += 1
                eligible = (x.is_cuda and x.dtype == torch.float32 and x.ndim == 4
                            and x.shape[1] == 128 and tuple(x.shape[-2:]) == (96, 96) and layer_matches(layer))
                if eligible:
                    stats["guarded_prior_calls"] += 1
                    stats[f"guarded_prior_{mode}_calls"] += 1
                if adapter_args.candidate == "combined" and eligible:
                    stats["candidate_boundary_calls"] += 1
                    stats[f"candidate_boundary_{mode}_calls"] += 1
                    padded = pad_cat(x, 2)
                    output = torch.ops.converse2d.forward(padded, padded, layer.weight, layer.bias,
                                                         1, float(layer.eps), layer.variant)
                    return crop_view(output, 2)
                return original_forward(layer, x)

            def configured_init(model, *args, **kwargs):
                # Keep the original class symbol: its explicit super(...) must
                # continue to reference an actual class, never a factory.
                original_init(model, *args, **kwargs)
                if hasattr(model, "reuse_training_spectra"):
                    model.reuse_training_spectra = False
                prior_layers = [layer for layer in model.p.modules() if isinstance(layer, layer_class)]
                expected = model.num_iterations * len(prior_layers)
                if expected != 35:
                    raise RuntimeError(f"Expected unchanged full model with 35 prior calls, got {expected}")
                stats["prior_modules"] = len(prior_layers)
                stats["model_iterations"] = model.num_iterations
                active = []

                def before_forward(module, inputs):
                    if getattr(module, "reuse_training_spectra", False):
                        raise RuntimeError("This candidate does not enable spectrum reuse")
                    x, _, scale = inputs
                    matching_shape = (x.is_cuda and x.dtype == torch.float32 and
                                      (x.shape[-2] * scale, x.shape[-1] * scale) == (96, 96))
                    expected_guarded = (module.num_iterations * sum(layer_matches(layer) for layer in prior_layers)
                                        if matching_shape else 0)
                    active.append((stats["prior_calls"], stats["guarded_prior_calls"],
                                   stats["candidate_boundary_calls"], expected_guarded))

                def after_forward(_module, _inputs, output):
                    if not active:
                        return
                    previous, guarded, applied, expected_guarded = active.pop()
                    if output is None:
                        return
                    if stats["prior_calls"] - previous != expected:
                        raise RuntimeError("Full forward did not execute exactly 35 prior solvers")
                    if stats["guarded_prior_calls"] - guarded != expected_guarded:
                        raise RuntimeError("Observed candidate guard coverage differs from model/input geometry")
                    wanted = expected_guarded if adapter_args.candidate == "combined" else 0
                    if stats["candidate_boundary_calls"] - applied != wanted:
                        raise RuntimeError("Boundary candidate was skipped or applied outside its guard")
                    stats["model_forwards"] += 1
                    mode = "grad_enabled_forwards" if torch.is_grad_enabled() else "inference_forwards"
                    stats[mode] += 1
                    if torch.is_grad_enabled() and not stats["first_training_graph_checked"]:
                        found = graph_has_spectral(output)
                        if not output.requires_grad or not found:
                            raise RuntimeError("Training lost the verified CUDA spectral/autograd route")
                        stats["first_training_graph_checked"] = True
                        stats["first_training_graph_has_spectral_solve"] = found

                model.register_forward_pre_hook(before_forward)
                model.register_forward_hook(after_forward, always_call=True)

            stack.enter_context(patch.object(layer_class, "forward", guarded_forward))
            stack.enter_context(patch.object(model_class, "__init__", configured_init))
            installed = True

        def load_backend(args):
            if args.variant != "current":
                raise ValueError("This adapter only selects current or the explicit combined candidate")
            current_ops, current_manifest = original_load(args)
            if adapter_args.candidate == "combined":
                import probe_training_s1_shapes as s1probe
                before = s1probe.source_hashes()
                forced_ops, forced_manifest = s1probe.load_forced(args.verbose_build)
                if before != s1probe.source_hashes() or before != forced_manifest["source_sha256"]:
                    raise RuntimeError("Production source changed during isolated candidate setup")
                compiled = current_manifest["build_manifest"]["inputs"]["sources"]
                if any(compiled.get(name) != value for name, value in before.items()):
                    raise RuntimeError("Verified current and forced builds use different source revisions")
                result = (forced_ops, dict(kind="isolated combined Converse candidate", candidate_id=selected_id,
                                          forced_build=forced_manifest, verified_current_build=current_manifest,
                                          production_source_unchanged=True,
                                          forced_scope="s1 training dispatch throughout isolated namespace, including DataNet s1 calls",
                                          tensor_boundary_scope="Only guarded C128/96x96 scale1 circular-pad2 Converse2D calls"))
            else:
                result = (current_ops, current_manifest)
            install_model_adapter()
            return result

        def source_hashes():
            result = original_hashes()
            result[ADAPTER.relative_to(ROOT).as_posix()] = file_hash(ADAPTER)
            result.update({"test/" + name: file_hash(ROOT / "test" / name) for name in HELPERS})
            return result

        def execute(args, protocol, report):
            report["comparison_candidate"] = adapter_args.candidate
            report["candidate_id"] = selected_id
            report["candidate_adapter"] = dict(path=str(ADAPTER), sha256=file_hash(ADAPTER),
                                               shared_worker_files_sha256=shared_hashes,
                                               helper_source_sha256=helper_hashes,
                                               policy="Isolated experimental comparison; does not mark combined as current production or quality-approved")
            report["candidate_verification"] = stats
            original_execute(args, protocol, report)
            if not stats["first_training_graph_checked"] or stats["grad_enabled_forwards"] < 1:
                raise RuntimeError("No verified training forward was observed")
            for name, expected in {**shared_hashes, **helper_hashes}.items():
                if file_hash(ROOT / "test" / name) != expected:
                    raise RuntimeError(f"Source changed during candidate run: {name}")

        stack.enter_context(patch.object(worker, "load_backend", load_backend))
        stack.enter_context(patch.object(worker, "source_hashes", source_hashes))
        stack.enter_context(patch.object(worker, "execute", execute))
        worker.__doc__ = __doc__ + "\n\nUnchanged worker arguments and recipe:\n" + worker.__doc__
        worker.main()


if __name__ == "__main__":
    main()
