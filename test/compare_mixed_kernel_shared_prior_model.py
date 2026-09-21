"""Full-model accuracy: FP64 small kernel branch + mixed DataNet + SharedTransfer prior.

    ./experiments/training_speed/run.ps1 test/compare_mixed_kernel_shared_prior_model.py

Reuse the fixed 136-tensor comparison, one FP64 diagnostic oracle and original
Python FP32 error baseline. Keep all five DataNet calls on the frozen mixed
route: FP64 generated kernels/preparation -> c64, FP32 activations/lambda/
private _training_spectral/IFFT. Replace only the 35 C128/k3/shared-s1 prior
calls with the isolated CUDA SharedTransfer. Do not replace the four DataNet
s1 calls with SharedTransfer; that would confound this ablation.

Require 40 total calls, 5 mixed DataNet, 35 SharedTransfer prior, five generated
kernel gradient hooks in FP64 and 35 prior activation-spectrum VJPs in c64.
No production/training file changes, timing, trainable cache, tolerance margin
or quality claim. Old absolute-FP64 pointwise failures remain diagnostic only.
"""
import argparse
from contextlib import ExitStack, contextmanager
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import compare_shared_s1_accuracy_to_python as audit
import diagnose_kernel_solver_boundary as boundary

MODE = "kernel64_mixeddata_sharedprior"
FROZEN_FILES = {
    "test/compare_shared_s1_accuracy_to_python.py": "a7b907c919ee8c218c590e1f2f81cdb1cd541b067847a8872f3f40a349830145",
    "test/diagnose_kernel_solver_boundary.py": "96b82c51a839a6357c405ad35a8ac1b52d30a94da5c828d0dfcbf75032b050d3",
    "test/diagnose_kernel_generation_precision.py": "2d71227f269e16c6cb9ca5a2ef81ab196c77c72a2ff11ea368c5e127c752d117",
}


@contextmanager
def combined_context(shared_ops):
    import torch
    from models import util_converse
    from probe_nearest_training import prepare_kernel

    if shared_ops is None:
        raise RuntimeError("The isolated SharedTransfer extension was not loaded")
    handles = []
    with boundary.boundary_context("kernel64_prepare64_spectral32") as stats:
        stats.update(prior_shared_cuda_calls=0, prior_activation_gradient_calls=0,
                     prior_activation_gradient_dtypes={}, unexpected_public_calls=0)

        def check_activation_gradient(gradient):
            if gradient.dtype != torch.complex64:
                raise RuntimeError("The shared prior activation VJP left complex64")
            stats["prior_activation_gradient_calls"] += 1
            key = str(gradient.dtype)
            stats["prior_activation_gradient_dtypes"][key] = stats["prior_activation_gradient_dtypes"].get(key, 0) + 1

        def prior_forward(x, prior, weight, bias, scale, eps, variant="v7"):
            # DataNet's frozen mixed route invokes _training_spectral directly,
            # so every public call intercepted here must be a prior solver.
            eligible = (x.is_cuda and x.dtype == torch.float32 and prior is x and scale == 1
                        and weight.dtype == bias.dtype == torch.float32 and variant == "v7"
                        and x.shape[1] == 128 and tuple(weight.shape[-2:]) == (3, 3)
                        and torch.is_grad_enabled() and not torch.is_autocast_enabled(x.device.type))
            if not eligible:
                stats["unexpected_public_calls"] += 1
                raise RuntimeError("A non-prior call attempted to enter the SharedTransfer ablation")
            # Preserve honest logical coverage. These are replacements, not
            # executions of generation_context's original FP32 prior route.
            stats["total_calls"] += 1
            stats["scales"]["1"] = stats["scales"].get("1", 0) + 1
            stats["shared_s1_calls"] += 1
            stats["prior_fp32_input_calls"] += 1
            value = x.contiguous()
            y = torch.fft.rfft2(value)
            if not y.requires_grad:
                raise RuntimeError("The fixed full-model prior activation lost its gradient connection")
            handles.append(y.register_hook(check_activation_gradient))
            kernel = prepare_kernel(weight, *value.shape[-2:])
            regularizer = torch.sigmoid(bias.contiguous() - 9.) + eps
            if y.dtype != torch.complex64 or kernel.dtype != torch.complex64 or regularizer.dtype != torch.float32:
                raise RuntimeError("The SharedTransfer prior did not use the original FP32 boundary")
            spectrum = shared_ops.shared_s1_transfer(y, kernel, regularizer)
            output = torch.fft.irfft2(spectrum, s=value.shape[-2:])
            if spectrum.dtype != torch.complex64 or output.dtype != torch.float32:
                raise RuntimeError("The prior solver returned the wrong dtype")
            stats["prior_shared_cuda_calls"] += 1
            return output

        try:
            with ExitStack() as stack:
                stack.enter_context(patch.object(torch.ops.converse2d, "forward", prior_forward))
                stack.enter_context(patch.object(util_converse, "converse2d_CUDA", prior_forward))
                yield stats
        finally:
            for handle in handles:
                handle.remove()


def verify_combined(stats):
    expected = dict(total_calls=40, shared_s1_calls=39, scales={"2": 1, "1": 39}, model_forwards=1,
        kernelnet_fp64_calls=1, kernel_projection_fp64_calls=[1] * 5,
        datanet_module_calls=5, datanet_module_fp32_returns=5, datanet_fp64_operator_calls=0,
        datanet_shared_s1_calls=4, prior_fp32_input_calls=35, prior_original_fp32_calls=0,
        prior_fp64_reference_calls=0, data_public_fp32_calls=0, data_private_fp32_calls=5,
        data_kernel_float_casts=0, data_kernel_fp64_preparations=5, shared_activation_fft_calls=4,
        generated_kernel_gradient_calls=5, generated_kernel_gradient_dtypes={"torch.float64": 5},
        prior_shared_cuda_calls=35, prior_activation_gradient_calls=35,
        prior_activation_gradient_dtypes={"torch.complex64": 35}, unexpected_public_calls=0)
    if any(stats[key] != value for key, value in expected.items()):
        raise RuntimeError(f"The combined route did not preserve the required ablation: {stats}; expected {expected}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/native_deconv_target/mixed_kernel_shared_prior_accuracy.json")
    args = parser.parse_args()
    for name, digest in FROZEN_FILES.items():
        if audit.frozen.sha256(ROOT / name) != digest:
            raise RuntimeError(f"A reused frozen script changed: {name}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    original_context, original_run, original_verify = audit.frozen.route_context, audit.frozen.run, audit.frozen.verify_hits
    shared_ops = None

    @contextmanager
    def selected_context(mode):
        context = combined_context(shared_ops) if mode == MODE else original_context(mode)
        with context as stats:
            yield stats

    def verify(stats, mode):
        return verify_combined(stats) if mode == MODE else original_verify(stats, mode)

    def annotated_run(report, save):
        nonlocal shared_ops
        from experiments.training_shared_s1.loader import load
        shared_ops, report["combined_shared_cuda_build"] = load(verbose=args.verbose_build)
        report["scope"] = __doc__
        report["experiment_adapter"] = dict(path=str(Path(__file__).resolve()),
            sha256=audit.frozen.sha256(__file__), frozen_dependencies=FROZEN_FILES,
            mixed_datanet_calls=5, shared_prior_calls=35, parameter_cache=False,
            precision_scope="Only the small kernel generator/projections and kernel FFT prepare remain FP64; activations/lambda/solve/IFFT and model parameters remain FP32")
        original_run(report, save)
        for name in (*FROZEN_FILES, "test/compare_mixed_kernel_shared_prior_model.py", "test/probe_nearest_training.py"):
            report["source_sha256"][name] = audit.frozen.sha256(ROOT / name)
        if any(audit.frozen.sha256(ROOT / name) != digest for name, digest in FROZEN_FILES.items()):
            raise RuntimeError("A reused frozen script changed during the accuracy run")

    argv = [str(Path(__file__).resolve()), "--mode", MODE, "--output", str(args.output)]
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, "argv", argv))
        stack.enter_context(patch.object(audit, "CANDIDATES", (MODE,)))
        stack.enter_context(patch.object(audit.frozen, "route_context", selected_context))
        stack.enter_context(patch.object(audit.frozen, "verify_hits", verify))
        stack.enter_context(patch.object(audit.frozen, "run", annotated_run))
        return audit.main()


if __name__ == "__main__":
    raise SystemExit(main())
