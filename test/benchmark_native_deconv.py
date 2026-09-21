"""FP32 operator forward + dx/dw/db VJP versus native transposed convolution.

These operators are NOT mathematically equivalent. Converse2D solves a periodic
regularized inverse problem with an x-derived prior and lambda=sigmoid(bias-9)+eps.
Native conv_transpose2d applies a zero-padded transposed convolution and additive
bias. Same input/output shapes and gradient targets establish a timing boundary,
not a quality comparison or permission to replace the operator.

groups=C is the depthwise shape/parameter-count reference; groups=1 is a separate
dense-channel reference with C times as many kernel parameters. Batched dynamic
kernels use explicitly labelled per-sample native calls plus concatenation: the
native API has no batched-weight argument. Those loop/cat costs remain timed.
Fixtures reproduce observed layer shapes using seeded synthetic tensor values.
The s1_module fixture includes circular pad2 -> full Converse operator -> crop2
inside both forward and VJP. Native directly maps 96x96 to 96x96 with padding1;
the zero-versus-periodic boundary and mathematical differences remain explicit.
"""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "s1_module": dict(shape=(4, 128, 96, 96), scale=1, kernel=3, dynamic=False, eps=1e-5, outer_padding=2, seed_offset=3),
    "s1_shared": dict(shape=(4, 128, 100, 100), scale=1, kernel=3, dynamic=False, eps=1e-5, seed_offset=0),
    "s3_dynamic": dict(shape=(4, 64, 32, 32), scale=3, kernel=7, dynamic=True, eps=1e-3, seed_offset=1),
    "s3_large_batch": dict(shape=(32, 32, 64, 80), scale=3, kernel=3, dynamic=False, eps=1e-5, seed_offset=2),
}


def native_geometry(config):
    _, _, height, width = config["shape"]
    scale, kernel = config["scale"], config["kernel"]
    padding, output_padding = (kernel - 1) // 2, scale - 1
    output = [(size - 1) * scale - 2 * padding + kernel - 1 + output_padding + 1
              for size in (height, width)]
    if output != [height * scale, width * scale] or not 0 <= output_padding < scale:
        raise ValueError("Native transposed-convolution output geometry does not match")
    return dict(stride=scale, padding=padding, output_padding=output_padding, dilation=1,
                output_hw=output,
                formula="(input-1)*stride - 2*padding + dilation*(kernel-1) + output_padding + 1")


def cpu_data(config, seed):
    import torch
    batch, channels, height, width = config["shape"]
    scale, kernel = config["scale"], config["kernel"]
    generator = torch.Generator(device="cpu").manual_seed(seed + config["seed_offset"])
    kernel_batch = batch if config["dynamic"] else 1
    weight = torch.randn(kernel_batch, channels, kernel, kernel, generator=generator)
    weight = weight.flatten(2).softmax(-1).reshape_as(weight)
    dense_shape = ((batch, channels, channels, kernel, kernel) if config["dynamic"]
                   else (channels, channels, kernel, kernel))
    return dict(x=torch.randn(config["shape"], generator=generator), weight=weight,
                bias=torch.zeros(channels),
                upstream=torch.randn(batch, channels, height * scale, width * scale, generator=generator)
                         / math.sqrt(batch * channels * height * width * scale * scale),
                dense_weight=torch.randn(dense_shape, generator=generator) / math.sqrt(channels * kernel * kernel))


def clear_cuda(ops):
    import torch
    clear = getattr(ops, "clear_cache", None)
    if clear is not None:
        clear()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def make_fixture(kind, config, cpu):
    import torch
    import torch.nn.functional as F
    from benchmark_python_training import graph_has_spectral
    from train_usrnet_dataset import tensor_hash
    batch, channels, height, width = config["shape"]
    kernel, scale, dynamic = config["kernel"], config["scale"], config["dynamic"]
    geometry = native_geometry(config)
    x = cpu["x"].cuda().requires_grad_()
    if kind == "converse":
        weight_cpu, bias_cpu = cpu["weight"], cpu["bias"].reshape(1, channels, 1, 1)
    elif kind == "native_depthwise":
        shape = (batch, channels, 1, kernel, kernel) if dynamic else (channels, 1, kernel, kernel)
        weight_cpu, bias_cpu = cpu["weight"].reshape(shape), cpu["bias"]
    else:
        weight_cpu, bias_cpu = cpu["dense_weight"], cpu["bias"]
    weight = weight_cpu.cuda().requires_grad_()
    bias = bias_cpu.cuda().requires_grad_()
    upstream = cpu["upstream"].cuda()
    groups = channels if kind == "native_depthwise" else 1

    def forward():
        if kind == "converse":
            padding = config.get("outer_padding", 0)
            observation = F.pad(x, (padding,) * 4, mode="circular") if padding else x
            prior = observation if scale == 1 else F.interpolate(observation, scale_factor=scale, mode="nearest")
            output = torch.ops.converse2d.forward(observation, prior, weight, bias, scale, config["eps"], "v7")
            crop = padding * scale
            return output[..., crop:-crop, crop:-crop] if crop else output
        kwargs = {key: geometry[key] for key in ("stride", "padding", "output_padding", "dilation")}
        if dynamic:
            return torch.cat([F.conv_transpose2d(x[index:index + 1], weight[index], bias,
                                                groups=groups, **kwargs) for index in range(batch)], dim=0)
        return F.conv_transpose2d(x, weight, bias, groups=groups, **kwargs)

    def run():
        output = forward()
        gradients = torch.autograd.grad(output, (x, weight, bias), upstream, create_graph=False)
        return output.detach(), gradients

    # Route/shape/finite checks are outside measured repetitions.
    output = forward()
    assert tuple(output.shape) == (batch, channels, height * scale, width * scale)
    spectral = graph_has_spectral(output)
    if spectral != (kind == "converse"):
        raise RuntimeError(f"Unexpected SpectralSolve route for {kind}")
    gradients = torch.autograd.grad(output, (x, weight, bias), upstream)
    if not torch.stack([torch.isfinite(output).all(), *[torch.isfinite(value).all() for value in gradients]]).all().item():
        raise FloatingPointError(f"Nonfinite {kind} fixture")
    for actual, expected in zip(gradients, (x, weight, bias)):
        if actual.dtype != torch.float32 or actual.shape != expected.shape:
            raise RuntimeError(f"Incorrect VJP shape/dtype for {kind}")
    metadata = dict(kind=kind, input_shape=list(x.shape), output_shape=list(output.shape),
                    weight_shape=list(weight.shape), bias_shape=list(bias.shape),
                    gradient_targets=["x", "weight", "bias"], gradients_present=3,
                    gradient_elements=dict(x=x.numel(), weight=weight.numel(), bias=bias.numel()),
                    weight_sha256=tensor_hash(dict(weight=weight_cpu)), bias_sha256=tensor_hash(dict(bias=bias_cpu)),
                    route=dict(api="torch.ops.converse2d.forward(v7)" if kind == "converse" else "torch.nn.functional.conv_transpose2d",
                               spectral_solve=spectral, groups=None if kind == "converse" else groups,
                               native_calls_per_forward=0 if kind == "converse" else batch if dynamic else 1,
                               dynamic_native_implementation="explicit per-sample loop + cat" if dynamic and kind != "converse" else None),
                    native_geometry=geometry if kind != "converse" else None,
                    converse_wrapper=(dict(padding=config.get("outer_padding", 0), padding_mode="circular",
                                           crop=config.get("outer_padding", 0) * scale,
                                           internal_observation_hw=[height + 2 * config.get("outer_padding", 0),
                                                                    width + 2 * config.get("outer_padding", 0)])
                                      if kind == "converse" else None),
                    same_kernel_values_as_converse=kind == "native_depthwise",
                    finite_checked=True)
    del output, gradients, actual, expected
    return run, metadata


def paired_native(native_kind, config, cpu, args, ops):
    from benchmark_fp32_training import sample
    rows, fixture_metadata, orders = {"converse": [], native_kind: []}, {}, []
    for round_index in range(args.rounds):
        order = ["converse", native_kind] if round_index % 2 == 0 else [native_kind, "converse"]
        orders.append(order)
        for kind in order:
            clear_cuda(ops)
            run, metadata = make_fixture(kind, config, cpu)
            fixture_metadata[kind] = metadata
            for _ in range(args.warmup):
                run()
            row = sample(run, args.iters)
            rows[kind].append(row)
            del run
            clear_cuda(ops)
    ratios = {key: [rows["converse"][index][key] / rows[native_kind][index][key]
                    for index in range(args.rounds)] for key in ("wall_ms", "event_ms")}
    return dict(native_reference=native_kind, order=orders, fixtures=fixture_metadata, rounds=rows,
                medians={kind: {key: statistics.median(row[key] for row in values) for key in values[0]}
                         for kind, values in rows.items()},
                converse_over_this_native={key: dict(per_round=values, median=statistics.median(values),
                                                     min=min(values), max=max(values)) for key, values in ratios.items()},
                interpretation="Independent paired denominator for this native reference; ratio is a timing comparison of different mathematics, not a quality-preserving speedup")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/native_deconv/benchmark.json")
    args = parser.parse_args()
    if min(args.warmup, args.rounds, args.iters) < 1:
        parser.error("warmup, rounds and iters must be positive")
    if args.output.exists():
        parser.error("Choose a new output path; existing benchmark evidence is not overwritten")
    import torch
    import train_usrnet_dataset as worker
    args.variant = "current"
    ops, build = worker.load_backend(args)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    hashes = worker.source_hashes()
    for name in ("benchmark_native_deconv.py", "benchmark_fp32_training.py", "benchmark_python_training.py"):
        hashes["test/" + name] = worker.file_hash(ROOT / "test" / name)
    report = dict(status="running", settings=worker.json_safe(vars(args)), source_sha256=hashes, build=build,
                  environment=dict(torch=str(torch.__version__), cuda=torch.version.cuda,
                                   gpu=torch.cuda.get_device_name(), tf32=False, amp=False,
                                   cudnn_benchmark=False, cudnn_deterministic=True),
                  mathematical_equivalence=False, quality_comparison=False,
                  protocol=__doc__, scope="Device-resident FP32 forward plus complete x/weight/bias VJP; no loss or optimizer; Converse includes differentiable kernel preparation, FFT/solve/IFFT and prior generation; s1_module additionally includes pad/crop; warm eager, no CUDA Graph/profiler",
                  data_scope="Seeded synthetic values at observed layer shapes; no claim of captured real activation values",
                  memory_scope="One GPU fixture at a time; PyTorch allocator total allocated/reserved peaks including inputs, parameters and upstream; not driver/process total VRAM",
                  cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worker.write_json(args.output, report)
    for name, config in CASES.items():
        if args.case and name not in args.case:
            continue
        cpu = cpu_data(config, args.seed)
        row = dict(name=name, config=config,
                   boundary="full pad/operator/crop layer" if config.get("outer_padding") else "public full operator (including FFT), without external module padding/cropping",
                   native_output_geometry=native_geometry(config),
                   input_upstream_sha256=worker.tensor_hash({key: cpu[key] for key in ("x", "upstream")}),
                   complete_fixture_sha256=worker.tensor_hash(cpu), comparisons={})
        for native in ("native_depthwise", "native_dense"):
            row["comparisons"][native] = paired_native(native, config, cpu, args, ops)
            print(json.dumps(dict(case=name, reference=native,
                                  medians=row["comparisons"][native]["medians"],
                                  converse_over_native=row["comparisons"][native]["converse_over_this_native"])), flush=True)
        report["cases"].append(row)
        worker.write_json(args.output, report)
        del cpu
    report["status"] = "complete"
    worker.write_json(args.output, report)
    print("Saved", args.output, flush=True)


if __name__ == "__main__":
    main()
