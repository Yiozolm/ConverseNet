"""Evaluate a supplied real HR/LR dataset with the unchanged full USRNet.

Example (run from the checkout with its CUDA build environment configured):
    python test/evaluate_usrnet_quality.py --hr-dir DATA/HR --lr-dir DATA/LR \
        --kernel DATA/kernel.npy --scale 3 --backend both --metric-space y
    python test/evaluate_usrnet_quality.py --hr-dir DATA/HR --lr-dir DATA/LR \
        --kernel-dir DATA/kernels --scale 2 --backend current --crop-border 2

Protocol:
* Pair every HR/LR image by its exact relative path without the extension.
  A per-image kernel uses that same relative path with suffix .npy. Missing,
  extra, duplicate, unreadable or wrongly sized inputs are errors. No images
  are synthesized, resized, modcropped, degraded or overwritten.
* Decode uint8 images as RGB with utils_image; grayscale is expanded to RGB.
  Require HR height/width == LR height/width * scale. Kernels must be finite
  real 7x7 arrays (or 1x1x7x7), used as supplied after one FP32 cast; no implicit
  normalization. The checkpoint must strictly match 5 iterations, 7 blocks,
  64 features and a 7x7 KernelNet. No gates or parameters are reinitialized.
* current and pytorch use the same FP32 model, inputs and checkpoint. Both run
  eval + inference_mode with TF32 disabled. pytorch is the independent ATen
  full-FFT FP32 backend, not an FP64 accuracy reference or convergence test.
* PSNR/SSIM use the existing utils_image functions after prediction clipping
  to [0,1], rounding to uint8 RGB, then optional MATLAB-style uint8 Y conversion.
  Crop each border by --crop-border (default: scale); SSIM needs at least 11x11
  pixels after cropping. Report per-image values and their arithmetic means.
  Also report unrounded, unclipped FP32 output differences for paired runs.
* Alternate paired backend order by image. One model is resident. Before each
  backend, clear extension caches and unused allocator blocks; then warm up.
  Measure repeated warm inference with CUDA events and synchronized wall time,
  plus total PyTorch peak allocated/reserved bytes (model, inputs and caches
  included; driver/library allocations excluded). All preparation remains in
  forward; no data transfer, metric work, compilation or warmup is timed there.
  Read/decode/CPU preparation, input-to-device transfer, and output-to-CPU copy
  are reported separately. These components are not called end-to-end latency;
  file hashing, setup, warmup and metrics are excluded, OS file caches uncontrolled.
* Inputs are read-only. The only written artifact is JSON; no real-data claim
  can be made until actual HR/LR pairs and their matching kernels are supplied.
"""
import argparse
from contextlib import contextmanager
import datetime
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def index_files(directory, suffixes):
    directory = directory.resolve()
    if not directory.is_dir():
        raise ValueError(f"Data directory does not exist: {directory}")
    indexed = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffixes:
            key = path.relative_to(directory).with_suffix("").as_posix()
            if key in indexed:
                raise ValueError(f"Duplicate relative stem {key!r}: {indexed[key]} and {path}")
            indexed[key] = path
    if not indexed:
        raise ValueError(f"No matching real input files found in {directory}")
    return indexed


def pair_inputs(hr_dir, lr_dir, kernel=None, kernel_dir=None):
    hr = index_files(hr_dir, IMAGE_SUFFIXES)
    lr = index_files(lr_dir, IMAGE_SUFFIXES)
    if hr.keys() != lr.keys():
        raise ValueError(f"HR/LR relative stems differ; missing LR: {sorted(hr.keys()-lr.keys())[:10]}; "
                         f"missing HR: {sorted(lr.keys()-hr.keys())[:10]}")
    if kernel_dir is not None:
        kernels = index_files(kernel_dir, {".npy"})
        if kernels.keys() != hr.keys():
            raise ValueError(f"Kernel/image relative stems differ; missing kernels: "
                             f"{sorted(hr.keys()-kernels.keys())[:10]}; extra kernels: "
                             f"{sorted(kernels.keys()-hr.keys())[:10]}")
    else:
        if kernel is None or kernel.suffix.lower() != ".npy" or not kernel.is_file():
            raise ValueError(f"A readable shared .npy kernel is required: {kernel}")
        kernels = dict.fromkeys(hr, kernel.resolve())
    return [dict(key=key, hr=hr[key], lr=lr[key], kernel=kernels[key]) for key in sorted(hr)]


def checkpoint_layout(state):
    """Validate the expected full architecture before strict model loading."""
    if not isinstance(state, dict) or not all(isinstance(k, str) for k in state):
        raise ValueError("Checkpoint must contain a state_dict with string tensor keys")
    iterations = sorted({int(m[1]) for key in state if (m := re.match(r"convs\.(\d+)\.", key))})
    blocks = sorted({int(m[1]) for key in state if (m := re.match(r"p\.m_body\.(\d+)\.", key))})
    required = {"conv1.weight": (64, 3, 1, 1), "conv2.weight": (3, 64, 1, 1),
                "kernelnet.fc1.weight": (64, 49), "d.alpha": (1, 64, 1, 1)}
    if iterations != list(range(5)) or blocks != list(range(7)):
        raise ValueError(f"Expected full USRNet with 5 iterations/7 blocks; checkpoint indices "
                         f"are iterations={iterations}, blocks={blocks}; architecture is not auto-adjusted")
    for key, shape in required.items():
        if key not in state or tuple(getattr(state[key], "shape", ())) != shape:
            raise ValueError(f"Checkpoint tensor {key!r} must have shape {shape}")
    return dict(num_iterations=5, num_blocks=7, in_channels=64, kernel_size=7, strict=True)


def read_pair(paths, scale, border):
    import numpy as np
    import torch
    from utils import utils_image as util

    began = time.perf_counter()
    images = {}
    for label in ("hr", "lr"):
        try:
            image = util.imread_uint(str(paths[label]), n_channels=3)
        except Exception as exc:
            raise ValueError(f"Cannot decode {label.upper()} image {paths[label]}: {exc}") from exc
        if image is None or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Require uint8 RGB/grayscale image: {paths[label]}")
        images[label] = image
    hr, lr = images["hr"], images["lr"]
    if hr.shape[:2] != (lr.shape[0] * scale, lr.shape[1] * scale):
        raise ValueError(f"{paths['key']}: HR {hr.shape} does not equal LR {lr.shape} times scale={scale}")
    if min(hr.shape[:2]) - 2 * border < 11:
        raise ValueError(f"{paths['key']}: fewer than 11 pixels remain for SSIM after border={border}")
    array = np.load(paths["kernel"], allow_pickle=False)
    if array.shape == (1, 1, 7, 7):
        array = array[0, 0]
    if array.shape != (7, 7) or not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise ValueError(f"{paths['kernel']}: require a real 7x7 or 1x1x7x7 kernel")
    if not np.isfinite(array).all():
        raise ValueError(f"{paths['kernel']}: kernel contains nonfinite values")
    array = np.array(array, dtype=np.float32, order="C", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{paths['kernel']}: kernel values overflow FP32")
    x = util.uint2tensor4(lr)
    k = torch.from_numpy(array).reshape(1, 1, 7, 7)
    return hr, x, k, dict(read_decode_prepare_wall_ms=(time.perf_counter()-began)*1000,
                         lr_shape=list(lr.shape), hr_shape=list(hr.shape),
                         kernel_sum_fp32=float(array.sum(dtype=np.float64)))


def transfer_inputs(x, kernel, device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
    began = time.perf_counter()
    x, kernel = x.to(device), kernel.to(device)
    if device.type == "cuda":
        end.record()
        torch.cuda.synchronize(device)
    return x, kernel, dict(wall_ms=(time.perf_counter()-began)*1000,
                          cuda_event_ms=start.elapsed_time(end) if device.type == "cuda" else None,
                          source_pinned=False)


@contextmanager
def backend_scope(model, backend):
    # An explicit CLI backend must not be silently replaced by a shell override.
    previous = os.environ.get("CONVERSE2D_BACKEND")
    target = "cuda" if backend == "current" else "pytorch"
    layers = [(layer, layer.backend) for layer in model.modules() if hasattr(layer, "backend")]
    os.environ["CONVERSE2D_BACKEND"] = target
    for layer, _ in layers:
        layer.backend = target
    try:
        yield
    finally:
        for layer, original in layers:
            layer.backend = original
        if previous is None:
            os.environ.pop("CONVERSE2D_BACKEND", None)
        else:
            os.environ["CONVERSE2D_BACKEND"] = previous


def measure(model, x, kernel, scale, backend, warmup, repeats):
    import torch
    cuda = x.is_cuda
    clear_cache = getattr(torch.ops.converse2d, "clear_cache", None)
    if clear_cache is not None:
        clear_cache()
    gc.collect()
    if cuda:
        torch.cuda.synchronize(x.device)
        torch.cuda.empty_cache()
    with backend_scope(model, backend), torch.inference_mode():
        for _ in range(warmup):
            warm_output = model(x, kernel, scale)
            del warm_output
        if cuda:
            torch.cuda.synchronize(x.device)
            torch.cuda.reset_peak_memory_stats(x.device)
            initial_allocated = torch.cuda.memory_allocated(x.device)
            initial_reserved = torch.cuda.memory_reserved(x.device)
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
        began = time.perf_counter()
        for repeat in range(repeats):
            if repeat:
                del output
            output = model(x, kernel, scale)
        if cuda:
            end.record()
            torch.cuda.synchronize(x.device)
        wall_ms = (time.perf_counter()-began)*1000/repeats
        timing = dict(wall_ms=wall_ms, cuda_event_ms=start.elapsed_time(end)/repeats if cuda else None,
                      initial_allocated_bytes=initial_allocated if cuda else None,
                      initial_reserved_bytes=initial_reserved if cuda else None,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(x.device) if cuda else None,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(x.device) if cuda else None)
        if output.dtype != torch.float32 or not torch.isfinite(output).all().item():
            raise ValueError(f"{backend}: require finite FP32 output; got dtype={output.dtype}")
        began = time.perf_counter()
        output_cpu = output.detach().cpu()
        if cuda:
            torch.cuda.synchronize(x.device)
        timing["output_to_cpu_wall_ms"] = (time.perf_counter()-began)*1000
    return output_cpu, timing


def quality(output, hr, metric_space, border):
    import numpy as np
    from utils import utils_image as util
    # Work on a separate CPU array: keep the raw output for paired diagnostics.
    raw = output[0].permute(1, 2, 0).numpy()
    if raw.shape != hr.shape:
        raise ValueError(f"Output {raw.shape} does not match HR {hr.shape}; no resize is permitted")
    prediction = np.rint(np.clip(raw, 0, 1)*255).astype(np.uint8)
    target = hr
    if metric_space == "y":
        prediction = util.rgb2ycbcr(prediction, only_y=True)
        target = util.rgb2ycbcr(target, only_y=True)
    psnr = float(util.calculate_psnr(prediction, target, border=border))
    ssim = float(util.calculate_ssim(prediction, target, border=border))
    if math.isnan(psnr) or not math.isfinite(ssim):
        raise ValueError("PSNR/SSIM produced invalid values for the declared metric protocol")
    return dict(psnr_db=psnr, ssim=ssim)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity" if value < 0 else "NaN"
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hr-dir", required=True, type=Path)
    parser.add_argument("--lr-dir", required=True, type=Path)
    kernels = parser.add_mutually_exclusive_group(required=True)
    kernels.add_argument("--kernel", type=Path, help="Shared .npy kernel; exactly 7x7 or 1x1x7x7")
    kernels.add_argument("--kernel-dir", type=Path, help="Per-image .npy files at matching relative stems")
    parser.add_argument("--scale", type=int, choices=(2, 3, 4), required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"model_zoo/converse_usrnet.pth")
    parser.add_argument("--backend", choices=("current", "pytorch", "both"), default="both")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--metric-space", choices=("rgb", "y"), default="rgb")
    parser.add_argument("--crop-border", type=int, help="Pixels cropped from each HR/output edge; default=scale")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT/"artifacts/usrnet_quality/evaluation.json")
    args = parser.parse_args(argv)
    if args.crop_border is None:
        args.crop_border = args.scale
    if args.crop_border < 0 or args.warmup < 1 or args.repeats < 1:
        parser.error("crop-border must be nonnegative; warmup and repeats must be positive")
    if args.device == "cpu" and args.backend != "pytorch":
        parser.error("CPU evaluation requires --backend pytorch; current uses the CUDA extension")
    output = args.output.resolve()
    protected = [args.hr_dir.resolve(), args.lr_dir.resolve()]
    if args.kernel_dir is not None:
        protected.append(args.kernel_dir.resolve())
    protected_files = [args.checkpoint.resolve()]
    if args.kernel is not None:
        protected_files.append(args.kernel.resolve())
    if (output.suffix.lower() != ".json" or output in protected_files or
            any(output.is_relative_to(path) for path in protected)):
        parser.error("Output must be a .json file outside the read-only dataset/kernel directories")
    try:
        pairs = pair_inputs(args.hr_dir, args.lr_dir, args.kernel, args.kernel_dir)
        if not args.checkpoint.is_file():
            raise ValueError(f"Checkpoint does not exist: {args.checkpoint}")
    except ValueError as exc:
        parser.error(str(exc))

    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable; use --device cpu --backend pytorch for CPU reference evaluation")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(0)
    if args.backend in ("current", "both"):
        from extension_loader import load_extension
        from torch.utils.cpp_extension import CUDA_HOME
        if os.environ.get("CONVERSE2D_CPU_ONLY") == "1" or CUDA_HOME is None:
            parser.error("Current CUDA evaluation requires CUDA_HOME and CONVERSE2D_CPU_ONLY unset; "
                         "use the configured build environment, or --backend pytorch")
        load_extension()  # Validate/rebuild current checkout, never trust an installed old binary.
    from models.converse_usrnet import ConverseUSRNet
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    try:
        architecture = checkpoint_layout(state)
        model = ConverseUSRNet(num_iterations=5, num_blocks=7, in_channels=64, backend="pytorch")
        model.load_state_dict(state, strict=True)
    except (ValueError, RuntimeError) as exc:
        parser.error(f"Checkpoint is incompatible with unchanged full USRNet: {exc}")
    model = model.float().eval().requires_grad_(False).to(args.device)
    device = next(model.parameters()).device
    source_paths = [Path(__file__).resolve(), ROOT/"models/converse_usrnet.py", ROOT/"models/util_converse.py",
                    ROOT/"models/converse_core.py", ROOT/"utils/utils_image.py", ROOT/"test/extension_loader.py"]
    source_paths += sorted((ROOT/"Converse2D/torch_converse2d").rglob("*.cpp"))
    source_paths += sorted((ROOT/"Converse2D/torch_converse2d").rglob("*.cu"))
    source_paths += sorted((ROOT/"Converse2D/torch_converse2d").rglob("*.h"))
    source_paths += sorted((ROOT/"Converse2D/torch_converse2d").rglob("*.cuh"))
    source_paths += [ROOT/"Converse2D/build_config.py",ROOT/"Converse2D/setup.py"]
    report = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  settings={key: str(value.resolve()) if isinstance(value, Path) else value
                            for key, value in vars(args).items()},
                  architecture=architecture, checkpoint_sha256=sha256(args.checkpoint),
                  source_sha256={path.relative_to(ROOT).as_posix(): sha256(path) for path in source_paths},
                  environment=dict(torch=str(torch.__version__), numpy=np.__version__, cuda=torch.version.cuda,
                                   device=str(device), gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                                   tf32=False, cudnn_benchmark=False, seed=0),
                  protocol=__doc__, images=[], complete=False)
    backends = ["current", "pytorch"] if args.backend == "both" else [args.backend]
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")

    save()
    for index, paths in enumerate(pairs):
        hr, x_cpu, k_cpu, preparation = read_pair(paths, args.scale, args.crop_border)
        x, kernel, transfer = transfer_inputs(x_cpu, k_cpu, device)
        order = backends if index % 2 == 0 else list(reversed(backends))
        row = dict(key=paths["key"], paths={key: str(paths[key]) for key in ("hr", "lr", "kernel")},
                   input_sha256={key: sha256(paths[key]) for key in ("hr", "lr", "kernel")},
                   preparation=preparation, input_to_device=transfer, backend_order=order, backends={})
        raw_outputs = {}
        for backend in order:
            raw, timing = measure(model, x, kernel, args.scale, backend, args.warmup, args.repeats)
            row["backends"][backend] = dict(quality=quality(raw, hr, args.metric_space, args.crop_border),
                                            timing=timing)
            raw_outputs[backend] = raw
        if len(backends) == 2:
            actual, reference = raw_outputs["current"].double(), raw_outputs["pytorch"].double()
            delta = actual-reference
            row["paired_raw_output"] = dict(max_abs=delta.abs().max().item(),
                                            relative_l2=(delta.norm()/reference.norm().clamp_min(1e-30)).item())
            del actual, reference, delta
        report["images"].append(row)
        save()
        print(f"{index+1}/{len(pairs)} {paths['key']}: " + ", ".join(
            f"{backend} PSNR={row['backends'][backend]['quality']['psnr_db']:.4f} "
            f"SSIM={row['backends'][backend]['quality']['ssim']:.6f}" for backend in backends), flush=True)
        del x, kernel, raw, raw_outputs
    report["summary"] = {}
    for backend in backends:
        values = [row["backends"][backend] for row in report["images"]]
        summary = dict(images=len(values),
                       mean_psnr_db=statistics.mean(row["quality"]["psnr_db"] for row in values),
                       mean_ssim=statistics.mean(row["quality"]["ssim"] for row in values),
                       median_inference_wall_ms=statistics.median(row["timing"]["wall_ms"] for row in values))
        for field in ("cuda_event_ms", "peak_allocated_bytes", "peak_reserved_bytes"):
            measured = [row["timing"][field] for row in values if row["timing"][field] is not None]
            summary[("max_" if field.startswith("peak_") else "median_")+field] = (
                (max(measured) if field.startswith("peak_") else statistics.median(measured)) if measured else None)
        report["summary"][backend] = summary
    report["complete"] = True
    save()
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
