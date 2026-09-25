"""Small, declared deployment coverage matrix; no CUDA work on import."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = (
    dict(name="b1_even_s3", batch=1, lr=(32, 32), scale=3, layout="contiguous", kernel_batch=1,
         reason="Existing HR96 anchor; actual prior FFT100x100."),
    dict(name="b4_even_s3", batch=4, lr=(32, 32), scale=3, layout="contiguous", kernel_batch=4,
         reason="Existing batch4 anchor; same HR96 and per-image kernels."),
    dict(name="b1_odd_s3", batch=1, lr=(31, 33), scale=3, layout="contiguous", kernel_batch=1,
         reason="Odd rectangular HR93x99; actual prior FFT97x103; previously only a recapture shape."),
    dict(name="b4_odd_s3", batch=4, lr=(31, 33), scale=3, layout="contiguous", kernel_batch=4,
         reason="Odd rectangular batch4 deployment with per-image kernels."),
    dict(name="b1_medium_s3", batch=1, lr=(64, 64), scale=3, layout="contiguous", kernel_batch=1,
         reason="HR192 and prior FFT196x196; B1 C64 features exceed the old affine2**21 threshold."),
    dict(name="b1_scale4_generic", batch=1, lr=(24, 25), scale=4, layout="contiguous", kernel_batch=1,
         reason="Supported SR scale4 uses existing generic DataNet solver branch; HR96x100, prior FFT100x104."),
    dict(name="b1_odd_strided", batch=1, lr=(31, 33), scale=3, layout="strided", kernel_batch=1,
         reason="Same logical odd shape with noncontiguous image and kernel; runner canonicalizes capture buffers."),
    dict(name="b4_channels_last_shared", batch=4, lr=(32, 32), scale=3, layout="channels_last", kernel_batch=1,
         reason="Channels-last caller image and a shared blur kernel; does not by itself imply internal LN fallback."),
)
MODES = ("no_grad", "inference_mode")


def case_seed(case, seed):
    """Subset selection/reordering must retain each declared case's input bytes."""
    return seed + next(index for index, item in enumerate(CASES) if item["name"] == case["name"])


def declared_shapes(case):
    h, w = case["lr"]
    hr = (h * case["scale"], w * case["scale"])
    return dict(input=[case["batch"], 3, h, w], output=[case["batch"], 3, *hr],
                layernorm=[case["batch"], 64, *hr],
                datanet_first_input=[case["batch"], 64, h, w],
                datanet_output_fft=list(hr), prior_input=[case["batch"], 128, *hr],
                prior_padded_fft=[hr[0] + 4, hr[1] + 4],
                layernorm_calls=70, prior_calls=35, datanet_calls=5,
                old_large_affine_threshold_crossed=case["batch"] * 64 * hr[0] * hr[1] >= 2**21)


def layout_tensor(torch, tensor, layout):
    if layout == "contiguous":
        return tensor.contiguous()
    if layout == "channels_last":
        return tensor.contiguous(memory_format=torch.channels_last)
    if layout == "strided":
        return tensor.transpose(-2, -1).contiguous().transpose(-2, -1)
    raise ValueError("Unknown caller layout: " + layout)


def make_fixture(torch, case, seed, device):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.rand(case["batch"], 3, *case["lr"], generator=generator)
    kernel = torch.rand(case["kernel_batch"], 1, 7, 7, generator=generator)
    kernel = kernel / kernel.sum((-2, -1), keepdim=True)
    x = layout_tensor(torch, x.to(device), case["layout"])
    kernel = layout_tensor(torch, kernel.to(device), "strided" if case["layout"] == "strided" else "contiguous")
    return x, kernel


def select_cases(names=None):
    if not names:
        return list(CASES)
    requested = names.split(",")
    known = {case["name"]: case for case in CASES}
    if len(set(requested)) != len(requested) or any(name not in known for name in requested):
        raise ValueError("Select unique declared case names")
    return [known[name] for name in requested]
