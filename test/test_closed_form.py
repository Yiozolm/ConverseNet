import pathlib
import sys

import pytest
import torch
import torch.nn.functional as F


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Converse2D"))

from models.converse_usrnet import ConvReverseDataNet  # noqa: E402
from models.util_converse import Converse2D  # noqa: E402

try:
    import converse2d_ext  # noqa: F401, E402
except ImportError:
    converse2d_ext = None


def _legacy_closed_form(x, weight, bias, scale, padding, padding_mode, eps, ops):
    if padding > 0:
        x = F.pad(
            x,
            pad=[padding, padding, padding, padding],
            mode=padding_mode,
            value=0,
        )

    regularization = torch.sigmoid(bias - 9.0) + eps
    h, w = x.shape[-2:]
    observation_hr = ops.upsample(x, scale)
    x0 = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")

    transfer = ops.p2o(weight, (h * scale, w * scale))
    transfer_conj = torch.conj(transfer)
    transfer_power = torch.abs(transfer).square()

    rhs = transfer_conj * torch.fft.fftn(observation_hr, dim=(-2, -1))
    rhs = rhs + torch.fft.fftn(regularization * x0, dim=(-2, -1))
    normal_rhs = torch.mean(ops.splits(transfer * rhs, scale), dim=-1)
    normal = torch.mean(ops.splits(transfer_power, scale), dim=-1)
    solved = normal_rhs / (normal + regularization)
    spectrum = (rhs - transfer_conj * solved.repeat(1, 1, scale, scale))
    spectrum = spectrum / regularization
    out = torch.fft.ifftn(spectrum, dim=(-2, -1)).real

    if padding > 0:
        crop = padding * scale
        out = out[..., crop:-crop, crop:-crop]
    return out


@pytest.mark.parametrize("scale", [1, 2, 3, 4])
def test_converse2d_residual_form_matches_legacy_forward_and_gradients(scale):
    torch.manual_seed(10 + scale)
    layer = Converse2D(
        2,
        2,
        kernel_size=3,
        scale=scale,
        padding=1,
        padding_mode="circular",
        eps=1e-5,
        backend="pytorch",
    ).double()
    x = torch.randn(2, 2, 5, 7, dtype=torch.float64, requires_grad=True)
    grad_output = torch.randn(2, 2, 5 * scale, 7 * scale, dtype=torch.float64)

    actual = layer(x)
    actual_grads = torch.autograd.grad(
        actual,
        (x, layer.weight, layer.bias),
        grad_outputs=grad_output,
    )

    reference = _legacy_closed_form(
        x,
        layer.weight,
        layer.bias,
        scale,
        layer.padding,
        layer.padding_mode,
        layer.eps,
        layer,
    )
    reference_grads = torch.autograd.grad(
        reference,
        (x, layer.weight, layer.bias),
        grad_outputs=grad_output,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-10)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("scale", [1, 2, 3])
def test_data_net_residual_form_matches_legacy_kernel_and_lambda_gradients(scale):
    torch.manual_seed(20 + scale)
    layer = ConvReverseDataNet(eps=1e-3).double()
    x = torch.randn(1, 64, 4, 5, dtype=torch.float64, requires_grad=True)
    kernel = torch.randn(1, 64, 3, 3, dtype=torch.float64, requires_grad=True)
    grad_output = torch.randn(1, 64, 4 * scale, 5 * scale, dtype=torch.float64)

    actual = layer(x, kernel, scale, padding=1, padding_mode="circular")
    actual_grads = torch.autograd.grad(
        actual,
        (x, kernel, layer.alpha),
        grad_outputs=grad_output,
    )

    reference = _legacy_closed_form(
        x,
        kernel,
        layer.alpha,
        scale,
        1,
        "circular",
        layer.eps,
        layer,
    )
    reference_grads = torch.autograd.grad(
        reference,
        (x, kernel, layer.alpha),
        grad_outputs=grad_output,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-9, atol=1e-10)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-8, atol=1e-10)


def test_residual_form_reduces_small_lambda_cancellation_error():
    torch.manual_seed(0)
    layer = Converse2D(
        2,
        2,
        kernel_size=3,
        scale=3,
        padding=1,
        padding_mode="circular",
        eps=1e-8,
        backend="pytorch",
    )
    with torch.no_grad():
        layer.bias.fill_(-3.0)
    x = torch.randn(2, 2, 8, 9)

    with torch.no_grad():
        actual = layer(x)
        legacy = _legacy_closed_form(
            x,
            layer.weight,
            layer.bias,
            layer.scale,
            layer.padding,
            layer.padding_mode,
            layer.eps,
            layer,
        )

        reference_layer = Converse2D(
            2,
            2,
            kernel_size=3,
            scale=3,
            padding=1,
            padding_mode="circular",
            eps=1e-8,
            backend="pytorch",
        ).double()
        reference_layer.load_state_dict(layer.state_dict())
        reference = reference_layer(x.double()).float()

    actual_error = (actual - reference).abs().max()
    legacy_error = (legacy - reference).abs().max()
    assert actual_error < 1e-4
    assert actual_error * 100 < legacy_error


@pytest.mark.skipif(converse2d_ext is None, reason="Converse2D extension is not built")
@pytest.mark.parametrize("scale", [1, 2, 3, 4])
def test_extension_residual_form_matches_pytorch_forward_and_gradients(scale):
    torch.manual_seed(30 + scale)
    layer = Converse2D(
        2,
        2,
        kernel_size=3,
        scale=scale,
        padding=0,
        eps=1e-5,
        backend="pytorch",
    ).double()
    x = torch.randn(2, 2, 5, 7, dtype=torch.float64, requires_grad=True)
    x0 = x if scale == 1 else F.interpolate(x, scale_factor=scale, mode="nearest")
    grad_output = torch.randn(2, 2, 5 * scale, 7 * scale, dtype=torch.float64)

    actual = torch.ops.converse2d.forward(
        x, x0, layer.weight, layer.bias, scale, layer.eps
    )
    actual_grads = torch.autograd.grad(
        actual,
        (x, layer.weight, layer.bias),
        grad_outputs=grad_output,
    )

    reference = layer(x)
    reference_grads = torch.autograd.grad(
        reference,
        (x, layer.weight, layer.bias),
        grad_outputs=grad_output,
    )

    torch.testing.assert_close(actual, reference, rtol=1e-11, atol=1e-12)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(actual_grad, reference_grad, rtol=1e-10, atol=1e-12)
