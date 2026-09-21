"""Automatic VJP adapter for a compiled forward with an eager high-order path.

The factory retains callables/code only, never input tensors or parameter
values. Each invocation owns its inner graph through save_for_backward.
Ordinary backward differentiates the compiled ATen/AOTAutograd graph; higher
derivatives reconstruct eager ATen from the original differentiable inputs.
No spatial gradient formula is implemented here.

This wrapper adds graph ownership and nested-autograd overhead; a bare
compiled benchmark does not establish this adapter's performance. Initial
compiled execution on view proxies and each gradient mask belongs in setup.

Set torch._functorch.config.donated_buffer=False BEFORE creating or first
executing the compiled callable, and keep it false for the entire experiment.
Use a fresh callable/cache configuration; changing this flag cannot repair a
previously compiled graph with donated saved buffers. The guard below checks
the live setting, not the historical settings of an arbitrary caller's code.
"""
import torch
import torch._functorch.config as functorch_config


def _require_retained_buffers():
    if functorch_config.donated_buffer is not False:
        raise RuntimeError("wrap_compiled requires torch._functorch.config.donated_buffer=False "
                           "before compilation and throughout execution; create a fresh compiled callable")


def wrap_compiled(eager_callable, compiled_callable):
    """Return callable(x, weight, bias) with three independently owned VJPs.

    Both callables must implement the same differentiable Tensor-valued
    function. Their captured configuration may include geometry/constants,
    but callers must not capture mutable input/parameter tensors as a cache.
    This adapter does not catch compilation failures or silently substitute
    eager execution for ordinary first-order training.
    """
    _require_retained_buffers()
    eager, compiled = eager_callable, compiled_callable

    class AutomaticVJP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, weight, bias):
            ctx.set_materialize_grads(False)
            with torch.enable_grad():
                # Different formal arguments need different gradient targets
                # even when callers pass the same Tensor in multiple slots.
                proxies = tuple(value.view_as(value) for value in (x, weight, bias))
                inner = compiled(*proxies)
            ctx.save_for_backward(x, weight, bias, *proxies, inner)
            # Outer autograd owns this new output; do not return/rewire inner.
            return inner.detach()

        @staticmethod
        def backward(ctx, grad_output):
            _require_retained_buffers()
            if grad_output is None:
                return None, None, None
            saved = ctx.saved_tensors  # Also enforces original input versions.
            needs = ctx.needs_input_grad
            higher = torch.is_grad_enabled()
            if higher:
                with torch.enable_grad():
                    proxies = tuple(value.view_as(value) for value in saved[:3])
                    output = eager(*proxies)
            else:
                proxies, output = saved[3:6], saved[6]
            requested = tuple(value for value, needed in zip(proxies, needs) if needed)
            if not requested or not output.requires_grad:
                return None, None, None
            values = iter(torch.autograd.grad(output, requested, grad_output,
                create_graph=higher, retain_graph=True, allow_unused=True))
            return tuple(next(values) if needed else None for needed in needs)

    def route(x, weight, bias):
        _require_retained_buffers()
        if not torch.is_grad_enabled() or not any(value.requires_grad for value in (x, weight, bias)):
            return compiled(x, weight, bias)
        return AutomaticVJP.apply(x, weight, bias)

    return route
