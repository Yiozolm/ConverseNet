"""Bounded CUDA Graph inference for ConverseUSRNet; checkpoint format is unchanged."""
from collections import OrderedDict
from dataclasses import dataclass
import os
import threading

import torch

from models.converse_usrnet import ConverseUSRNet
from models.util_converse import _try_import_converse2d_ext


# Serialize our captures and submissions across runners. Other CUDA work and
# model edits must not run concurrently with capture in the same process.
_GRAPH_LOCK = threading.RLock()


@dataclass
class _Entry:
    graph: torch.cuda.CUDAGraph
    x: torch.Tensor
    kernel: torch.Tensor
    output: torch.Tensor
    done: torch.cuda.Event
    # Keep the captured addresses alive even if a caller replaces parameters.
    model_tensors: tuple

    def close(self):
        self.done.synchronize()
        self.graph.reset()


class USRNetCUDAGraph:
    """Opt-in inference callable with an LRU of fixed-shape CUDA graphs.

    Set enabled=False to call the model directly, preserving autograd, autocast
    and CPU execution. Switching off also clears any captured graphs. Explicit
    construction defaults to enabled=True for compatibility; model(...) itself
    never enables CUDA Graphs.

    Use an eval-mode ConverseUSRNet with FP32/FP64 CUDA inputs inside no_grad
    or inference_mode. Each call copies inputs and returns an independent output.
    A miss warms up and captures synchronously; only hits have replay latency.
    Parameter versions, addresses, module configuration and backend flags are
    checked on every call. Normal load_state_dict/optimizer edits invalidate the
    cache automatically. After edits through .data or custom Python behavior,
    call clear() explicitly. Do not mutate the model concurrently with calls.

    Autocast, hooks, training and nesting in another capture are not supported.
    Graph count is bounded, but each shape has its own private memory pool;
    clear() waits for pending work and releases all graph-owned references.
    """

    def __init__(self, model, *, enabled=True, max_graphs=1, warmup=3):
        if type(model) is not ConverseUSRNet:
            raise TypeError("model must be a ConverseUSRNet")
        if type(max_graphs) is not int or max_graphs < 1:
            raise ValueError("max_graphs must be a positive integer")
        if type(warmup) is not int or warmup < 1:
            raise ValueError("warmup must be a positive integer")
        self.model = model
        self.max_graphs = max_graphs
        self.warmup = warmup
        self._entries = OrderedDict()
        self._signature = None
        self.captures = 0
        self.enabled = enabled

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        if type(value) is not bool:
            raise TypeError("enabled must be a bool")
        with _GRAPH_LOCK:
            if not value:
                self.clear()
            self._enabled = value

    @property
    def cached_graphs(self):
        return len(self._entries)

    def clear(self):
        """Wait for pending replays and release graph pools (not eager caches)."""
        with _GRAPH_LOCK:
            for entry in self._entries.values():
                entry.close()
            self._entries.clear()
            self._signature = None

    def __del__(self):
        # Dropping a runner must not free static buffers while a replay from a
        # different caller stream is still using them. Explicit clear() is
        # preferred, especially before interpreter/CUDA runtime shutdown.
        try:
            self.clear()
        except Exception:
            pass  # Runtime globals can already be gone during shutdown.

    def _model_signature(self, device):
        modules = tuple(self.model.modules())
        if any(m.training for m in modules):
            self.clear()
            raise RuntimeError("CUDA Graph inference requires model.eval()")
        if any(m._forward_hooks or m._forward_pre_hooks for m in modules):
            self.clear()
            raise RuntimeError("remove forward hooks before CUDA Graph inference")
        tensors = tuple(self.model.parameters()) + tuple(self.model.buffers())
        if any(t.device != device for t in tensors):
            self.clear()
            raise ValueError("model and inputs must be on the same CUDA device")
        if any(t.is_inference() for t in tensors):
            raise ValueError("create model tensors outside inference_mode for version tracking")
        # Public scalar/tuple attributes contain the built-in modules' eps,
        # padding, iterations, backend, normalization shape, etc.
        config = tuple((id(m), type(m), tuple(
            (name, value) for name, value in vars(m).items()
            if not name.startswith('_') and isinstance(value, (str, int, float, bool, tuple, type(None)))
        )) for m in modules)
        tensor_state = tuple((id(t), t.data_ptr(), t._version, tuple(t.shape),
                              t.stride(), t.dtype, t.device) for t in tensors)
        flags = (os.environ.get("CONVERSE2D_BACKEND", ""),
                 torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark,
                 torch.backends.cudnn.deterministic, torch.backends.cudnn.allow_tf32,
                 torch.backends.cuda.matmul.allow_tf32,
                 torch.get_float32_matmul_precision(),
                 torch.are_deterministic_algorithms_enabled())
        return (config, tensor_state, flags), tensors

    def _validate(self, x, kernel, scale):
        if torch.is_grad_enabled():
            raise RuntimeError("use torch.no_grad() or torch.inference_mode() for graph inference")
        if torch.is_autocast_enabled('cuda'):
            raise RuntimeError("CUDA Graph inference currently requires autocast disabled")
        if type(scale) is not int or scale < 1:
            raise ValueError("scale must be a positive integer")
        if not x.is_cuda or kernel.device != x.device:
            raise ValueError("x and kernel must be on the same CUDA device")
        if x.dtype not in (torch.float32, torch.float64) or kernel.dtype != x.dtype:
            raise ValueError("x and kernel must have the same float32 or float64 dtype")
        if x.layout != torch.strided or kernel.layout != torch.strided:
            raise ValueError("x and kernel must be strided tensors")
        if x.ndim != 4 or x.shape[1] != 3 or any(d == 0 for d in x.shape):
            raise ValueError("x must have nonempty shape (B,3,H,W)")
        ks = self.model.kernelnet.kernel_size
        if kernel.ndim != 4 or kernel.shape[0] not in (1, x.shape[0]) or kernel.shape[1:] != (1, ks, ks):
            raise ValueError("kernel must have shape (1|B,1,7,7)")
        if min(x.shape[-2:]) * scale < ks:
            raise ValueError("kernel must fit the output spatial dimensions")
        if self.model.conv1.weight.dtype != x.dtype:
            raise ValueError("model and inputs must have the same dtype")

    def _capture(self, x, kernel, scale, tensors):
        _try_import_converse2d_ext()
        use_scope = hasattr(torch.ops.converse2d, 'forward')
        if use_scope:
            # An installed extension may predate graph-owned spectrum support.
            if (not hasattr(torch.ops.converse2d, 'supports_cuda_graphs') or
                    not torch.ops.converse2d.supports_cuda_graphs() or
                    not hasattr(torch.ops.converse2d, 'begin_graph_cache') or
                    not hasattr(torch.ops.converse2d, 'end_graph_cache')):
                raise RuntimeError("rebuild Converse2D: loaded extension lacks CUDA Graph cache safety")
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        graph = torch.cuda.CUDAGraph()
        owned_spectra = ()
        try:
            if use_scope:
                torch.ops.converse2d.begin_graph_cache()
            try:
                with torch.cuda.stream(stream), torch.inference_mode():
                    static_x = x.clone(memory_format=torch.contiguous_format)
                    static_kernel = kernel.clone(memory_format=torch.contiguous_format)
                    for _ in range(self.warmup):
                        self.model(static_x, static_kernel, scale)
                stream.synchronize()
                with torch.inference_mode(), torch.cuda.graph(graph, stream=stream):
                    output = self.model(static_x, static_kernel, scale)
            finally:
                if use_scope:
                    owned_spectra = tuple(torch.ops.converse2d.end_graph_cache())
        except Exception:
            stream.synchronize()
            graph.reset()
            raise
        done = torch.cuda.Event()
        done.record(stream)
        self.captures += 1
        # Retain storage snapshots as well as spectra: Module.to() and .data
        # replacement may change the storage of the original Parameter object.
        owned_parameters = tuple(t.detach() for t in tensors)
        return _Entry(graph, static_x, static_kernel, output, done, owned_parameters + owned_spectra)

    def __call__(self, x, kernel, scale):
        with _GRAPH_LOCK:
            if not self._enabled:
                return self.model(x, kernel, scale)
            self._validate(x, kernel, scale)
            with torch.cuda.device(x.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("USRNetCUDAGraph cannot run inside another capture")
                signature, tensors = self._model_signature(x.device)
                if signature != self._signature:
                    self.clear()
                    self._signature = signature
                key = (tuple(x.shape), tuple(kernel.shape), x.dtype, x.device, scale)
                entry = self._entries.get(key)
                if entry is None:
                    # Evict before capture to bound the number of live pools.
                    if len(self._entries) >= self.max_graphs:
                        _, old = self._entries.popitem(last=False)
                        old.close()
                        del old
                    entry = self._capture(x, kernel, scale, tensors)
                    self._entries[key] = entry
                self._entries.move_to_end(key)
                stream = torch.cuda.current_stream(x.device)
                stream.wait_event(entry.done)
                # Serialize shared buffers across caller streams. The clone is
                # allocated on the caller stream, so later calls cannot overwrite it.
                with torch.inference_mode():
                    entry.x.copy_(x)
                    entry.kernel.copy_(kernel)
                    entry.graph.replay()
                    output = entry.output.clone()
                entry.done.record(stream)
                return output
