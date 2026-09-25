"""Execute the real USRNet forward on meta tensors to verify deployment shapes.

No extension import, CUDA initialization, compilation, or GPU allocation occurs.
Meta strides are not evidence about CUDA output layout; the GPU probe records it.
"""
import argparse
from collections import Counter
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

from inference_cases import ROOT, MODES, case_seed, declared_shapes, make_fixture, select_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output; earlier evidence is preserved")
    os.environ["CONVERSE2D_SKIP_BUILD"] = "1"
    sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
    import benchmark_production_layernorm as ablation
    identities = ablation.verified_sources(ROOT)
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    # util_converse probes optional imports on import. Block those names in this
    # process so this shape-only lane cannot even load a CUDA extension library.
    with patch.dict(sys.modules, {name: None for name in
         ("converse2d_ext", "torch_converse2d", "torch_converse2d.converse2d_ext")}):
        from models.converse_usrnet import ConverseUSRNet, ConvReverseDataNet
        from models.util_converse import Converse2D, LayerNorm
    with torch.device("meta"):
        model = ConverseUSRNet(backend="pytorch").eval()
    rows = []
    for case in select_cases(args.cases):
        expected = declared_shapes(case)
        for mode in MODES:
            calls = []
            handles = []
            def observe(module, inputs, output):
                record = dict(kind=type(module).__name__, input=list(inputs[0].shape), output=list(output.shape))
                if isinstance(module, Converse2D):
                    record.update(padding=module.padding, padding_mode=module.padding_mode,
                        padded_fft=[int(inputs[0].shape[-2]) + 2 * module.padding,
                                    int(inputs[0].shape[-1]) + 2 * module.padding], scale=module.scale)
                if isinstance(module, ConvReverseDataNet):
                    record.update(kernel=list(inputs[1].shape), scale=inputs[2], output_fft=list(output.shape[-2:]))
                calls.append(record)
            for module in model.modules():
                if isinstance(module, (LayerNorm, Converse2D, ConvReverseDataNet)):
                    handles.append(module.register_forward_hook(observe))
            try:
                x, kernel = make_fixture(torch, case, case_seed(case, 8011), "meta")
                with torch.no_grad() if mode == "no_grad" else torch.inference_mode():
                    output = model(x, kernel, case["scale"])
                assert list(output.shape) == expected["output"]
                counts = Counter(row["kind"] for row in calls)
                assert counts == dict(LayerNorm=70, Converse2D=35, ConvReverseDataNet=5), counts
                for row in calls:
                    if row["kind"] == "LayerNorm":
                        assert row["input"] == expected["layernorm"]
                    elif row["kind"] == "Converse2D":
                        assert row["input"] == expected["prior_input"]
                        assert row["padded_fft"] == expected["prior_padded_fft"] and row["padding"] == 2
                    else:
                        assert row["output_fft"] == expected["datanet_output_fft"]
                rows.append(dict(case=case, mode=mode, declared=expected, actual_calls=calls,
                                 input_stride=list(x.stride()), kernel_stride=list(kernel.stride()), passed=True))
            finally:
                for handle in handles:
                    handle.remove()
    assert not torch.cuda.is_initialized()
    files = [Path(__file__), ROOT / "tools/training_followup/inference_cases.py",
             ROOT / "models/converse_usrnet.py", ROOT / "models/util_converse.py", ROOT / "models/cuda_graph.py"]
    report = dict(kind="usrnet_deployment_cpu_shape_check", status="passed",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), cuda_initialized=False,
        extension_imports_blocked=True, torch=str(torch.__version__), cases=rows,
        production_layernorm_source_sha256=identities["production_normalized_source_sha256"],
        source_sha256={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        scope="Real production model forward executed with meta tensors and portable FP32 reference; verifies shapes/call counts only, not CUDA numerics, strides, performance or new fusion support.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(dict(status="passed", case_modes=len(rows), cuda_initialized=False, output=str(args.output.resolve()))))


if __name__ == "__main__":
    main()
