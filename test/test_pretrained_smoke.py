"""Checkpoint/API integration smoke test; this is not a dataset quality benchmark."""
import argparse
import json
import sys

import torch
from extension_loader import ROOT, load_extension

sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--installed", action="store_true", help="Use the package built in Converse2D/ instead of JIT")
    args = parser.parse_args()
    if args.installed:
        sys.path.insert(0, str(ROOT / "Converse2D"))
        import torch_converse2d
    else:
        load_extension()
    from models.util_converse import Converse2D
    from models.converse_dncnn import ConverseDnCNN
    from models.converse_srresnet import ConverseMSRResNet
    # Isolate FFT rounding from cuDNN's lower-precision convolution arithmetic.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(17)
    results = []
    cases = [("converse_dncnn", ConverseDnCNN, 1), ("converse_srresnet", ConverseMSRResNet, 3)]
    with torch.inference_mode():
        for name, factory, channels in cases:
            # Construct outside inference mode so normal parameter versions can be cached.
            with torch.inference_mode(False):
                model = factory().cuda().eval()
                model.load_state_dict(torch.load(ROOT / "model_zoo" / f"{name}.pth", map_location="cuda", weights_only=True), strict=True)
            x = torch.rand(1,channels,24,32,device="cuda")
            for dtype in (torch.float32, torch.float64):
                model.to(dtype)
                for layer in model.modules():
                    if isinstance(layer,Converse2D): layer.backend = "pytorch"
                reference = model(x.to(dtype))
                for layer in model.modules():
                    if isinstance(layer,Converse2D): layer.backend = "cuda"
                output = model(x.to(dtype))
                tolerance = 1e-5 if dtype == torch.float32 else 1e-11
                torch.testing.assert_close(output,reference,atol=tolerance,rtol=tolerance)
                assert torch.isfinite(output).all()
                results.append({"model":name,"dtype":str(dtype),"output_shape":list(output.shape),
                                "max_abs_vs_pytorch":(output-reference).abs().max().item(),
                                "mean_abs_vs_pytorch":(output-reference).abs().mean().item()})
                torch.ops.converse2d.clear_cache()
    data = {"installed_package":args.installed,"tf32":False,
            "input":"seeded random [0,1], 24x32", "results":results}
    output=ROOT / "artifacts" / "pretrained_smoke.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,indent=2),encoding="utf-8")
    print(json.dumps(data,indent=2))


if __name__ == "__main__":
    main()
