import os
import sys
import argparse
import torch
import pathlib
from torch.profiler import profile, record_function, ProfilerActivity

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.util_converse import Converse2D

def main():
    parser = argparse.ArgumentParser(description="Converse2D Profiler")
    parser.add_argument("--backend", type=str, default="pytorch", choices=["pytorch", "cuda"],
                        help="Which backend to profile: pytorch or cuda")
    parser.add_argument("--H", type=int, default=256, help="Input height")
    parser.add_argument("--W", type=int, default=256, help="Input width")
    parser.add_argument("--B", type=int, default=2, help="Batch size")
    parser.add_argument("--C", type=int, default=8, help="Channels")
    parser.add_argument("--scale", type=int, default=2, help="Upsampling scale")
    args = parser.parse_args()

    os.environ["CONVERSE2D_BACKEND"] = args.backend.lower()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Running Converse2D with backend: {args.backend.upper()} on {device}")

    x = torch.randn(args.B, args.C, args.H, args.W, device=device)
    model = Converse2D(args.C, args.C, kernel_size=5, scale=args.scale, padding=4, backend=args.backend).to(device)

    # warmup
    for _ in range(10):
        _ = model(x)

    trace_file = f"profile_converse2d_{args.backend.lower()}.json"

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        with record_function(f"Converse2D::forward({args.backend.upper()})"):
            y = model(x)

    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=25))
    prof.export_chrome_trace(trace_file)
    print(f"[INFO] Saved trace to: {trace_file}")

if __name__ == "__main__":
    main()
