import torch
import time
from util_converse import Converse2D


def benchmark(model, input_tensor, model_name, num_runs=100):
    """
    Measures the average forward pass time of a model.
    """
    print(f"Warming up {model_name} backend...")
    # Warm-up runs to stabilize performance measurement
    for _ in range(10):
        _ = model(input_tensor)
    torch.cuda.synchronize()

    print(f"Running benchmark for {model_name} backend ({num_runs} iterations)...")
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    # Wait for all kernels to complete
    torch.cuda.synchronize()
    end_time = time.time()

    avg_time = (end_time - start_time) / num_runs
    return avg_time

def run_comparison():
    if not torch.cuda.is_available():
        print("CUDA is not available. Performance comparison cannot be run.")
        return

    if Converse2D is None:
        print("Converse2D class not loaded. Aborting benchmark.")
        return

    params = {
        'in_channels': 64,
        'out_channels': 64, # Must be the same as in_channels for Converse2D
        'kernel_size': 3,
        'scale': 2,
        'padding': 2,
        'batch_size': 4,
        'height': 256,
        'width': 256,
        'device': torch.device("cuda")
    }

    print("\n--- Benchmark Configuration ---")
    for key, value in params.items():
        if key != 'device':
            print(f"{key.replace('_', ' ').capitalize()}: {value}")
    print(f"Device: {params['device']}")
    print("---------------------------------\n")


    # Create a dummy input tensor on the GPU
    input_tensor = torch.randn(
        params['batch_size'],
        params['in_channels'],
        params['height'],
        params['width']
    ).to(params['device'])

    try:
        # Initialize PyTorch backend model
        print("Initializing PyTorch backend model...")
        converse_torch = Converse2D(
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            kernel_size=params['kernel_size'],
            scale=params['scale'],
            padding=params['padding'],
            backend='torch'
        ).to(params['device'])
        print("PyTorch backend model initialized.")

        # Initialize CUDA backend model (this will trigger the JIT compilation)
        print("\nInitializing CUDA backend model (compilation may take a moment)...")
        converse_cuda = Converse2D(
            in_channels=params['in_channels'],
            out_channels=params['out_channels'],
            kernel_size=params['kernel_size'],
            scale=params['scale'],
            padding=params['padding'],
            backend='cuda'
        ).to(params['device'])
        print("CUDA backend model initialized and compiled successfully.")

    except Exception as e:
        print(f"\nAn error occurred during model initialization: {e}")
        print("Please ensure that a compatible CUDA toolkit is installed and configured correctly for PyTorch.")
        return

    # Run benchmarks
    torch_time = benchmark(converse_torch, input_tensor, "PyTorch")
    cuda_time = benchmark(converse_cuda, input_tensor, "CUDA")

    # --- Step 4: Report the results ---
    print("\n--- Performance Comparison Results ---")
    print(f"Input Tensor Shape: ({params['batch_size']}, {params['in_channels']}, {params['height']}, {params['width']})")
    print(f"PyTorch Backend Average Time: {torch_time * 1000:.4f} ms")
    print(f"CUDA Backend Average Time:    {cuda_time * 1000:.4f} ms")
    print("--------------------------------------")

    if cuda_time > 0:
        speedup = torch_time / cuda_time
        print(f"The CUDA implementation is approximately {speedup:.2f}x faster than the PyTorch implementation.")
    else:
        print("Could not calculate speedup due to zero execution time.")

if __name__ == "__main__":
    run_comparison()

"""
--- Device Details ---
GPU Architecture: RTX 2080ti
CUDA version: 12.8
Torch verison: 2.8.0

--- Benchmark Configuration ---
In channels: 64
Out channels: 64
Kernel size: 3
Scale: 2
Padding: 2
Batch size: 4
Height: 256
Width: 256
Device: cuda
---------------------------------

--- Performance Comparison Results ---
Input Tensor Shape: (4, 64, 256, 256)
PyTorch Backend Average Time: 131.7963 ms
CUDA Backend Average Time:    67.5533 ms
--------------------------------------
The CUDA implementation is approximately 1.95x faster than the PyTorch implementation.

"""