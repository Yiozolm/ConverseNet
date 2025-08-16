import torch
from util_converse import Converse2D


def benchmark(model, input_tensor, model_name, pass_type='forward', num_runs=100):
    """
    Measures the average forward or backward pass time of a model.
    Uses torch.cuda.Event for precise GPU timing.
    """
    if pass_type not in ['forward', 'backward']:
        raise ValueError("pass_type must be 'forward' or 'backward'")

    print(f"Warming up {model_name} backend for {pass_type} pass...")
    # Warm-up runs to stabilize performance and CUDA kernels
    for _ in range(10):
        output = model(input_tensor)
        if pass_type == 'backward':
            grad_output = torch.ones_like(output)
            output.backward(gradient=grad_output)

    torch.cuda.synchronize()

    print(f"Running benchmark for {model_name} backend {pass_type} pass ({num_runs} iterations)...")
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    total_time = 0.0

    if pass_type == 'forward':
        start_event.record()
        for _ in range(num_runs):
            _ = model(input_tensor)
        end_event.record()
    else:
        for _ in range(num_runs):
            output = model(input_tensor)
            grad_output = torch.ones_like(output)
            torch.cuda.synchronize() 

            start_event.record()
            output.backward(gradient=grad_output)
            end_event.record()
            
            torch.cuda.synchronize()
            total_time += start_event.elapsed_time(end_event)

            del output, grad_output

    if pass_type == 'forward':
        torch.cuda.synchronize()
        total_time = start_event.elapsed_time(end_event)

    avg_time = (total_time / 1000.0) / num_runs
    return avg_time

# =======================================================================
# The 'run_comparison' function and the rest of the file remain the same.
# No changes are needed below this line.
# =======================================================================
def run_comparison():
    if not torch.cuda.is_available():
        print("CUDA is not available. Performance comparison cannot be run.")
        return

    if Converse2D is None:
        print("Converse2D class not loaded. Aborting benchmark.")
        return

    params = {
        'in_channels': 64,
        'out_channels': 64,
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

    input_tensor = torch.randn(
        params['batch_size'],
        params['in_channels'],
        params['height'],
        params['width']
    ).to(params['device'])

    try:
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

    # --- Run Forward Pass benchmarks ---
    print("\n--- Benchmarking Forward Pass ---")
    cuda_forward_time = benchmark(converse_cuda, input_tensor, "CUDA", pass_type='forward')
    torch_forward_time = benchmark(converse_torch, input_tensor, "PyTorch", pass_type='forward')
    
    # --- Run Backward Pass benchmarks ---
    print("\n--- Benchmarking Backward Pass ---")
    # Enable gradient computation on the input tensor for the backward pass
    input_tensor.requires_grad_(True)
    cuda_backward_time = benchmark(converse_cuda, input_tensor, "CUDA", pass_type='backward')
    torch_backward_time = benchmark(converse_torch, input_tensor, "PyTorch", pass_type='backward')

    # --- Report the results ---
    print("\n\n--- Performance Comparison Results ---")
    print(f"Input Tensor Shape: ({params['batch_size']}, {params['in_channels']}, {params['height']}, {params['width']})")
    print("--------------------------------------")

    # Forward pass results
    print("\n--- Forward Pass ---")
    print(f"PyTorch Backend Average Time: {torch_forward_time * 1000:.4f} ms")
    print(f"CUDA Backend Average Time:    {cuda_forward_time * 1000:.4f} ms")
    if cuda_forward_time > 0:
        forward_speedup = torch_forward_time / cuda_forward_time
        print(f"CUDA implementation is {forward_speedup:.2f}x faster.")
    else:
        print("Could not calculate forward pass speedup.")

    # Backward pass results
    print("\n--- Backward Pass ---")
    print(f"PyTorch Backend Average Time: {torch_backward_time * 1000:.4f} ms")
    print(f"CUDA Backend Average Time:    {cuda_backward_time * 1000:.4f} ms")
    if cuda_backward_time > 0:
        backward_speedup = torch_backward_time / cuda_backward_time
        print(f"CUDA implementation is {backward_speedup:.2f}x faster.")
    else:
        print("Could not calculate backward pass speedup.")
    
    print("\n--------------------------------------")


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
--------------------------------------

--- Forward Pass ---
PyTorch Backend Average Time: 202.0048 ms
CUDA Backend Average Time:    76.2283 ms
CUDA implementation is 2.65x faster.

--- Backward Pass ---
PyTorch Backend Average Time: 105.8770 ms
CUDA Backend Average Time:    123.8077 ms
CUDA implementation is 0.86x faster.

--------------------------------------
The CUDA implementation is approximately 1.95x faster than the PyTorch implementation.

"""