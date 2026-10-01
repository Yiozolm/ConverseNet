"""One warmed Level1A inference call between CUDA profiler start/stop markers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--dtype', choices=('fp32', 'fp16', 'bf16'), required=True)
    parser.add_argument('--metadata-dir', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path[:0] = [str(root), str(root / 'test')]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    from tools.v4_mixed_precision.adapter import mixed_converse2d
    manifest_path = root / '.build/cuda/source_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    original_arch = os.environ.get('TORCH_CUDA_ARCH_LIST')
    try:
        arch = manifest['inputs']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
        if arch:
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        else:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
        spec = importlib.util.spec_from_file_location('mixed_profile_checked_loader', root / 'test/extension_loader.py')
        loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loader)
        loader.load_extension()
    finally:
        if original_arch is None:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        else:
            os.environ['TORCH_CUDA_ARCH_LIST'] = original_arch
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Match the unprofiled cost study. Global deterministic mode also fills
    # every uninitialized allocation, adding kernels/traffic absent in perf.py.
    torch.use_deterministic_algorithms(False)
    sources = loader.production_source_hashes()
    dependencies = ('tools/v4_mixed_precision/profile_worker.py', 'tools/v4_mixed_precision/adapter.py',
                    'models/converse_core.py', 'test/extension_loader.py', 'Converse2D/build_config.py')
    source_hashes = {name: sha(root / name) for name in dependencies}
    generator = torch.Generator(device='cpu').manual_seed(41001)
    original = torch.randn(4, 128, 100, 100, generator=generator)
    weight_cpu = torch.rand(1, 128, 3, 3, generator=generator)
    bias_cpu = torch.zeros(1, 128, 1, 1)
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.dtype]
    x, weight, bias = original.to(device='cuda', dtype=dtype), weight_cpu.cuda(), bias_cpu.cuda()
    def tensor_sha(value):
        return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    fixtures = dict(original_x=tensor_sha(original), stored_x=tensor_sha(x), weight=tensor_sha(weight), bias=tensor_sha(bias))
    def call():
        if args.dtype == 'fp32':
            return torch.ops.converse2d.forward(x, x, weight, bias, 1, 1e-5, 'v7')
        return mixed_converse2d(x, x, weight, bias, 1, 1e-5, output_dtype=torch.float32, backend='cuda')
    torch.ops.converse2d.clear_cache()
    with torch.no_grad():
        for _ in range(5):
            warm_output = call()
            if not torch.isfinite(warm_output).all() or warm_output.dtype != torch.float32:
                raise RuntimeError('Warm fixture must produce finite FP32 output')
            # Match perf.py output lifetime, without retaining a previous output
            # while allocating the next call's temporaries.
            del warm_output
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        try:
            output = call()
            torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
    if not torch.isfinite(output).all() or output.dtype != torch.float32:
        raise RuntimeError('Profiled output must be finite FP32')
    binary = root / '.build/cuda' / manifest['library']
    if sources != loader.production_source_hashes() or sha(binary) != manifest['binary_sha256']:
        raise RuntimeError('Checked source or binary changed during worker execution')
    if source_hashes != {name: sha(root / name) for name in dependencies}:
        raise RuntimeError('Profile worker dependency changed')
    report = dict(kind='mixed_level1_ncu_worker', status='complete', pid=os.getpid(), dtype=args.dtype,
                  shape=[4,128,100,100], scale=1, shared_prior=True, warmup_calls=5, captured_calls=1,
                  weight_dtype='float32', bias_dtype='float32', output_dtype='float32', fixtures=fixtures,
                  logical_input_storage_bytes=x.numel()*x.element_size(), logical_output_bytes=output.numel()*output.element_size(),
                  output_sha256=tensor_sha(output), source_sha256=source_hashes, production_sources=sources,
                  checked_manifest=manifest, binary_sha256=sha(binary),
                  torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  deterministic_algorithms=False, cudnn_deterministic=True,
                  scope='Warm kernel cache; GPU resident input; complete call includes input upcast and FP32 solve. No output cast.',
                  profiler_only=True, numerical_admission=False)
    path = args.metadata_dir / f'worker-{os.getpid()}-{time.time_ns()}.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
