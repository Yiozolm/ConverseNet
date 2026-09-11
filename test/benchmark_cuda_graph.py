"""Pretrained USRNet eager versus bounded graph runner, including GPU I/O."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import torch
from extension_loader import ROOT, load_extension
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--output', default='artifacts/benchmark_cuda_graph.json')
    args = parser.parse_args()
    if min(args.iters, args.rounds) < 1:
        parser.error('iters and rounds must be positive')
    load_extension()
    from models.converse_usrnet import ConverseUSRNet
    from models.cuda_graph import USRNetCUDAGraph
    torch.manual_seed(615)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    model = ConverseUSRNet(backend='cuda').cuda().eval()
    model.load_state_dict(torch.load(ROOT/'model_zoo/converse_usrnet.pth',
                                    map_location='cuda', weights_only=True))
    results = []
    for h, w in ((32, 40), (64, 80)):
        runner = USRNetCUDAGraph(model)
        x = torch.rand(1, 3, h, w, device='cuda')
        coords = torch.arange(7, device='cuda') - 3
        k = torch.exp(-(coords[:, None].square() + coords[None, :].square()) / (2 * 1.2**2))
        k = (k/k.sum())[None, None]
        with torch.inference_mode():
            for _ in range(10): model(x, k, 2)
            torch.cuda.synchronize()
            before = {'allocated_mib': torch.cuda.memory_allocated()/2**20,
                      'reserved_mib': torch.cuda.memory_reserved()/2**20}
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            runner(x, k, 2)
            torch.cuda.synchronize()
            capture_ms = (time.perf_counter() - start)*1000
            after = {'allocated_mib': torch.cuda.memory_allocated()/2**20,
                     'reserved_mib': torch.cuda.memory_reserved()/2**20,
                     'peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20}
            errors = []
            for shift in (0., .025):
                changed_x = (x+shift).clamp(0, 1)
                changed_k = k+shift*.01
                changed_k = changed_k/changed_k.sum((-2, -1), keepdim=True)
                expected = model(changed_x, changed_k, 2)
                actual = runner(changed_x, changed_k, 2)
                torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
                errors.append((actual-expected).abs().max().item())
            functions = {'eager': lambda: model(x, k, 2),
                         'graph': lambda: runner(x, k, 2)}
            samples = {name: [] for name in functions}
            for repeat in range(args.rounds):
                order = list(functions)
                if repeat % 2: order.reverse()
                for name in order:
                    fn = functions[name]
                    for _ in range(5): fn()
                    torch.cuda.synchronize()
                    a = torch.cuda.Event(enable_timing=True)
                    b = torch.cuda.Event(enable_timing=True)
                    t0 = time.perf_counter()
                    a.record()
                    for _ in range(args.iters): fn()
                    b.record()
                    b.synchronize()
                    samples[name].append({'gpu_ms': a.elapsed_time(b)/args.iters,
                                          'wall_ms': (time.perf_counter()-t0)*1000/args.iters})
        timings = {name: {metric: statistics.median(s[metric] for s in values)
                         for metric in ('gpu_ms', 'wall_ms')}
                   for name, values in samples.items()}
        row = {'shape': list(x.shape), 'scale': 2, 'max_abs_errors': errors,
               'first_call_ms': capture_ms, 'memory_before': before, 'memory_after': after,
               'timings': timings, 'rounds': samples,
               'speedup': timings['eager']['wall_ms']/timings['graph']['wall_ms']}
        results.append(row)
        print(json.dumps(row), flush=True)
        runner.clear()
        torch.ops.converse2d.clear_cache()
    sources = ('models/cuda_graph.py', 'models/converse_usrnet.py',
               'Converse2D/torch_converse2d/converse2d.cpp',
               'Converse2D/torch_converse2d/converse2d_kernels.cu')
    result = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
              'dtype': 'float32', 'tf32': False, 'iters': args.iters,
              'measurement': 'Full runner: signature checks, input copies, replay, output clone; no profiler.',
              'memory_note': 'Snapshots include eager caches; reserved deltas are not total graph pool size.',
              'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
              'results': results}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
