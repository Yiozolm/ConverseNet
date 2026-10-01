"""Level 1 boundary cost study, not an optimized low-precision solver.

Run after gate.py. All casts inside the adapter count toward complete-call
latency. Input storage is preexisting, as for the FP32 control. No AMP/TF32.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from tools.v4_mixed_precision.adapter import mixed_converse2d
from tools.v4_mixed_precision.gate import load_checked
from numerical_policy import error_metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def measure(call, *, cold=False, calls=10):
    if calls < 1:
        raise ValueError('A measurement requires at least one call')
    # Each cold round contains ten separately timed complete calls. Every
    # cache clear is outside its sample's timing interval; retain all samples.
    if cold:
        samples = []
        for _ in range(calls):
            torch.ops.converse2d.clear_cache()
            sample = measure(call, calls=1)
            sample['cache_clears'] = 1
            samples.append(sample)
        return dict(wall_ms=statistics.mean(row['wall_ms'] for row in samples),
                    cuda_ms=statistics.mean(row['cuda_ms'] for row in samples),
                    peak_increment_bytes=max(row['peak_increment_bytes'] for row in samples),
                    mean_peak_increment_bytes=statistics.mean(row['peak_increment_bytes'] for row in samples),
                    calls=calls, cache_clears=calls, samples=samples)
    torch.cuda.synchronize()
    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    tick = time.perf_counter()
    start.record()
    for _ in range(calls):
        output = call()
        # Avoid counting an overlapping live previous output in peak memory.
        del output
    end.record()
    end.synchronize()
    return dict(wall_ms=(time.perf_counter() - tick) * 1000 / calls,
                cuda_ms=start.elapsed_time(end) / calls,
                peak_increment_bytes=torch.cuda.max_memory_allocated() - before,
                calls=calls, cache_clears=0)


def inference_partitions(gate):
    """Use only independently admitted dtype/weight/output inference scopes."""
    groups = gate.get('summary', {}).get('by_dtype_ablation', [])
    lookup = {(row['dtype'], row['ablation']): row for row in groups}
    if len(groups) != len(lookup):
        raise ValueError('Duplicate numerical partition summary')
    admitted, skipped = [], []
    for dtype in ('fp16', 'bf16'):
        for ablation in ('activation_only', 'activation_and_weight'):
            summary = lookup.get((dtype, ablation))
            if summary is None:
                raise ValueError('Missing numerical dtype/ablation partition')
            rows = [row for row in gate['rows'] if not row['expected_range_probe']
                    and row['mode'] != 'training' and row['dtype'] == dtype and row['ablation'] == ablation]
            for level in ('level1a', 'level1b'):
                flag = 'inference_' + level + '_passed'
                actually_passed = bool(rows) and all(row.get(level, {}).get('passed', False) for row in rows)
                if summary.get(flag) is not actually_passed:
                    raise ValueError('Numerical partition summary disagrees with retained case results: ' + flag)
                entry = dict(dtype=dtype, ablation=ablation, level=level, inference_cases=len(rows),
                             output_dtype='torch.float32' if level == 'level1a' else
                                          ('torch.float16' if dtype == 'fp16' else 'torch.bfloat16'))
                if actually_passed and (level == 'level1a' or summary.get('inference_level1a_passed') is True):
                    admitted.append(entry)
                else:
                    entry.update(reason=flag + ' is false; no timing or profile will execute for this partition',
                                 failed_cases=[row['name'] for row in rows if not row.get(level, {}).get('passed', False)])
                    skipped.append(entry)
    return admitted, skipped


def attribution(call):
    """Nonoverlapping dispatcher self-device times; never benchmark timings."""
    try:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as prof:
            output = call()
            torch.cuda.synchronize()
        del output
        records = []
        for event in prof.key_averages():
            # CPU events own device kernels. GPU events would double count.
            if event.device_type != torch.autograd.DeviceType.CPU:
                continue
            if not event.key.startswith(('aten::', 'converse2d::')):
                continue
            duration = float(event.self_device_time_total)
            if duration > 0:
                records.append(dict(name=event.key, self_cuda_us=duration, count=event.count))
        groups = dict(cast_copy=0., fft=0., ifft=0., spectral_and_custom=0., other=0.)
        for row in records:
            name = row['name']
            group = ('fft' if name == 'aten::_fft_r2c' else
                     'ifft' if name == 'aten::_fft_c2r' else
                     'cast_copy' if name in ('aten::copy_', 'aten::_to_copy') else
                     'spectral_and_custom' if name == 'converse2d::forward' else 'other')
            groups[group] += row['self_cuda_us']
        return dict(status='available' if records else 'unavailable', groups_us=groups if records else None,
                    events=records, scope='Profile-only dispatcher self CUDA times; custom includes PSF preparation on uncached weights; copies include layout copies.')
    except Exception as exc:
        return dict(status='unavailable', reason=str(exc), groups_us=None)


def resident_bytes(values):
    unique = {id(value): value for value in values}
    return sum(t.numel() * t.element_size() for t in unique.values())


def study(rows, admitted, expected_manifest):
    torch.manual_seed(1901)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    loaded_manifest, _ = load_checked()
    if loaded_manifest != expected_manifest:
        raise RuntimeError('Checked production manifest changed before timing')
    allowed = {(row['dtype'], row['ablation'], row['level']) for row in admitted}
    # Includes the shape behind the earlier bandwidth diagnosis, but this
    # experiment measures HALF-spectrum inference, not that training kernel.
    for batch, channels, height, width, scale in (
            (4, 128, 100, 100, 1), (1, 64, 31, 37, 2),
            (1, 32, 17, 19, 3), (1, 16, 16, 16, 4)):
        x = torch.randn(batch, channels, height, width, device='cuda')
        prior = x if scale == 1 else torch.randn(batch, channels, height * scale, width * scale, device='cuda')
        weight = torch.softmax(torch.randn(1, channels, 9, device='cuda'), -1).reshape(1, channels, 3, 3)
        bias = torch.zeros(1, channels, 1, 1, device='cuda')
        baseline = lambda: torch.ops.converse2d.forward(x, prior, weight, bias, scale, 1e-5, 'v7')
        with torch.no_grad():
            baseline_output = baseline()
            for dtype in (torch.float16, torch.bfloat16):
                for quantize_weight in (False, True):
                    dtype_name = 'fp16' if dtype == torch.float16 else 'bf16'
                    ablation = 'activation_and_weight' if quantize_weight else 'activation_only'
                    if not any((dtype_name, ablation, level) in allowed for level in ('level1a', 'level1b')):
                        continue
                    xq = x.to(dtype)
                    pq = xq if prior is x else prior.to(dtype)
                    wq = weight.to(dtype) if quantize_weight else weight
                    for output_dtype in (torch.float32, dtype):
                        level = 'level1a' if output_dtype == torch.float32 else 'level1b'
                        if (dtype_name, ablation, level) not in allowed:
                            continue
                        candidate = lambda: mixed_converse2d(xq, pq, wq, bias, scale,
                                                             output_dtype=output_dtype)
                        result = candidate()
                        accuracy = error_metrics(result, baseline_output)
                        if not accuracy['finite']:
                            raise RuntimeError('Cannot time a nonfinite candidate')
                        qx = xq.float()
                        qp = qx if pq is xq else pq.float()
                        expected = torch.ops.converse2d.forward(qx, qp, wq.float(), bias, scale, 1e-5, 'v7').to(output_dtype)
                        if not torch.equal(result, expected):
                            raise RuntimeError('Boundary differs from identical-input production FP32 solver')
                        del qx, qp, expected
                        del result
                        for cold in (False, True):
                            repeats = []
                            for repeat in range(2):
                                # Each independent group starts clean. Warm
                                # groups then keep the same weight/bias objects
                                # and never clear between measured calls.
                                torch.ops.converse2d.clear_cache()
                                for call in (baseline, candidate):
                                    for _ in range(5):
                                        if cold:
                                            torch.ops.converse2d.clear_cache()
                                        call()
                                pairs = []
                                for round_id in range(9):
                                    order = ('baseline', 'candidate') if round_id % 2 == 0 else ('candidate', 'baseline')
                                    pair = dict(round=round_id, order=list(order))
                                    for label in order:
                                        pair[label] = measure(baseline if label == 'baseline' else candidate, cold=cold)
                                    pairs.append(pair)
                                repeats.append(dict(pairs=pairs, warmup_calls_per_route=5, timed_calls_per_route=90,
                                    group_initial_cache_clears=1, warmup_cache_clears=10 if cold else 0,
                                    timed_cache_clears_per_route=90 if cold else 0,
                                    wall_speedup=statistics.median(p['baseline']['wall_ms'] / p['candidate']['wall_ms'] for p in pairs),
                                    cuda_speedup=statistics.median(p['baseline']['cuda_ms'] / p['candidate']['cuda_ms'] for p in pairs)))
                            row = dict(shape=list(x.shape), scale=scale, shared_prior=prior is x,
                                dtype=str(dtype), quantize_weight=quantize_weight, output_dtype=str(output_dtype),
                                numerical_partition=dict(dtype=dtype_name, ablation=ablation, level=level, inference_passed=True),
                                cache='cold' if cold else 'warm_attempted', repeats=repeats,
                                accuracy_vs_fp32=accuracy,
                                resident_argument_bytes=dict(fp32=resident_bytes((x, prior, weight, bias)),
                                                             mixed=resident_bytes((xq, pq, wq, bias))),
                                output_bytes=x.numel() * scale * scale * torch.empty((), dtype=output_dtype).element_size(),
                                memory_scope='Logical input storage and allocator peak increments; NOT measured DRAM traffic. Both fixture copies reside during this experiment.')
                            if not cold:
                                def cast_inputs():
                                    a = xq.float()
                                    p = a if pq is xq else pq.float()
                                    return a, p, wq.float()
                                row['input_cast_only'] = [measure(cast_inputs) for _ in range(3)]
                                row['output_cast_only'] = [measure(lambda: baseline_output.to(output_dtype)) for _ in range(3)]
                                # Refresh the warm routes before untimed profile diagnostics.
                                baseline(); candidate()
                                row['profile_baseline'] = attribution(baseline)
                                row['profile_candidate'] = attribution(candidate)
                            rows.append(row)
                        print(f'PERF {list(x.shape)} s{scale} {dtype} weight={quantize_weight} output={output_dtype}', flush=True)
        torch.ops.converse2d.clear_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    gate = json.loads(args.gate.read_text(encoding='utf-8'))
    # Failed partitions remain rejected. Only independently admitted inference
    # partitions can execute below; timing does not confer production approval.
    if gate.get('status') != 'complete' or gate.get('kind') != 'mixed_precision_level1_gate':
        raise ValueError('Completed numerical report required before cost study')
    metadata = gate['metadata']
    if gate.get('device') != 'cuda' or gate.get('self_check'):
        raise ValueError('CPU self-check is not GPU evidence')
    for name, digest in metadata['source_sha256'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('Numerical source changed since gate: ' + name)
    manifest = json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())
    if manifest != metadata['checked_manifest']:
        raise ValueError('Checked extension changed since numerical gate')
    admitted, skipped = inference_partitions(gate)
    if not admitted:
        raise ValueError('No numerically admitted inference partition is available for timing')
    report = dict(kind='v4_level1_boundary_cost', status='running', gate_path=str(args.gate),
                  gate_sha256=sha(args.gate), numerical_report_passed=gate.get('passed'),
                  numerical_partition_summary=gate.get('summary'),
                  admitted_partitions=admitted, skipped_partitions=skipped,
                  scope='Inference boundary cost only; no speedup or production admission gate',
                  git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  sources={p.relative_to(ROOT).as_posix(): sha(p) for p in Path(__file__).parent.glob('*.py')},
                  torch=str(torch.__version__), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
                  autocast=False, tf32=False)
    try:
        report['rows'] = []
        study(report['rows'], admitted, manifest)
        report['checked_manifest'] = json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())
        if report['checked_manifest'] != manifest or sha(ROOT / '.build/cuda' / manifest['library']) != manifest['binary_sha256']:
            raise RuntimeError('Checked build changed during cost study')
        for name, digest in {**metadata['source_sha256'], **report['sources']}.items():
            if sha(ROOT / name) != digest:
                raise RuntimeError('Measured source changed during cost study: ' + name)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)


if __name__ == '__main__':
    main()
