"""Bounded, checked-build full-model diagnostics; never a speed benchmark."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=('usrnet', 'srresnet'), default='usrnet')
    parser.add_argument('--mode', choices=('train', 'inference', 'graph'), default='train')
    parser.add_argument('--height', type=int, default=32)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--scale', type=int, default=3)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--tool', choices=('torch', 'nsys'), default='torch')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.tool == 'nsys' and not args.worker:
        nsys = Path(r'C:\Program Files\NVIDIA Corporation\Nsight Systems 2025.3.2\target-windows-x64\nsys.exe')
        argv = [str(Path(__file__).resolve()), *sys.argv[1:], '--worker']
        bootstrap = args.output / 'worker.py'
        # uv's Windows venv executable is a forwarding launcher; profile the
        # actual interpreter directly while preserving this environment's site.
        bootstrap.write_text('import runpy, site, sys\nsite.addsitedir(' +
                             repr(str(Path(sys.prefix)/'Lib/site-packages')) + ')\nsys.argv = ' +
                             repr(argv) + '\nrunpy.run_path(sys.argv[0], run_name="__main__")\n')
        command = [str(nsys), 'profile', '--trace=cuda,nvtx', '--sample=none', '--cpuctxsw=none',
                   '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--kill=false',
                   '--wait=all', '--show-output=true', '--export=sqlite',
                   '-o', str(args.output / 'capture'), getattr(sys, '_base_executable', sys.executable), '-u', str(bootstrap)]
        with (args.output / 'capture.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        report = dict(command=command, returncode=result.returncode,
                      report_exists=(args.output/'capture.nsys-rep').exists(), harness_sha256=sha(__file__))
        (args.output/'launcher.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report))
        raise SystemExit(result.returncode or (0 if report['report_exists'] else 1))

    os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
        raise RuntimeError('Unset backend overrides for checked CUDA diagnostics')
    sys.path.insert(0, str(args.root))
    import torch
    loader_spec = importlib.util.spec_from_file_location('roadmap_loader', args.root/'test/extension_loader.py')
    loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader)
    loader.load_extension()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260925)
    if args.model == 'usrnet':
        from models.converse_usrnet import ConverseUSRNet
        model = ConverseUSRNet(backend='cuda').cuda()
    else:
        from models.converse_srresnet import ConverseMSRResNet
        model = ConverseMSRResNet(in_channels=3, out_channels=3, num_features=64, num_blocks=16, upscale=4).cuda()
        if args.scale != 4:
            raise ValueError('SRResNet uses scale 4')
    checkpoint = args.root/'model_zoo'/('converse_' + args.model + '.pth')
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    x = torch.rand(args.batch, 3, args.height, args.width, device='cuda')
    kernel = torch.rand(args.batch, 1, 7, 7, device='cuda')
    kernel /= kernel.sum((-2, -1), keepdim=True)
    target = torch.rand(args.batch, 3, args.height*args.scale, args.width*args.scale, device='cuda')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, foreach=False, fused=False)
    runner = None
    if args.mode == 'graph':
        if args.model != 'usrnet':
            raise ValueError('Only USRNet has a production Graph runner')
        from models.cuda_graph import USRNetCUDAGraph
        runner = USRNetCUDAGraph(model.eval())

    def forward():
        return model(x, kernel, args.scale) if args.model == 'usrnet' else model(x)

    def step():
        if args.mode == 'train':
            model.train()
            optimizer.zero_grad(set_to_none=True)
            value = forward()
            loss = (value-target).square().mean()
            loss.backward()
            optimizer.step()
            return loss
        model.eval()
        with torch.no_grad():
            return runner(x, kernel, args.scale) if runner else forward()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    handles, stacks = [], {}
    if args.mode != 'graph' and args.tool == 'torch':
        for module in model.modules():
            if type(module).__name__ not in ('LayerNorm', 'Conv2d', 'GELU', 'Converse2D', 'ConvReverseDataNet'):
                continue
            def pre(m, _):
                context = torch.profiler.record_function('module/' + type(m).__name__)
                context.__enter__()
                stacks.setdefault(id(m), []).append(context)
            def post(m, _, output):
                stacks[id(m)].pop().__exit__(None, None, None)
            handles += [module.register_forward_pre_hook(pre), module.register_forward_hook(post, always_call=True)]
    if runner and args.tool == 'torch':
        for name in ('_validate', '_model_signature'):
            original = getattr(runner, name)
            def wrapped(*a, _name=name, _original=original, **kw):
                with torch.profiler.record_function('graph/' + _name):
                    return _original(*a, **kw)
            setattr(runner, name, wrapped)
    if args.tool == 'torch':
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profile:
            step()
            torch.cuda.synchronize()
        profile.export_chrome_trace(str(args.output/'trace.json.gz'))
        events = [dict(name=e.key, device_type=str(e.device_type), count=e.count, cpu_self_us=e.self_cpu_time_total,
                       cpu_total_us=e.cpu_time_total, device_self_us=e.self_device_time_total,
                       device_total_us=e.device_time_total) for e in profile.key_averages()]
        (args.output/'events.json').write_text(json.dumps(sorted(events, key=lambda e:e['device_self_us'], reverse=True), indent=2))
    else:
        torch.cuda.cudart().cudaProfilerStart()
        with torch.autograd.profiler.emit_nvtx(record_shapes=True):
            step()
            torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    for handle in handles:
        handle.remove()
    metadata = dict(model=args.model, mode=args.mode, shape=list(x.shape), scale=args.scale,
                    baseline=str(args.root), source_sha256=loader.production_source_hashes(),
                    build_manifest=json.loads((args.root/'.build/cuda/source_manifest.json').read_text()),
                    checkpoint_sha256=sha(checkpoint), harness_sha256=sha(__file__),
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                    warmup=3, steps=1, timing_claim=False, peak_allocated=torch.cuda.max_memory_allocated())
    (args.output/'metadata.json').write_text(json.dumps(metadata, indent=2))
    if runner:
        runner.clear()
    print(json.dumps({k:metadata[k] for k in ('model','mode','shape','scale','peak_allocated')}))


if __name__ == '__main__':
    main()
