"""Pretrained model quantization sensitivity; no production precision change.

This simulates FP16/BF16 boundary storage by quantize->FP32 round trips. All
model computation, master parameters, bias/alpha and solver arithmetic stay
FP32. It is not AMP, a low-precision kernel, a memory benchmark, or production
admission. Hooks and temporary weight Parameters are local and always restored.

Example (root-owned GPU window, existing checked build):
  python tools/v4_mixed_precision/model_study.py --output NEW_REPORT.json --samples 6
"""
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Resolve the repository namespace before any external packages.
sys.path[:] = [str(ROOT), str(ROOT / 'test')] + [p for p in sys.path
    if Path(p or os.getcwd()).resolve() not in (HERE, ROOT, ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch

DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16}
VARIANTS = {
    'activation_only': (True, False, False),
    'activation_weight': (True, True, False),
    'output_only': (False, False, True),
    'activation_output': (True, False, True),
    'activation_weight_output': (True, True, True),
}
CHECKPOINTS = {'dncnn': 'converse_dncnn.pth', 'srresnet': 'converse_srresnet.pth',
               'usrnet': 'converse_usrnet.pth'}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def tensor_record(value):
    dense = value.detach().resolve_conj().resolve_neg().cpu().contiguous()
    return dict(shape=list(value.shape), stride=list(value.stride()), dtype=str(value.dtype),
                sha256=hashlib.sha256(dense.numpy().tobytes()).hexdigest())


def state_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        item = tensor_record(value)
        digest.update(json.dumps([name, item], sort_keys=True).encode())
    return digest.hexdigest()


def error_metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError('Error comparison requires identical shapes')
    a, b = actual.detach().double(), reference.detach().double()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(finite=False, max_abs=None, rel_l2=None)
    delta = a - b
    maximum = delta.abs().max().item()
    ratio = (delta.norm() / b.norm().clamp_min(1e-300)).item()
    return dict(finite=True, max_abs=maximum,
                rel_l2=ratio if math.isfinite(ratio) else None,
                rel_l2_defined=math.isfinite(ratio))


def range_record(value):
    value = value.detach()
    finite_count = int(torch.isfinite(value).sum().item())
    row = dict(shape=list(value.shape), dtype=str(value.dtype), finite_count=finite_count,
               numel=value.numel(), nonfinite_count=value.numel() - finite_count)
    if finite_count == value.numel() and value.numel():
        low, high = torch.aminmax(value)
        row.update(min=low.item(), max=high.item(), max_abs=value.abs().max().item(),
                   mean_abs=value.abs().double().mean().item(), zero_count=int((value == 0).sum().item()))
    else:
        row.update(min=None, max=None, max_abs=None, mean_abs=None, zero_count=None)
    return row


def quantize(value, dtype):
    if value.dtype != torch.float32:
        raise ValueError('Sensitivity boundary requires FP32 input')
    return value.to(dtype).to(torch.float32)


def quantization_record(value, restored):
    row = error_metrics(restored, value)
    row.update(new_nonfinite_count=int((torch.isfinite(value) & ~torch.isfinite(restored)).sum().item()),
               new_zero_count=int(((value != 0) & (restored == 0)).sum().item()),
               changed_count=int((restored != value).sum().item()))
    return row


def clear_cache():
    if hasattr(torch.ops.converse2d, 'clear_cache'):
        torch.ops.converse2d.clear_cache()


@contextmanager
def sensitivity_scope(model, dtype_name, variant, rows):
    """Own only newly registered hooks and temporary Converse2D weights.

    Forward must run inside no_grad/inference_mode. Generated USRNet kernels
    are transformed as call arguments; learned weight masters are restored by
    object identity. No parameter storage is edited in place.
    """
    from models.util_converse import Converse2D
    from models.converse_usrnet import ConvReverseDataNet
    baseline = variant == 'fp32'
    flags = (False, False, False) if baseline else VARIANTS[variant]
    input_cast, weight_cast, output_cast = flags
    dtype = None if baseline else DTYPES[dtype_name]
    selected = [(name, module) for name, module in model.named_modules()
                if isinstance(module, (Converse2D, ConvReverseDataNet))]
    if not selected:
        raise RuntimeError('No Converse2D boundaries selected')
    saved, handles, stacks, calls = [], [], defaultdict(list), Counter()
    masters = [(name, value, value.data_ptr(), value._version) for name, value in model.named_parameters()]
    before = state_hash(model)
    hook_ids = {name: (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
                for name, module in selected}
    weight_stats = {}
    clear_cache()
    try:
        for name, module in selected:
            if isinstance(module, Converse2D):
                master = module.weight
                with torch.no_grad():
                    diagnostic = {label: quantization_record(master, quantize(master, low))
                                  for label, low in (DTYPES.items() if baseline else ((dtype_name, dtype),))}
                    if weight_cast:
                        temporary = torch.nn.Parameter(quantize(master, dtype).clone(),
                                                       requires_grad=master.requires_grad)
                        saved.append((module, master))
                        module.weight = temporary
                weight_stats[name] = dict(master=range_record(master), quantization=diagnostic,
                                          applied=weight_cast)

            def pre(this, arguments, name=name):
                if torch.is_grad_enabled():
                    raise RuntimeError('Model sensitivity hooks are inference-only')
                if torch.is_autocast_enabled('cuda') or torch.is_autocast_enabled('cpu'):
                    raise RuntimeError('Autocast is not part of this sensitivity experiment')
                if not arguments or not torch.is_tensor(arguments[0]):
                    raise ValueError('Expected positional FP32 activation')
                values = list(arguments)
                value = values[0]
                call = calls[name]
                calls[name] += 1
                row = dict(layer=name, layer_type=type(this).__name__, call=call,
                           input=range_record(value), input_quantization={},
                           input_cast_applied=input_cast, output_cast_applied=output_cast,
                           bias_or_alpha_fp32=True)
                for label, low in (DTYPES.items() if baseline else ((dtype_name, dtype),)):
                    restored = quantize(value, low)
                    row['input_quantization'][label] = quantization_record(value, restored)
                    if input_cast:
                        values[0] = restored
                if isinstance(this, ConvReverseDataNet):
                    if len(values) < 3 or not torch.is_tensor(values[1]):
                        raise ValueError('Expected USRNet x, generated kernel, scale')
                    kernel = values[1]
                    row.update(scale=int(values[2]), generated_kernel=range_record(kernel),
                               kernel_quantization={}, kernel_cast_applied=weight_cast)
                    for label, low in (DTYPES.items() if baseline else ((dtype_name, dtype),)):
                        restored = quantize(kernel, low)
                        row['kernel_quantization'][label] = quantization_record(kernel, restored)
                        if weight_cast:
                            values[1] = restored
                    if this.alpha.dtype != torch.float32:
                        raise ValueError('USRNet alpha master must stay FP32')
                else:
                    row.update(scale=int(this.scale), learned_weight=weight_stats[name])
                    if this.bias.dtype != torch.float32 or this.weight.dtype != torch.float32:
                        raise ValueError('Converse2D bias/compute weight must stay FP32')
                stacks[id(this)].append(row)
                return tuple(values)

            def post(this, arguments, output):
                if output.dtype != torch.float32:
                    raise ValueError('Production solver output must remain FP32')
                row = stacks[id(this)].pop()
                row['output'] = range_record(output)
                row['output_quantization'] = {}
                returned = output
                for label, low in (DTYPES.items() if baseline else ((dtype_name, dtype),)):
                    restored = quantize(output, low)
                    row['output_quantization'][label] = quantization_record(output, restored)
                    if output_cast:
                        returned = restored
                rows.append(row)
                return returned

            handles.append(module.register_forward_pre_hook(pre))
            handles.append(module.register_forward_hook(post))
        yield dict(module_count=len(selected), calls=calls)
    finally:
        for handle in handles:
            handle.remove()
        for module, master in reversed(saved):
            module.weight = master
        clear_cache()
        current = dict(model.named_parameters())
        if any(current.get(name) is not value or value.data_ptr() != pointer or value._version != version
               for name, value, pointer, version in masters):
            raise RuntimeError('Master parameter identity/version changed during sensitivity experiment')
        if state_hash(model) != before:
            raise RuntimeError('Model state changed during sensitivity experiment')
        if any(hook_ids[name] != (tuple(module._forward_pre_hooks), tuple(module._forward_hooks))
               for name, module in selected):
            raise RuntimeError('Pre-existing module hook registrations were not restored')


def image_samples(args):
    """Fixed audited holdout center crops; no source image is modified."""
    if args.synthetic:
        return [dict(key=f'synthetic-{index}', seed=args.seed + index, rgb=None,
                     provenance=dict(kind='synthetic_no_ground_truth', seed=args.seed + index))
                for index in range(args.samples)]
    from PIL import Image, ImageOps
    manifest = args.manifest
    if manifest is None:
        default = ROOT / 'artifacts/v4_campaign/dataset_absolute.json'
        manifest = default if default.exists() else None
    if manifest is None:
        raise ValueError('No real-data manifest available; pass --manifest or explicitly use --synthetic')
    manifest = manifest.resolve()
    document = json.loads(manifest.read_text(encoding='utf-8-sig'))
    directory = Path(document['image_root'])
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    records = sorted((row for row in document['images'] if row['split'] in ('validation', 'val', 'holdout')),
                     key=lambda row: row['relative_path'])
    if len(records) < args.samples:
        raise ValueError('Requested more samples than available held-out images')
    examples = []
    for index, row in enumerate(records[:args.samples]):
        path = (directory / row['relative_path']).resolve()
        if not path.is_relative_to(directory) or file_hash(path) != row['file_sha256']:
            raise ValueError('Manifest image path/hash mismatch: ' + str(path))
        with Image.open(path) as image:
            rgb = np.asarray(ImageOps.exif_transpose(image).convert('RGB')).copy()
        h, w = rgb.shape[:2]
        if (w, h) != (row['width'], row['height']) or min(h, w) < args.patch_size:
            raise ValueError('Image dimensions do not support the declared crop')
        rgb_hash = hashlib.sha256(f'RGB:{w}x{h}:'.encode() + rgb.tobytes()).hexdigest()
        if rgb_hash != row['rgb_sha256']:
            raise ValueError('Normalized RGB hash differs from manifest')
        top, left = (h - args.patch_size) // 2, (w - args.patch_size) // 2
        crop = np.ascontiguousarray(rgb[top:top + args.patch_size, left:left + args.patch_size])
        examples.append(dict(key=row['relative_path'], seed=args.seed + index, rgb=crop,
            provenance=dict(kind='real_heldout_photo_declared_synthetic_degradation',
                manifest=str(manifest), manifest_sha256=file_hash(manifest), file=str(path),
                file_sha256=row['file_sha256'], rgb_sha256=rgb_hash,
                crop=[top, left, args.patch_size, args.patch_size],
                crop_rgb_sha256=hashlib.sha256(crop.tobytes()).hexdigest(),
                limitation='Fixed local holdout crops; not an official benchmark or measured real LR/HR pairs')))
    return examples


def model_sample(name, example, args):
    from PIL import Image
    from utils import utils_image as image_util
    from tools.roadmap_quality.usrnet_training_data import degrade
    generator = torch.Generator().manual_seed(example['seed'])
    provenance = dict(example['provenance'])
    if example['rgb'] is None:
        channels = 1 if name == 'dncnn' else 3
        x = torch.rand(1, channels, 24, 32, generator=generator)
        target = None
    else:
        rgb = example['rgb']
        if name == 'dncnn':
            target = np.asarray(Image.fromarray(rgb).convert('L')).copy()[..., None]
            clean = torch.from_numpy(target.transpose(2, 0, 1).copy()).float()[None] / 255
            x = clean + torch.randn(clean.shape, generator=generator) * (25 / 255)
            provenance['degradation'] = 'PIL L grayscale; FP32 AWGN sigma=25/255, unclipped'
        elif name == 'srresnet':
            target = rgb
            low = image_util.imresize_np(rgb.astype(np.float32) / 255, 1 / 4, antialiasing=True)
            x = torch.from_numpy(np.ascontiguousarray(low.transpose(2, 0, 1)))[None]
            provenance['degradation'] = 'Repository MATLAB-style bicubic downsample x4, antialiasing, unclipped'
        else:
            target = rgb
            x = None
    kernel = None
    if name == 'usrnet':
        paths = sorted((ROOT / 'blur_kernels').glob('kernel_*.npy'))
        if len(paths) != 5:
            raise ValueError('Expected five declared repository blur kernels')
        path = paths[example['seed'] % len(paths)]
        array = np.load(path, allow_pickle=False)
        if array.shape != (7, 7) or not np.isfinite(array).all() or np.iscomplexobj(array):
            raise ValueError('Invalid repository blur kernel')
        array = np.ascontiguousarray(array, dtype=np.float32)
        kernel = torch.from_numpy(array.copy()).reshape(1, 1, 7, 7)
        provenance.update(kernel_file=str(path), kernel_sha256=file_hash(path),
                          kernel_sum=float(array.sum(dtype=np.float64)), scale=args.usr_scale)
        if example['rgb'] is not None:
            low = degrade(example['rgb'].astype(np.float32) / 255, array, args.usr_scale,
                          .01, np.random.default_rng(example['seed']))
            x = torch.from_numpy(np.ascontiguousarray(low.transpose(2, 0, 1)))[None]
            provenance['degradation'] = 'Repository 7x7 circular blur, phase-zero downsample, unclipped AWGN sigma=.01'
    arguments = (x, kernel, args.usr_scale) if name == 'usrnet' else (x,)
    provenance['seed'] = example['seed']
    provenance['arguments'] = {str(index): tensor_record(value) if torch.is_tensor(value) else value
                               for index, value in enumerate(arguments)}
    provenance['target_sha256'] = None if target is None else hashlib.sha256(target.tobytes()).hexdigest()
    return arguments, target, provenance


def quality(output, target, border):
    if target is None:
        return dict(available=False, reason='No ground truth supplied for synthetic sensitivity inputs')
    if not bool(torch.isfinite(output).all()):
        return dict(available=False, reason='Nonfinite prediction; PSNR/SSIM not evaluated')
    from utils import utils_image as image_util
    raw = output.detach().cpu()[0].permute(1, 2, 0).numpy()
    if raw.shape != target.shape or min(target.shape[:2]) - 2 * border < 11:
        raise ValueError('Ground-truth/output shapes or SSIM crop are incompatible')
    prediction = np.rint(np.clip(raw, 0, 1) * 255).astype(np.uint8)
    psnr = float(image_util.calculate_psnr(prediction, target, border=border))
    ssim = float(image_util.calculate_ssim(prediction, target, border=border))
    return dict(available=True, psnr_db=psnr if math.isfinite(psnr) else None,
                perfect_uint8_match=math.isinf(psnr) and psnr > 0,
                ssim=ssim if math.isfinite(ssim) else None, border=border,
                clipped_fraction=float(np.mean((raw < 0) | (raw > 1))),
                protocol='Prediction clipped [0,1], rounded uint8; same uint8 GT; repository PSNR/SSIM, RGB or grayscale')


def distribution(values):
    finite = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return dict(count=0, unavailable=len(values))
    qs = np.quantile(finite, [0, .1, .5, .9, 1]).tolist()
    return dict(count=len(finite), unavailable=len(values)-len(finite), mean=statistics.mean(finite),
                minimum=qs[0], p10=qs[1], median=qs[2], p90=qs[3], maximum=qs[4])


def summaries(report):
    groups = defaultdict(list)
    layer_groups = defaultdict(list)
    for row in report['rows']:
        groups[(row['model'], row['storage_dtype'], row['variant'])].append(row)
        for layer in row.get('layers', []):
            for label, error in layer['input_quantization'].items():
                layer_groups[(row['model'], row['variant'], label, layer['layer'], 'activation')].append(error)
            for label, error in layer.get('kernel_quantization', {}).items():
                layer_groups[(row['model'], row['variant'], label, layer['layer'], 'generated_kernel')].append(error)
            for label, error in layer.get('learned_weight', {}).get('quantization', {}).items():
                layer_groups[(row['model'], row['variant'], label, layer['layer'], 'learned_weight')].append(error)
            for label, error in layer.get('output_quantization', {}).items():
                layer_groups[(row['model'], row['variant'], label, layer['layer'], 'solver_output')].append(error)
    overall = []
    for (name, dtype, variant), rows in groups.items():
        entry = dict(model=name, storage_dtype=dtype, variant=variant, samples=len(rows),
                     runtime_errors=sum(row.get('status') == 'error' for row in rows),
                     nonfinite_outputs=sum(not row.get('difference', {}).get('finite', False) for row in rows))
        for key in ('rel_l2', 'max_abs'):
            entry[key] = distribution([row.get('difference', {}).get(key) for row in rows])
        for key in ('psnr_db', 'ssim'):
            entry[key] = distribution([row.get('quality', {}).get(key) for row in rows])
            entry[key + '_delta_vs_fp32'] = distribution([row.get('quality_delta_vs_fp32', {}).get(key) for row in rows])
        overall.append(entry)
    ranked = []
    for (name, variant, dtype, layer, role), rows in layer_groups.items():
        ranked.append(dict(model=name, variant=variant, storage_dtype=dtype, layer=layer, role=role,
            calls=len(rows), rel_l2=distribution([row['rel_l2'] for row in rows]),
            max_abs=distribution([row['max_abs'] for row in rows]),
            new_nonfinite_count=sum(row['new_nonfinite_count'] for row in rows),
            new_zero_count=sum(row['new_zero_count'] for row in rows)))
    ranked.sort(key=lambda row: (row['new_nonfinite_count'], row['rel_l2'].get('maximum', -1)), reverse=True)
    return dict(model_distributions=overall, layer_boundary_quantization_ranking=ranked,
        ranking_limit='Boundary quantization on actual propagated activations; not a causal one-layer ablation ranking',
        quality_thresholds=None, production_admission=False, convergence_evidence=False)


def make_model(name, device):
    from models.converse_dncnn import ConverseDnCNN
    from models.converse_srresnet import ConverseMSRResNet
    from models.converse_usrnet import ConverseUSRNet
    constructors = dict(dncnn=ConverseDnCNN, srresnet=ConverseMSRResNet, usrnet=ConverseUSRNet)
    model = constructors[name]().to(device).eval()
    path = ROOT / 'model_zoo' / CHECKPOINTS[name]
    model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True), strict=True)
    for layer in model.modules():
        if hasattr(layer, 'backend'):
            layer.backend = 'cuda' if device.type == 'cuda' else 'auto'
    if any(value.dtype != torch.float32 for value in model.parameters()):
        raise ValueError('All model masters must be FP32')
    return model, dict(path=str(path), sha256=file_hash(path), state_sha256=state_hash(model))


def run(args, report):
    if os.environ.get('CONVERSE2D_BACKEND') or os.environ.get('CONVERSE2D_CPU_ONLY') == '1':
        raise RuntimeError('Unset backend and CPU-only overrides for an actual-production sensitivity run')
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable')
        os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
        from extension_loader import load_extension
        load_extension()
        report['checked_build'] = json.loads((ROOT / '.build/cuda/source_manifest.json').read_text())
    else:
        report['checked_build'] = None
        report['cpu_limit'] = 'Portable FP32 model only; CPU results do not validate CUDA arithmetic'
    old_det = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    old_matmul = torch.backends.cuda.matmul.allow_tf32
    report['environment'] = dict(torch=str(torch.__version__), cuda=torch.version.cuda, device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        autocast=False, tf32=False, deterministic=True, cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    examples = image_samples(args)
    report['samples'] = [dict(key=e['key'], **e['provenance']) for e in examples]
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        with torch.backends.cudnn.flags(enabled=True, benchmark=False, deterministic=True, allow_tf32=False):
            for name in args.models:
                model, checkpoint = make_model(name, device)
                report['checkpoints'][name] = checkpoint
                border = 0 if name == 'dncnn' else 4 if name == 'srresnet' else args.usr_scale
                for example in examples:
                    cpu_args, target, provenance = model_sample(name, example, args)
                    inputs = tuple(value.to(device) if torch.is_tensor(value) else value for value in cpu_args)
                    input_state = [tensor_record(value) for value in inputs if torch.is_tensor(value)]
                    layers = []
                    with sensitivity_scope(model, None, 'fp32', layers) as scope, torch.inference_mode():
                        baseline = model(*inputs).detach().clone()
                    baseline_calls = dict(scope['calls'])
                    expected = {'dncnn': 20, 'srresnet': 21, 'usrnet': 40}[name]
                    if sum(baseline_calls.values()) != expected:
                        raise RuntimeError('Incomplete model boundary coverage: ' + str(baseline_calls))
                    if name == 'usrnet' and baseline_calls.get('d') != 5:
                        raise RuntimeError('All five USRNet generated-kernel calls must be covered')
                    if not bool(torch.isfinite(baseline).all()):
                        raise RuntimeError('FP32 model baseline is nonfinite: ' + name + '/' + example['key'])
                    baseline_quality = quality(baseline, target, border)
                    report['rows'].append(dict(model=name, sample=example['key'], storage_dtype='fp32',
                        variant='fp32', status='complete', provenance=provenance, layers=layers,
                        boundary_calls=dict(scope['calls']), output=tensor_record(baseline),
                        difference=error_metrics(baseline, baseline), quality=baseline_quality,
                        state_restored=True, output_restored=True))
                    # Separate final-model output storage cast from per-solver output hooks.
                    report['rows'][-1]['final_model_output_cast_diagnostics'] = {
                        label: dict(difference=error_metrics(quantize(baseline, low), baseline),
                                    quality=quality(quantize(baseline, low), target, border))
                        for label, low in DTYPES.items()}
                    for label in args.dtypes:
                        for variant in args.variants:
                            row = dict(model=name, sample=example['key'], storage_dtype=label, variant=variant,
                                       status='running', provenance=provenance, layers=[])
                            report['rows'].append(row)
                            try:
                                with sensitivity_scope(model, label, variant, row['layers']) as scope, torch.inference_mode():
                                    actual = model(*inputs).detach().clone()
                                if dict(scope['calls']) != baseline_calls:
                                    raise RuntimeError('Experimental model boundary coverage differs from baseline')
                                row.update(status='complete', boundary_calls=dict(scope['calls']),
                                           output=tensor_record(actual), difference=error_metrics(actual, baseline),
                                           quality=quality(actual, target, border), state_restored=True)
                                row['quality_delta_vs_fp32'] = {
                                    key: row['quality'][key] - baseline_quality[key]
                                    if row['quality'].get(key) is not None and baseline_quality.get(key) is not None else None
                                    for key in ('psnr_db', 'ssim')}
                            except Exception as error:
                                row.update(status='error', error=repr(error), traceback=traceback.format_exc())
                            clear_cache()
                            with torch.inference_mode():
                                restored = model(*inputs)
                            row['output_restored'] = bool(torch.equal(restored, baseline))
                            if not row['output_restored'] or state_hash(model) != checkpoint['state_sha256']:
                                raise RuntimeError('Baseline restoration failed after ' + label + '/' + variant)
                            if input_state != [tensor_record(value) for value in inputs if torch.is_tensor(value)]:
                                raise RuntimeError('Caller inputs changed during sensitivity experiment')
                            print(name, example['key'], label, variant, row['status'], row.get('difference'), flush=True)
                del model
                clear_cache()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_matmul
        torch.use_deterministic_algorithms(old_det, warn_only=old_warn)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--synthetic', action='store_true', help='No GT/quality claims; explicit random sensitivity inputs')
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--usr-scale', type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--models', nargs='+', choices=tuple(CHECKPOINTS), default=list(CHECKPOINTS))
    parser.add_argument('--dtypes', nargs='+', choices=tuple(DTYPES), default=list(DTYPES))
    parser.add_argument('--variants', nargs='+', choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        parser.error('Choose a new report path; previous successes and failures are retained')
    if args.samples < 1 or args.patch_size < 24 or args.patch_size % 4 or args.patch_size % args.usr_scale:
        parser.error('Positive samples and patch-size >=24 divisible by 4 and usr-scale required')
    if args.synthetic and args.manifest:
        parser.error('Choose an audited manifest or explicit synthetic inputs, not both')
    paths = [Path(__file__), ROOT / 'test/extension_loader.py', ROOT / 'utils/utils_image.py',
             ROOT / 'tools/roadmap_quality/usrnet_training_data.py', *sorted((ROOT / 'models').glob('*.py'))]
    report = dict(schema_version=1, kind='pretrained_quantization_sensitivity', status='running',
        created_utc=datetime.now(timezone.utc).isoformat(), rows=[], checkpoints={},
        git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths},
        simulation='FP16/BF16 quantize->float boundaries; complete model computation remains FP32',
        quantized_boundaries='Every Converse2D and USRNet ConvReverseDataNet activation; optional learned/generated kernel and solver output',
        preserved='All master bias/alpha and all other layers remain FP32; no AMP, TF32 or low-precision FFT',
        not_claimed=['production admission', 'training support', 'complete AMP', 'memory savings', 'speedup', 'convergence'],
        metric_work_is_untimed=True, quality_thresholds=None, settings=vars(args).copy())
    report['settings'] = {key: str(value) if isinstance(value, Path) else value for key, value in report['settings'].items()}
    try:
        run(args, report)
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        report['summary'] = summaries(report)
        report['all_outputs_finite'] = bool(report['rows']) and all(row.get('difference', {}).get('finite', False) for row in report['rows'])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        print('Report:', args.output, flush=True)


if __name__ == '__main__':
    main()
