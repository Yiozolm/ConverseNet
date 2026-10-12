"""Independent bit-exact pad/cast and complete-module admission after baseline.

Only the root executes this CUDA gate. Numerical failures are retained per case;
setup/source-identity failures stop admission. The kernel block size is recorded
and never selected by this gate. No timing is performed here.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F
import fp32_baseline as frozen
from numerical_policy import comparison, denominator_statistics
from tools.v4_mixed_fusion import baseline, loader

DTYPES = {'fp16': torch.float16, 'bf16': torch.bfloat16}
MODES = ('circular', 'reflect', 'replicate', 'constant')
GEOMETRIES = ((1,3,5,7,2,3), (2,3,17,19,2,3), (1,3,3,33,2,3),
              (1,3,15,23,2,3), (1,64,31,37,2,3), (4,128,96,96,2,3), (1,64,96,96,6,7))
DEPENDENCIES = ('tools/v4_mixed_fusion/gate.py', 'tools/v4_mixed_fusion/baseline.py',
                'tools/v4_mixed_fusion/adapter.py', 'tools/v4_mixed_fusion/loader.py',
                'models/util_converse.py', 'models/converse_core.py', 'models/pointwise.py',
                'test/fp32_baseline.py', 'test/numerical_policy.py', 'test/extension_loader.py',
                'Converse2D/build_config.py')
ADMISSION_SCOPE = dict(input_dtype='float16',padding_mode='circular',scale=1,min_eps=1e-5,
                       weight_dtype='float32',bias_dtype='float32',output_dtype='float32',
                       layout='contiguous_NCHW',execution='inference_or_frozen_gradmode')
SCOPE_VERSION = 'normal_regularization_v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(value):
    raw = value.detach().resolve_neg().resolve_conj().cpu().contiguous().reshape(-1).view(torch.uint8)
    return dict(shape=list(value.shape), stride=list(value.stride()), storage_offset=value.storage_offset(),
                dtype=str(value.dtype), contiguous=value.is_contiguous(),
                finite=bool(torch.isfinite(value).all()), sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest())


def exact(a, b):
    left, right = record(a), record(b)
    return dict(passed=left == right and left['finite'] and right['finite'], candidate=left, baseline=right)


def finite_records_ok(value):
    if isinstance(value,dict):
        return ('finite' not in value or value['finite'] is True) and all(finite_records_ok(v) for v in value.values())
    if isinstance(value,(list,tuple)):
        return all(finite_records_ok(v) for v in value)
    return not isinstance(value,float) or math.isfinite(value)


def scope_matches(row):
    eps=row.get('module_eps')
    return (row.get('kind')=='module' and row.get('dtype')=='fp16' and row.get('padding_mode')=='circular'
            and row.get('scale')==1 and row.get('layout')=='contiguous'
            and row.get('context') in ('no_grad','inference_mode','frozen')
            and isinstance(eps,(int,float)) and math.isfinite(eps) and eps>=ADMISSION_SCOPE['min_eps'])


def module_equivalence_ok(row):
    required=('route_verified','fp32_shared_prior_and_mode_verified','input_parameters_unchanged',
              'parameter_identity_preserved','caller_state_preserved','current_reference_finite','no_output_grad')
    return (row.get('exact_current_module',{}).get('passed') is True
            and row.get('repeated_exact',{}).get('passed') is True
            and all(row.get(key) is True for key in required) and finite_records_ok(row))


def admission_partition(rows, *, expected_cases=None, expected_active=None):
    """Admit only the new explicit domain; never rewrite full-matrix verdicts."""
    active=[]; fallback=[]; active_fail=[]; fallback_fail=[]; contract_fail=[]; inherited=[]
    for row in rows:
        name=row.get('name','<unnamed>')
        if row.get('kind')!='module':
            if row.get('passed') is not True or not finite_records_ok(row): contract_fail.append(name)
            continue
        selected=scope_matches(row)
        current=row.get('baseline_quantized_fp32_budget',{})
        candidate=row.get('candidate_quantized_fp32_budget',{})
        common=(module_equivalence_ok(row) and row.get('expected_fused') is selected
                and row.get('fused_calls')==(2 if selected else 0))
        if selected:
            active.append(name)
            if not (common and row.get('passed') is True and current.get('passed') is True
                    and candidate.get('passed') is True and row.get('baseline_budget_failure') is False):
                active_fail.append(name)
        else:
            fallback.append(name)
            budgets_pass=current.get('passed') is True and candidate.get('passed') is True
            # Equal current-module bytes imply equal errors to the same RQ/RQ64.
            # A candidate-only numerical failure cannot hide as baseline debt.
            explained=(row.get('baseline_budget_failure') is True and current.get('passed') is False
                       and candidate.get('passed') is False and current==candidate)
            verdict_consistent=(row.get('passed') is True and budgets_pass and row.get('baseline_budget_failure') is False
                                or row.get('passed') is False and explained)
            if not (common and verdict_consistent): fallback_fail.append(name)
            elif explained: inherited.append(name)
    complete=(expected_cases is None or len(rows)==expected_cases) and len({r.get('name') for r in rows})==len(rows)
    active_coverage=bool(active) and (expected_active is None or len(active)==expected_active)
    passed=complete and active_coverage and not(active_fail or fallback_fail or contract_fail)
    return dict(passed=bool(passed),active_module_cases=len(active),fallback_module_cases=len(fallback),
                expected_cases=expected_cases,expected_active_cases=expected_active,complete_rows=complete,
                active_failures=active_fail,fallback_failures=fallback_fail,contract_failures=contract_fail,
                unchanged_fallback_budget_failures=inherited,
                note='Fallback budget failures stay failed rows; only unchanged finite fallback equivalence is admitted.')


def valid_padding(h, w, amount, mode):
    return type(amount) is int and amount > 0 and mode in MODES and (
        mode not in ('circular', 'reflect') or (amount <= min(h,w) if mode == 'circular' else amount < min(h,w)))


def pad_specs():
    for dtype in DTYPES:
        for mode in MODES:
            yield dict(kind='pad_exact', name=f'pad/{dtype}/{mode}/all_finite_patterns', dtype=dtype,
                       mode=mode, pattern='all_finite', shape=None, padding=2)
            for shape in ((1,1,3,33), (1,3,15,23), (2,3,5,7), (1,64,31,37), (4,128,96,96)):
                for pattern in ('coordinate', 'random'):
                    yield dict(kind='pad_exact', name=f'pad/{dtype}/{mode}/{shape}/{pattern}',
                               dtype=dtype, mode=mode, pattern=pattern, shape=shape, padding=2)
            for h,w,amount in ((1,1,1), (2,3,2), (1,17,3), (2,3,6)):
                if valid_padding(h,w,amount,mode):
                    yield dict(kind='pad_exact', name=f'pad/{dtype}/{mode}/edge{h}x{w}p{amount}',
                               dtype=dtype, mode=mode, pattern='coordinate', shape=(1,3,h,w), padding=amount)


def finite_patterns(dtype):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(dtype)
    values = values[torch.isfinite(values)]
    return values.reshape(1,1,256,-1)


def pad_input(spec, device):
    dtype = DTYPES[spec['dtype']]
    if spec['pattern'] == 'all_finite':
        value = finite_patterns(dtype)
    elif spec['pattern'] == 'coordinate':
        value = (torch.arange(math.prod(spec['shape']), dtype=torch.float32) % 97 - 48).reshape(spec['shape']).to(dtype)
    else:
        value = torch.randn(spec['shape'], generator=torch.Generator().manual_seed(51733)).to(dtype)
    return value.to(device)


def module_specs():
    for dtype in DTYPES:
        for mode in MODES:
            for index, geom in enumerate(GEOMETRIES):
                for regime in ('normal', 'weak'):
                    for context in ('no_grad', 'inference_mode', 'frozen'):
                        b,c,h,w,p,k = geom
                        kb = b if index % 2 else 1
                        kc = c if (index//2) % 2 else 1
                        yield dict(kind='module', name=f'module/{dtype}/{mode}/g{index}/{regime}/{context}',
                                   dtype=dtype, padding_mode=mode, shape=(b,c,h,w), padding=p, kernel=k,
                                   kb=kb, kc=kc, regime=regime, context=context, scale=1, layout='contiguous')
        # Real B2/C3 cross-product, rather than treating B1's degenerate KB=1
        # as proof of all four broadcast reductions.
        for kb,kc in ((1,1),(1,3),(2,1),(2,3)):
            for regime in ('normal','weak'):
                for context in ('no_grad','inference_mode','frozen'):
                    yield dict(kind='module',name=f'module/{dtype}/broadcast{kb}x{kc}/{regime}/{context}',
                               dtype=dtype,padding_mode='circular',shape=(2,3,5,7),padding=2,kernel=3,
                               kb=kb,kc=kc,regime=regime,context=context,scale=1,layout='contiguous')


def context(name):
    return {'no_grad': torch.no_grad, 'inference_mode': torch.inference_mode,
            'frozen': torch.enable_grad, 'training': torch.enable_grad}[name]()


class Spy:
    def __init__(self, extension):
        self.extension, self.calls = extension, []
    def pad_cast(self, x, padding, mode):
        self.calls.append(dict(shape=list(x.shape), dtype=str(x.dtype), padding=padding, mode=mode,
                               grad_enabled=torch.is_grad_enabled(), requires_grad=x.requires_grad))
        return self.extension.pad_cast(x, padding, mode)


def fixture(spec, device='cuda'):
    from models.util_converse import Converse2D
    b,c,h,w = spec['shape']
    generator = torch.Generator().manual_seed(53117 + h + 13*w + c)
    raw = torch.randn(spec['shape'], generator=generator).to(device=device, dtype=DTYPES[spec['dtype']])
    if spec.get('layout') == 'strided':
        raw = torch.stack((raw, raw), -1)[...,0]
    elif spec.get('layout') == 'transpose':
        raw = raw.transpose(-1,-2).contiguous().transpose(-1,-2)
    elif spec.get('layout') == 'negative':
        raw = torch._neg_view(-raw)
    k = spec['kernel']
    weight = torch.randn(spec['kb'],spec['kc'],k,k,generator=generator) / (k*k)**.5
    if spec.get('regime') == 'weak':
        weight.mul_(1e-6)
    bias = torch.full((1,c,1,1), -40.) if spec.get('regime') == 'weak' else torch.randn(1,c,1,1,generator=generator)*.2
    module = Converse2D(c,c,k,scale=spec.get('scale',1),padding=spec['padding'],
                        padding_mode=spec['padding_mode'],eps=1e-8 if spec.get('regime') == 'weak' else 1e-5,
                        backend=spec.get('backend','cuda' if device == 'cuda' else 'auto')).to(device)
    module.weight = torch.nn.Parameter(weight.to(device), requires_grad=False)
    module.bias = torch.nn.Parameter(bias.to(device), requires_grad=False)
    return module, raw


def expected_fast(module, x):
    # Independent selected scope, not the candidate implementation's predicate.
    return (x.is_cuda and x.dtype == torch.float16 and x.ndim == 4 and x.is_contiguous()
            and not x.is_neg() and not x.is_conj() and type(module.scale) is int and module.scale == 1
            and module.padding_mode == 'circular' and valid_padding(*x.shape[-2:],module.padding,'circular')
            and module.backend in ('auto','cuda') and module.variant == 'v7'
            and isinstance(module.eps,(int,float)) and math.isfinite(module.eps) and module.eps>=ADMISSION_SCOPE['min_eps']
            and not (torch.is_grad_enabled() and any(t.requires_grad for t in (x,module.weight,module.bias)))
            and not torch.is_autocast_enabled(x.device.type)
            and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32)


def references(module, x):
    p = module.padding
    q = x.detach().float()
    padded = F.pad(q,(p,)*4,mode=module.padding_mode,value=0) if p else q
    prior = padded if module.scale == 1 else F.interpolate(padded,scale_factor=module.scale,mode='nearest')
    ref_fn = frozen.converse2d_reference if module.backend == 'pytorch' else frozen.converse2d_fp32
    low = ref_fn(padded,prior,module.weight,module.bias,module.scale,module.eps)
    high_input = padded.double()
    high_prior = high_input if prior is padded else prior.double()
    high = frozen.converse2d_reference(high_input, high_prior,
                                      module.weight.double(),module.bias.double(),module.scale,module.eps)
    cut = p*module.scale
    if cut:
        low, high = low[...,cut:-cut,cut:-cut], high[...,cut:-cut,cut:-cut]
    stats = denominator_statistics((padded,prior,module.weight,module.bias),module.scale,module.eps,x.device)
    return low, high, stats


def module_case(extension, adapter, spec):
    module, x = fixture(spec)
    if spec['context'] in ('no_grad','inference_mode'):
        x.requires_grad_(True); module.weight.requires_grad_(True); module.bias.requires_grad_(True)
    before = [record(v) for v in (x,module.weight,module.bias)]
    before_requires = [v.requires_grad for v in (x,module.weight,module.bias)]
    before_identity = [(id(v),v.data_ptr(),v._version) for v in (x,module.weight,module.bias)]
    spy, core_calls = Spy(extension), []
    with context(spec['context']):
        state_before=(torch.is_grad_enabled(),torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'),
                      torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        eligible = expected_fast(module,x)
        control = module(x.float())
        original_core = torch.ops.converse2d.forward
        def observe(a,b,*args,**kwargs):
            core_calls.append(dict(shared=a is b, dtype=str(a.dtype), grad_enabled=torch.is_grad_enabled(), shape=list(a.shape)))
            return original_core(a,b,*args,**kwargs)
        with patch.object(torch.ops.converse2d,'forward',new=observe):
            actual = adapter.mixed_module_forward(module,x,spy,original_forward=module.forward)
            repeated = adapter.mixed_module_forward(module,x,spy,original_forward=module.forward)
        with torch.no_grad():
            low, high, stats = references(module,x)
        exact_check = exact(actual,control)
        current_budget = comparison(control,low,high,regime=spec['regime'])
        candidate_budget = comparison(actual,low,high,regime=spec['regime'])
        state_preserved=state_before==(torch.is_grad_enabled(),torch.is_autocast_enabled('cuda'),torch.get_autocast_dtype('cuda'),
                                      torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    route = len(spy.calls) == (2 if eligible else 0)
    core_verified = (len(core_calls) == 2 and all(v['shared'] and v['dtype']=='torch.float32'
                     and v['grad_enabled']==(spec['context']=='frozen') for v in core_calls))
    unchanged = (before == [record(v) for v in (x,module.weight,module.bias)]
                 and before_requires == [v.requires_grad for v in (x,module.weight,module.bias)])
    identity_preserved=before_identity==[(id(v),v.data_ptr(),v._version) for v in (x,module.weight,module.bias)]
    repeated_check=exact(actual,repeated)
    reference_finite=all(budget[key]['finite'] for budget in (current_budget,candidate_budget) for key in ('baseline','candidate'))
    no_output_grad=not any(v.requires_grad for v in (actual,control,repeated))
    return dict(**spec, passed=bool(exact_check['passed'] and current_budget['passed'] and candidate_budget['passed']
                and repeated_check['passed'] and route and unchanged and identity_preserved and state_preserved and core_verified and no_output_grad),
                module_eps=float(module.eps),repeated_exact=repeated_check,current_reference_finite=reference_finite,
                parameter_identity_preserved=identity_preserved,no_output_grad=no_output_grad,
                exact_current_module=exact_check, baseline_quantized_fp32_budget=current_budget,
                candidate_quantized_fp32_budget=candidate_budget, baseline_budget_failure=not current_budget['passed'],
                denominator_statistics=stats, expected_fused=eligible, fused_calls=len(spy.calls),
                route_verified=route, fp32_shared_prior_and_mode_verified=core_verified,
                core_calls=core_calls, input_parameters_unchanged=unchanged,caller_state_preserved=state_preserved)


def pad_case(extension, spec):
    x = pad_input(spec,'cuda')
    before = record(x)
    with torch.no_grad():
        expected = F.pad(x.float(),(spec['padding'],)*4,mode=spec['mode'],value=0)
        actual = extension.pad_cast(x,spec['padding'],spec['mode'])
    check = exact(actual,expected)
    return dict(**spec, actual_shape=list(actual.shape), passed=check['passed'] and before==record(x)
                and actual.dtype==torch.float32 and actual.is_contiguous() and not actual.requires_grad,
                exact_padding=check, input_unchanged=before==record(x),
                finite_pattern_count=x.numel() if spec['pattern']=='all_finite' else None,
                scope='Pad component only; finite largest BF16 values do not imply finite FFT-domain arithmetic')


def fallback_case(extension, adapter, label, dtype):
    spec = dict(dtype=dtype,padding_mode='circular',shape=(1,3,5,7),padding=2,kernel=3,kb=1,kc=3,regime='normal',scale=1)
    if label in ('strided','transpose','negative'):
        spec['layout'] = label
    elif label == 'padding_zero': spec['padding'] = 0
    elif label == 'scale2': spec['scale'] = 2
    elif label == 'pytorch': spec['backend'] = 'pytorch'
    device = 'cpu' if label=='cpu' else 'cuda'
    module,x = fixture(spec,device)
    if label == 'fp32': x=x.float()
    spy = Spy(extension)
    with torch.no_grad(), (torch.autocast('cuda',dtype=torch.float16) if label=='autocast' else nullcontext()):
        state_before=(torch.is_grad_enabled(),torch.is_autocast_enabled('cuda'))
        expected = module(x.float())
        actual = adapter.mixed_module_forward(module,x,spy,original_forward=module.forward)
        state_preserved=state_before==(torch.is_grad_enabled(),torch.is_autocast_enabled('cuda'))
    check = exact(actual,expected)
    return dict(kind='fallback',name=f'fallback/{dtype}/{label}',passed=check['passed'] and not spy.calls and state_preserved,
                fused_calls=len(spy.calls),exact_current_module=check,caller_state_preserved=state_preserved)


def training_fallback(extension,adapter,dtype,mask):
    spec=dict(dtype=dtype,padding_mode='circular',shape=(1,3,5,7),padding=2,kernel=3,kb=1,kc=3,regime='normal')
    module,raw=fixture(spec)
    x=raw.detach().requires_grad_(mask[0]); module.weight.requires_grad_(mask[1]); module.bias.requires_grad_(mask[2])
    selected=[v for v in (x,module.weight,module.bias) if v.requires_grad]
    spy=Spy(extension); records=[]
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with torch.enable_grad():
            for call in (lambda:module(x.float()),lambda:adapter.mixed_module_forward(module,x,spy,original_forward=module.forward)):
                output=call()
                first=torch.autograd.grad(output.square().mean()*1e-3,selected,create_graph=True)
                # Bias-only and weight/input subsets use a scalar norm to retain
                # differentiable ATen fallback without FP16 loss overflow.
                loss=sum(g.float().square().sum() for g in first)
                second=torch.autograd.grad(loss,selected,allow_unused=True)
                records.append([record(output),*[record(v) for v in first],
                                *[None if v is None else record(v) for v in second]])
    finally:
        torch.use_deterministic_algorithms(previous)
    finite=finite_records_ok(records)
    return dict(kind='training_fallback',name=f'training/{dtype}/{mask}',passed=records[0]==records[1] and not spy.calls and finite,
                fused_calls=len(spy.calls),baseline=records[0],candidate=records[1],higher_order_checked=True,finite_verified=finite)


def cache_case(extension,adapter,dtype):
    spec=dict(dtype=dtype,padding_mode='circular',shape=(2,3,17,19),padding=2,kernel=3,kb=1,kc=3,regime='normal')
    module,x=fixture(spec); spy=Spy(extension); checks=[]
    with torch.no_grad():
        for stage in ('initial','warm','weight_update','bias_update','shape_change'):
            if stage=='weight_update': module.weight.add_(.01)
            if stage=='bias_update': module.bias.add_(.2)
            if stage=='shape_change': x=x[...,:13,:17].contiguous()
            expected=module(x.float())
            actual=adapter.mixed_module_forward(module,x,spy,original_forward=module.forward)
            checks.append(dict(stage=stage,**exact(actual,expected)))
    return dict(kind='cache',name='cache/'+dtype,passed=all(v['passed'] for v in checks) and finite_records_ok(checks)
                and len(spy.calls)==(5 if dtype=='fp16' else 0),checks=checks,fused_calls=len(spy.calls),finite_verified=finite_records_ok(checks))


def stream_graph_case(extension,adapter,dtype):
    spec=dict(dtype=dtype,padding_mode='circular',shape=(1,3,17,19),padding=2,kernel=3,kb=1,kc=3,regime='normal')
    module,x=fixture(spec); stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream),torch.no_grad():
        side=adapter.mixed_module_forward(module,x,extension,original_forward=module.forward)
        side_reference=module(x.float())
    stream.synchronize(); side_check=exact(side,side_reference)
    def capture(call):
        graph=torch.cuda.CUDAGraph(); torch.ops.converse2d.begin_graph_cache()
        try:
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                for _ in range(3): call()
            stream.synchronize()
            with torch.no_grad(),torch.cuda.graph(graph,stream=stream): output=call()
        finally:
            owners=tuple(torch.ops.converse2d.end_graph_cache())
        return graph,output,owners
    first=capture(lambda:module(x.float()))
    second=capture(lambda:adapter.mixed_module_forward(module,x,extension,original_forward=module.forward))
    checks=[]
    try:
        with torch.no_grad():
            for step in range(2):
                if step: x.add_(.125)
                first[0].replay(); second[0].replay(); torch.cuda.synchronize()
                checks.append(exact(second[1],first[1]))
    finally:
        first[0].reset(); second[0].reset()
    finite=finite_records_ok([side_check,*checks])
    return dict(kind='stream_graph',name='stream_graph/'+dtype,passed=side_check['passed'] and all(v['passed'] for v in checks) and finite,
                side_stream=side_check,graph_pairs=checks,owned_spectrum_counts=[len(first[2]),len(second[2])],
                finite_verified=finite,
                graph_scope='Matching captured baseline/candidate; input updates only; parameter changes require recapture')


def invalid_contracts(extension,adapter):
    spec=dict(dtype='fp16',padding_mode='circular',shape=(1,3,5,7),padding=2,kernel=3,kb=1,kc=3,regime='normal')
    module,x=fixture(spec); checks=[]
    cases=[('primitive_cpu',lambda:extension.pad_cast(x.cpu(),2,'circular')),
           ('primitive_fp32',lambda:extension.pad_cast(x.float(),2,'circular')),
           ('primitive_fp64',lambda:extension.pad_cast(x.double(),2,'circular')),
           ('primitive_zero_pad',lambda:extension.pad_cast(x,0,'circular')),
           ('primitive_bad_mode',lambda:extension.pad_cast(x,2,'invalid')),
           ('primitive_reflect_limit',lambda:extension.pad_cast(x,5,'reflect')),
           ('primitive_circular_limit',lambda:extension.pad_cast(x,6,'circular')),
           ('primitive_huge_pad',lambda:extension.pad_cast(x,2**30,'constant')),
           ('primitive_strided',lambda:extension.pad_cast(torch.stack((x,x),-1)[...,0],2,'circular')),
           ('primitive_negative',lambda:extension.pad_cast(torch._neg_view(-x),2,'circular')),
           ('adapter_fp64',lambda:adapter.mixed_module_forward(module,x.double(),extension))]
    with torch.no_grad():
        for name,call in cases:
            try: call()
            except (RuntimeError,ValueError,TypeError) as error: checks.append(dict(name=name,passed=True,error=str(error)))
            else: checks.append(dict(name=name,passed=False))
    try:
        with torch.enable_grad(): extension.pad_cast(x.detach().requires_grad_(),2,'circular')
    except (RuntimeError,ValueError,TypeError) as error: checks.append(dict(name='primitive_differentiable_input',passed=True,error=str(error)))
    else: checks.append(dict(name='primitive_differentiable_input',passed=False))
    try:
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16): extension.pad_cast(x,2,'circular')
    except (RuntimeError,ValueError,TypeError) as error: checks.append(dict(name='primitive_autocast',passed=True,error=str(error)))
    else: checks.append(dict(name='primitive_autocast',passed=False))
    return dict(kind='invalid_contract',name='invalid_contracts',passed=all(v['passed'] for v in checks),checks=checks)


def verify_baseline(path,production):
    report=json.loads(Path(path).read_text(encoding='utf-8'))
    expected={(name,mode) for name in baseline.DEFAULT_CASES for mode in MODES}
    observed={(r['specification']['name'],r['specification']['padding_mode']) for r in report.get('rows',[])}
    if report.get('kind')!='mixed_fusion_unchanged_baseline' or report.get('status')!='complete' or not report.get('passed') \
            or not report.get('gpu_evidence') or report.get('self_check') or observed!=expected or len(report['rows'])!=len(expected) \
            or not all(r['accuracy']['passed'] for r in report['rows']):
        raise RuntimeError('The complete 16-case admitted CUDA baseline report is required')
    for name,digest in report['source_sha256'].items():
        if sha(ROOT/name)!=digest: raise RuntimeError('Baseline source changed: '+name)
    if report['checked_manifest']!=production['checked_manifest'] or report['production_sources']!=production['production_source_sha256']:
        raise RuntimeError('Baseline checked production identity differs from current build')
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def sources(research):
    result={name:sha(ROOT/name) for name in DEPENDENCIES}
    result.update({(HERE/name).relative_to(ROOT).as_posix():sha(HERE/name) for name in research['identity']['local_sources']})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,required=True)
    parser.add_argument('--block-threads',type=int,choices=(128,256,512),default=256)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Use a fresh report filename')
    report=dict(kind='mixed_fusion_operator_gate',status='running',passed=False,full_matrix_passed=False,
                active_domain_admitted=False,admission_scope=ADMISSION_SCOPE,scope_version=SCOPE_VERSION,complete=False,
                created_utc=datetime.now(timezone.utc).isoformat(),block_threads=args.block_threads,rows=[],
                unverified=['Near-INT32-sized input allocations are not made; huge-padding overflow rejection is exercised.',
                            'No cross-GPU run; graphs do not promise parameter mutation without recapture.'],
                performance_admission=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save(): args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    with args.output.open('x',encoding='utf-8') as stream: json.dump(report,stream)
    try:
        torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
        torch.use_deterministic_algorithms(False)
        production=loader.load_production_checked()
        evidence=verify_baseline(args.baseline,production)
        extension,research=loader.load_checked(args.artifacts,block_threads=args.block_threads)
        if extension.block_threads != args.block_threads:
            raise RuntimeError('Loaded kernel block macro differs from requested configuration')
        adapter=importlib.import_module('tools.v4_mixed_fusion.adapter')
        report.update(production=production,checked_manifest=production['checked_manifest'],baseline_report=evidence,
                      research_manifest=research,research_artifacts=str(args.artifacts.resolve()),source_sha256=sources(research))
        jobs=[(s,lambda s=s:pad_case(extension,s)) for s in pad_specs()]
        jobs += [(s,lambda s=s:module_case(extension,adapter,s)) for s in module_specs()]
        for dtype in DTYPES:
            for label in ('strided','transpose','negative','padding_zero','scale2','pytorch','cpu','fp32','autocast'):
                jobs.append((dict(kind='fallback',name=f'fallback/{dtype}/{label}'),lambda d=dtype,l=label:fallback_case(extension,adapter,l,d)))
            for mask in ((True,False,False),(False,True,False),(False,False,True),(True,True,True)):
                jobs.append((dict(kind='training_fallback',name=f'training/{dtype}/{mask}'),lambda d=dtype,m=mask:training_fallback(extension,adapter,d,m)))
            jobs += [(dict(kind='cache',name='cache/'+dtype),lambda d=dtype:cache_case(extension,adapter,d)),
                     (dict(kind='stream_graph',name='stream_graph/'+dtype),lambda d=dtype:stream_graph_case(extension,adapter,d))]
        jobs.append((dict(kind='invalid_contract',name='invalid_contracts'),lambda:invalid_contracts(extension,adapter)))
        report['expected_cases']=len(jobs)
        for index,(specification,call) in enumerate(jobs):
            try: row=call()
            except Exception as error:
                row=dict(**specification,passed=False,status='case_error',error=dict(type=type(error).__name__,message=str(error)))
                if 'illegal memory access' in str(error).lower() or 'device-side assert' in str(error).lower():
                    report['rows'].append(row); raise
            report['rows'].append(row)
            if not row['passed']: print('FAIL',row['name'],row.get('error','numerical/contract check'),flush=True)
            if (index+1)%25==0: save(); print(f'Checked {index+1}/{len(jobs)}',flush=True)
        if sources(research)!=report['source_sha256']: raise RuntimeError('Measured source dependency changed')
        if loader.load_production_checked()!=production: raise RuntimeError('Production checked identity changed')
        _,verified=loader.load_checked(args.artifacts,block_threads=args.block_threads)
        if verified!=research: raise RuntimeError('Research checked identity changed')
        report.update(status='complete',complete=True,passed=all(r['passed'] for r in report['rows']),
                      failed_cases=[r['name'] for r in report['rows'] if not r['passed']],
                      fused_module_cases=sum(bool(r.get('expected_fused')) for r in report['rows']),
                      verified_fused_module_cases=sum(bool(r.get('expected_fused')) and r.get('route_verified',False) for r in report['rows']))
        report['full_matrix_passed']=report['passed']
        expected_active=sum(s['dtype']=='fp16' and s['padding_mode']=='circular' and s['regime']=='normal' for s in module_specs())
        report['admission_partition']=admission_partition(report['rows'],expected_cases=len(jobs),expected_active=expected_active)
        report['active_domain_admitted']=report['admission_partition']['passed']
    except Exception as error:
        report.update(status='error',passed=False,full_matrix_passed=False,active_domain_admitted=False,complete=False,
                      error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        save(); raise
    save(); print(json.dumps({k:report[k] for k in ('status','passed','full_matrix_passed','active_domain_admitted','admission_partition','expected_cases','fused_module_cases','failed_cases')},indent=2))
    return 0 if report['passed'] else 2


if __name__=='__main__':
    raise SystemExit(main())
