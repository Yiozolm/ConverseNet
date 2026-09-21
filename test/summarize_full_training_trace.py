"""CPU-only kernel attribution for a full-USRNet Torch Profiler Chrome trace.

The sole denominator is original ph=X, cat=kernel events. GPU annotations,
copies and memset events never enter kernel totals. GPU launches are linked
through External id to CPU records, then through CPU ancestors; backward
records use verified fwdbwd sequence links or explicitly labelled fallbacks.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
EPS=.011  # microseconds; tolerate serialization rounding of nested endpoints


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def seq(event):return event.get('args',{}).get('Sequence number')
def end(event):return event['ts']+event.get('dur',0)


def group_rows(records,key,total):
    sums=defaultdict(lambda:[0,0.])
    for record in records:
        label=record[key]
        sums[label][0]+=1;sums[label][1]+=record['duration_us']
    return [{'name':name,'kernels':values[0],'duration_us':values[1],'kernel_sum_pct':100*values[1]/total}
            for name,values in sorted(sums.items(),key=lambda item:-item[1][1])]


def module_type(path):
    if path=='d':return 'ConverseSolver/DataNet'
    if re.fullmatch(r'p\.m_body\.\d+\.conv1\.3',path):return 'ConverseSolver/Prior'
    if re.fullmatch(r'p\.m_body\.\d+\.conv[12]\.0',path):return 'LayerNorm'
    if re.fullmatch(r'p\.m_body\.\d+\.(conv1\.[15]|conv2\.[13])',path):return 'Conv2d'
    if path in ('conv1','conv2') or re.fullmatch(r'convs\.\d+',path):return 'Conv2d'
    if re.fullmatch(r'p\.m_body\.\d+\.(conv1\.[24]|conv2\.2)',path) or path=='kernelnet.gelu':return 'GELU'
    if re.fullmatch(r'kernelnet\.fc[123]',path):return 'Linear'
    if re.fullmatch(r'p\.m_body\.\d+',path):return 'ResidualGateAndAdd'
    if path=='kernelnet':return 'KernelNetLayout'
    return 'OtherModule' if path else 'UnattributedModule'


def family(name):
    for label in ('solve_alias','solve_output','adjoint_q','adjoint_inputs','adjoint_filter',
                  'forward_scale1','backward_scale1','filter_scale1'):
        if label in name:return 'spectral/'+label
    lowered=name.lower()
    if any(token in lowered for token in ('regular_fft','fft2d','fft3d','fft_device','cufft')):return 'FFT'
    if '_fft_' in lowered:return 'FFT_support'
    if 'wgrad' in lowered:return 'convolution_weight_gradient'
    if 'dgrad' in lowered:return 'convolution_input_gradient'
    if 'convol' in lowered:return 'convolution_other'
    if 'gemm' in lowered or 'gemv' in lowered:return 'matrix_multiply'
    if 'gelu' in lowered:return 'GELU'
    if 'reduce' in lowered:return 'reduction'
    if 'copy' in lowered or 'cast' in lowered:return 'copy_or_cast'
    if 'fill' in lowered:return 'fill'
    return 'other_elementwise_or_kernel'


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--trace',type=Path,default=ROOT/'artifacts/training_research/torch_current/full.trace.json')
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    trace=args.trace.resolve();meta_path=(args.metadata or trace.with_name('metadata.json')).resolve()
    output=(args.output or trace.with_name('attribution.json')).resolve()
    if output in (trace,meta_path):parser.error('Output must not overwrite the source trace or metadata')
    source_sha=sha(trace)
    document=json.loads(trace.read_text(encoding='utf-8-sig'));events=document['traceEvents']
    metadata=json.loads(meta_path.read_text(encoding='utf-8-sig'))
    cpu={i:e for i,e in enumerate(events) if e.get('ph')=='X' and e.get('cat') in ('cpu_op','user_annotation')}
    threads=defaultdict(list);external=defaultdict(list);at_start=defaultdict(list)
    for index,event in cpu.items():
        threads[(event['pid'],event['tid'])].append(index)
        if event.get('cat')=='cpu_op':
            external[event.get('args',{}).get('External id')].append(index)
            at_start[(event['pid'],event['tid'],round(event['ts'],3))].append(index)
    parents={};non_nested=[]
    for indices in threads.values():
        stack=[]
        for index in sorted(indices,key=lambda i:(cpu[i]['ts'],-end(cpu[i]),i)):
            event=cpu[index]
            while stack and (end(cpu[stack[-1]])<event['ts']-EPS or end(cpu[stack[-1]])<end(event)-EPS):
                if end(cpu[stack[-1]])>event['ts']+EPS:non_nested.append([stack[-1],index])
                stack.pop()
            parents[index]=stack[-1] if stack else None
            stack.append(index)
    ancestor_cache={}
    def ancestors(index):
        if index not in ancestor_cache:
            parent=parents[index]
            ancestor_cache[index]=(index,)+(ancestors(parent) if parent is not None else ())
        return ancestor_cache[index]
    steps=[i for i,e in cpu.items() if e['name'].startswith('NsightStep/')]
    phases=[i for i,e in cpu.items() if e['name'].startswith('Phase/')]
    def enclosing_time(index,choices):
        event=cpu[index]
        matches=[i for i in choices if cpu[i]['pid']==event['pid'] and cpu[i]['ts']<=event['ts']+EPS and end(cpu[i])+EPS>=end(event)]
        return min(matches,key=lambda i:cpu[i]['dur']) if matches else None
    contexts={}
    for index,event in cpu.items():
        chain=ancestors(index)
        scope=next((i for i in chain if cpu[i]['name'].startswith(('Module/','Solver/'))),None)
        label=cpu[scope]['name'] if scope is not None else ''
        path=label.split('/',3)[3] if label.startswith('Solver/') else label.split('/',2)[2] if label else ''
        step=next((i for i in chain if cpu[i]['name'].startswith('NsightStep/')),None)
        if step is None:step=enclosing_time(index,steps)
        phase=next((i for i in chain if cpu[i]['name'].startswith('Phase/')),None)
        if phase is None:phase=enclosing_time(index,phases)
        backward=next((i for i in chain if cpu[i]['name'].startswith('autograd::engine::evaluate_function:')),None)
        contexts[index]={'scope':label,'module':path,'module_type':module_type(path),
            'solver_scope':next((cpu[i]['name'] for i in chain if cpu[i]['name'].startswith('Solver/')),''),
            'step':int(cpu[step]['name'].split('/')[1]) if step is not None else -1,
            'phase':cpu[phase]['name'].split('/',1)[1] if phase is not None else 'unattributed_phase',
            'kernel_preparation':any(cpu[i]['name']=='converse2d::prepare_training_kernel' for i in chain),
            'circular_padding':any(cpu[i]['name']=='aten::_pad_circular' for i in chain),
            'inside_public_solve':any(cpu[i]['name']=='converse2d::forward' for i in chain),
            'backward_owner':backward}
    # Explicit trace flow endpoints identify forward records, validated by the
    # matching Sequence number. This avoids treating every repeated seq tag as
    # a unique differentiable operation (data setup also carries seq tags).
    flows=defaultdict(dict)
    for event in events:
        if event.get('cat')=='fwdbwd':flows[(event['pid'],event['id'])][event['ph']]=event
    seq_flow=defaultdict(set);flow_checks=Counter()
    for pair in flows.values():
        if 's' not in pair or 'f' not in pair:flow_checks['incomplete']+=1;continue
        first,last=pair['s'],pair['f']
        source=at_start.get((first['pid'],first['tid'],round(first['ts'],3)),[])
        target=at_start.get((last['pid'],last['tid'],round(last['ts'],3)),[])
        matches=[(a,b) for a in source for b in target if seq(cpu[a]) is not None and seq(cpu[a])==seq(cpu[b])]
        if not matches:flow_checks['no_matching_sequence']+=1;continue
        flow_checks['verified_pairs']+=1
        for a,b in matches:seq_flow[(cpu[b]['pid'],seq(cpu[b]))].add(a)
    seq_candidates=defaultdict(list);solver_bounds={}
    for index,event in cpu.items():
        value=seq(event)
        if event['cat']=='cpu_op' and value is not None and contexts[index]['backward_owner'] is None:
            seq_candidates[(event['pid'],value)].append(index)
            solver=contexts[index]['solver_scope']
            if solver:
                key=(contexts[index]['step'],solver)
                bounds=solver_bounds.setdefault(key,[value,value])
                bounds[0]=min(bounds[0],value);bounds[1]=max(bounds[1],value)
    def signature(index):
        c=contexts[index]
        return (c['step'],c['scope'],c['solver_scope'],c['kernel_preparation'])
    mapped_cache={}
    def backward_context(owner):
        if owner in mapped_cache:return mapped_cache[owner]
        event=cpu[owner];key=(event['pid'],seq(event));candidate=seq_flow.get(key,set())
        method='verified_fwdbwd_sequence'
        if not candidate:
            candidate=seq_candidates.get(key,[]);method='sequence_context_consensus'
        signatures={signature(i) for i in candidate}
        if len(signatures)==1:
            selected=min(candidate,key=lambda i:cpu[i]['dur'])
            result=(contexts[selected],method,selected)
        elif 'SpectralSolve' in event['name'] and seq(event) is not None:
            matches=[key for key,(low,high) in solver_bounds.items()
                     if key[0]==contexts[owner]['step'] and low<=seq(event)<=high]
            if len(matches)==1:
                step,label=matches[0]
                scope=next(i for i,e in cpu.items() if e['name']==label and contexts[i]['step']==step)
                result=(contexts[scope],'unique_solver_sequence_interval_inference',scope)
            else:result=(None,'unattributed_backward_sequence',None)
        else:result=(None,'unattributed_backward_sequence',None)
        mapped_cache[owner]=result;return result
    records=[];external_errors=[]
    for index,event in enumerate(events):
        if event.get('cat')!='kernel' or event.get('ph')!='X':continue
        candidates=external.get(event.get('args',{}).get('External id'),[])
        owner=candidates[0] if len(candidates)==1 else None
        if owner is None:external_errors.append(index)
        context=contexts[owner] if owner is not None else None
        method='external_id_cpu_ancestors' if owner is not None else 'unattributed_external_id'
        direction='forward';forward_origin=owner
        if context is not None and context['backward_owner'] is not None:
            direction='backward'
            mapped,method,forward_origin=backward_context(context['backward_owner'])
            origin=mapped
        else:origin=context
        phase=context['phase'] if context is not None else 'unattributed_phase'
        path=origin['module'] if origin else ''
        kind=origin['module_type'] if origin else 'UnattributedModule'
        group=family(event['name'])
        stage=direction if phase=='model_forward_backward' and (direction=='backward' or path) else ('loss_and_backward_setup' if phase=='model_forward_backward' else phase)
        if stage not in ('forward','backward'):kind='TrainingLoop/'+stage
        if group.startswith('spectral/') and kind=='UnattributedModule':kind='ConverseSolver/unknown_occurrence'
        preparation=bool(origin and origin['kernel_preparation'])
        component=('kernel_preparation' if preparation else 'spectral_solve' if group.startswith('spectral/')
                   else 'activation_fft' if kind.startswith('ConverseSolver') and group in ('FFT','FFT_support')
                   else 'solver_support' if kind.startswith('ConverseSolver') else kind)
        records.append({'trace_event_index':index,'name':event['name'],'duration_us':event['dur'],
                        'timestamp_us':event['ts'],'device':event.get('args',{}).get('device'),
                        'stream':event.get('args',{}).get('stream'),'cpu_event_index':owner,
                        'cpu_op':cpu[owner]['name'] if owner is not None else None,
                        'backward_cpu_owner':cpu[context['backward_owner']]['name']
                            if context and context['backward_owner'] is not None else None,
                        'forward_origin_event_index':forward_origin,'mapping':method,
                        'forward_origin_cpu_op':cpu[forward_origin]['name'] if forward_origin is not None else None,
                        'step':context['step'] if context else -1,'stage':stage,
                        'module':path or 'unattributed','module_type':kind,'component':component,
                        'solver_scope':origin['solver_scope'] if origin else '',
                        'forward_circular_padding':bool(origin and origin['circular_padding']),
                        'slice_origin':('circular_padding' if origin and origin['circular_padding'] else
                            'kernel_preparation' if preparation else
                            'post_solve_crop' if origin and origin['solver_scope'] and not origin['inside_public_solve']
                                and forward_origin is not None and cpu[forward_origin]['name']=='aten::slice' else 'other'),
                        'family':group,'kernel_preparation':preparation})
    total=sum(r['duration_us'] for r in records)
    partitions={key:group_rows(records,key,total) for key in ('stage','module_type','component','family','mapping')}
    for rows in partitions.values():
        assert sum(r['kernels'] for r in rows)==len(records)
        assert math.isclose(sum(r['duration_us'] for r in rows),total,abs_tol=1e-5)
    anchors=[r for r in records if r['family']=='spectral/solve_alias']
    terminals=[r for r in records if r['family']=='spectral/solve_output']
    call_map=metadata['call_map'];by_call={(r['step'],r['label']):r for r in call_map}
    anchor_counts=Counter((r['step'],r['solver_scope']) for r in anchors)
    terminal_counts=Counter((r['step'],r['solver_scope']) for r in terminals)
    anchor_ok=(len(anchors)==len(terminals)==len(call_map) and set(anchor_counts)==set(by_call)
               and set(terminal_counts)==set(by_call) and all(v==1 for v in anchor_counts.values())
               and all(v==1 for v in terminal_counts.values()))
    anchor_order_ok=([(r['step'],r['solver_scope']) for r in sorted(anchors,key=lambda r:r['timestamp_us'])]
                     ==[(r['step'],r['label']) for r in sorted(call_map,key=lambda r:(r['step'],r['index']))])
    anchor_ok=anchor_ok and anchor_order_ok
    anchor_evidence=[{'step':r['step'],'solver':r['solver_scope'],'trace_event_index':r['trace_event_index'],
                     'cpu_event_index':r['cpu_event_index'],'gpu_ts_us':r['timestamp_us'],
                     'gpu_pid':events[r['trace_event_index']]['pid'],'gpu_tid':events[r['trace_event_index']]['tid'],
                     'external_id':events[r['trace_event_index']]['args'].get('External id'),
                     'call_map':by_call.get((r['step'],r['solver_scope']))} for r in sorted(anchors,key=lambda r:r['timestamp_us'])]
    runtime=defaultdict(lambda:[0,0.])
    for e in events:
        if e.get('cat')=='cuda_runtime' and e.get('ph')=='X':runtime[e['name']][0]+=1;runtime[e['name']][1]+=e.get('dur',0)
    device_union={}
    for device in sorted({r['device'] for r in records}):
        device_records=[r for r in records if r['device']==device]
        base=min(r['timestamp_us'] for r in device_records)
        # Subtract the large epoch timestamp before adding short durations.
        # Otherwise float cancellation can make a non-overlapping union exceed
        # the exact sum of stored duration fields by sub-microsecond amounts.
        intervals=sorted((r['timestamp_us']-base,r['timestamp_us']-base+r['duration_us']) for r in device_records)
        merged=[]
        for start,finish in intervals:
            if merged and start<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],finish)
            else:merged.append([start,finish])
        occupied=sum(b-a for a,b in merged);window=max(b for _,b in intervals)-intervals[0][0]
        device_union[device]={'kernel_union_us':occupied,'kernel_bounding_window_us':window,
                              'non_kernel_time_in_window_us':window-occupied,
                              'warning':'Non-kernel time includes copies, synchronization and submission gaps; not exclusively CPU idle.'}
    assert sha(trace)==source_sha,'Source trace changed while being analyzed'
    summary={'trace':str(trace),'trace_sha256':source_sha,'metadata':str(meta_path),'metadata_sha256':sha(meta_path),
        'source_sha256':metadata.get('source_sha256'),
        'denominator':{'category':'kernel','phase':'X','events':len(records),'duration_us':total,
                       'excluded_categories':dict(Counter(e.get('cat','') for e in events if e.get('cat')!='kernel')),
                       'meaning':'Sum of actual kernel event durations; not wall time and not exclusive of cross-stream overlap.'},
        'partitions':partitions,'by_step':{step:group_rows([r for r in records if r['step']==step],'component',total)
                                        for step in sorted({r['step'] for r in records})},
        'by_step_percentage_denominator':'Whole-trace cat=kernel sum, so per-step rows need not sum to 100%.',
        'module_stage':group_rows([{**r,'module_stage':r['stage']+'/'+r['module']} for r in records],'module_stage',total),
        'component_cpu_owners':group_rows([{**r,'component_op':r['component']+'/'+r['stage']+'/'+str(r['cpu_op'])}
                                          for r in records],'component_op',total),
        'top_cpu_owners':group_rows([{**r,'op':r['stage']+'/'+str(r['cpu_op'])} for r in records],'op',total)[:45],
        'validation':{'partitions_cover_all_kernel_events_once':True,'ambiguous_or_missing_external_ids':len(external_errors),
                      'non_nested_cpu_interval_pairs':len(non_nested),'verified_fwdbwd_flows':dict(flow_checks),
                      'anchor_mapping_valid':anchor_ok,'solve_alias_count':len(anchors),'solve_output_count':len(terminals),
                      'anchor_timestamp_order_matches_call_map':anchor_order_ok,
                      'call_map_count':len(call_map),'per_step_alias':dict(Counter(r['step'] for r in anchors))},
        'unattributed':{'external_event_indices':external_errors,
                         'backward_sequence_kernel_events':sum(r['mapping']=='unattributed_backward_sequence' for r in records)},
        'layer_guide_evidence':{'unit':'Chronological solver invocation, not unique model parameter or transformer layer',
             'num_layers_per_forward':40,'passes':metadata['config']['steps'],'anchor_offset':0,
             'anchor_regex':'.*solve_alias.*','end_anchor_regex':'.*solve_output.*',
             'limitation':'Anchors bracket the solve stage, not complete module latency; work before solve_alias (including FFT/prep) lies outside each guide.',
             'verified':anchor_ok,'anchors':anchor_evidence},
        'cuda_runtime': [{'name':n,'calls':v[0],'host_duration_us':v[1]} for n,v in sorted(runtime.items(),key=lambda p:-p[1][1])],
        'gpu_kernel_intervals':device_union,
        'limitations':['CPU ancestors establish launch provenance, not GPU-time containment.',
           'CppNode SpectralSolve has no directly recorded forward Sequence event; unique enclosing solver Sequence intervals are explicitly marked as inference.',
           'Unmapped backward nodes (notably CopySlices) remain unattributed, not reassigned by kernel appearance.',
           'Profiler capture uses two complete synchronized audit steps after warmup/reset, not full 250-step time or convergence proof.'],
        'kernel_attribution':records}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# Full-training kernel attribution','',f'Only original `cat=kernel, ph=X`: **{len(records):,} kernels / {total/1000:.3f} ms** over {metadata["config"]["steps"]} captured steps. Profiling time is diagnostic only.','']
    for key in ('stage','module_type','component','family','mapping'):
        lines += [f'## {key}','','| Partition | Kernels | Kernel sum ms | Kernel sum % |','|---|---:|---:|---:|']
        lines += [f'| {r["name"]} | {r["kernels"]} | {r["duration_us"]/1000:.3f} | {r["kernel_sum_pct"]:.3f} |' for r in partitions[key]]
        lines += ['']
    lines += ['## Validation','',json.dumps(summary['validation'],ensure_ascii=False),'',
              'External id selects a CPU operator. Forward module ownership comes from its innermost CPU scope. Backward uses fwdbwd endpoints with matching Sequence numbers; explicitly labelled consensus/unique-solver-interval fallbacks never masquerade as direct flow links.',
              '', 'Solver guides are L0..L39 for each forward: DataNet at L0/L8/L16/L24/L32; seven prior solvers follow each. They are anchor-to-anchor navigation aids, not total layer latency. Source trace remains unchanged.',
              '', 'Unattributed backward nodes, including CopySlices, remain visible. GPU annotation, memcpy, memset and runtime durations are excluded from every kernel denominator.',
              '', 'The module labels for `convs.1`..`convs.4` inherit the previous DataNet iteration in the capture hook; use their module paths for semantics rather than interpreting that I-label as ownership.',
              '', f'Full provenance and per-kernel records: `{output.name}`.']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'denominator':summary['denominator'],'validation':summary['validation'],
                      'module_type':partitions['module_type'],'component':partitions['component']},indent=2))
    print('Saved',output)
    if not anchor_ok:raise SystemExit('Anchor validation failed; do not publish layer guides.')


if __name__=='__main__':main()
