"""Repeat hot-cache timing with graphs; retain noisy eager measurements too."""
import json
import statistics
import time

import torch
from extension import NAMES, load_all
from study import OUT, make, call, timed


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ops,_=load_all()
    configs=[((1,8,32,40),1,'signed'),((1,32,128,128),1,'box'),((1,64,128,128),2,'normalized'),
             ((1,8,127,129),1,'box'),((8,16,64,64),3,'signed'),((2,3,64,80),2,'gaussian')]
    rows=[]
    for shape,s,kind in configs:
        case=dict(shape=shape,scale=s,seed=211,kernel=kind,broadcast=(False,True),prior='nearest',bias=0.)
        a=make(case);graphs={};held={};batch=16
        with torch.no_grad():
            for name,op in ops.items():
                op.clear_cache();op.precision_reset()
                stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
                op.begin_graph_cache()
                with torch.cuda.stream(stream):
                    for _ in range(5):call(op,a,s,1e-5)
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph,stream=stream):
                        outputs=[call(op,a,s,1e-5) for _ in range(batch)]
                finally:owners=op.end_graph_cache()
                graphs[name]=graph;held[name]=(outputs,owners)
                for _ in range(5):graph.replay()
                torch.cuda.synchronize()
            samples={name:[] for name in NAMES}
            for rr in range(15):
                for name in NAMES[rr%3:]+NAMES[:rr%3]:
                    val=timed(graphs[name].replay,20)
                    samples[name].append({k:v/batch for k,v in val.items()})
            for name in NAMES:
                rows.append(dict(case=case,variant=name,mode='hot_graph',event_ms=statistics.median(r['event_ms'] for r in samples[name]),
                    wall_ms=statistics.median(r['wall_ms'] for r in samples[name]),samples=samples[name],
                    stats=ops[name].precision_stats(),calls_per_graph=batch))
                graphs[name].reset()
            del graphs,held,outputs,owners
            # Per-call synchronization bounds host queueing. FFT plans are warm;
            # each cold sample clears only the kernel-spectrum cache before timing.
            for name,op in ops.items():
                for _ in range(5):call(op,a,s,1e-5)
            cold={n:[] for n in NAMES}
            warm={n:[] for n in NAMES}
            for rr in range(15):
                for name in NAMES[rr%3:]+NAMES[:rr%3]:
                    op=ops[name]
                    c=[];w=[]
                    for _ in range(5):
                        op.clear_cache()
                        c.append(timed(lambda:call(op,a,s,1e-5),1))
                        w.append(timed(lambda:call(op,a,s,1e-5),1))
                    cold[name].append({k:statistics.median(t[k] for t in c) for k in ('event_ms','wall_ms')})
                    warm[name].append({k:statistics.median(t[k] for t in w) for k in ('event_ms','wall_ms')})
            for label,samples in (('cold_sync',cold),('warm_sync',warm)):
                for name in NAMES:
                    rows.append(dict(case=case,variant=name,mode=label,samples=samples[name],
                        event_ms=statistics.median(r['event_ms'] for r in samples[name]),
                        wall_ms=statistics.median(r['wall_ms'] for r in samples[name])))
        (OUT/'timing_repeat.json').write_text(json.dumps(dict(rows=rows,rounds=15,notes='WDDM/shared GPU; compare paired rounds, not absolute latency across runs'),indent=2),encoding='utf-8')
        print('REPEAT',shape,s,flush=True)
    print('DONE',flush=True)


if __name__=='__main__':main()
