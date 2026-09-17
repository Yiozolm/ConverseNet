"""Skill-derived timeline, with uint32 wrap handling and exact tile dependencies.

Unlike nearest-in-time arrow matching, slot return i releases loader tile i+2.
Wait bars mean elapsed time inside a wait region, including instrumentation and
barrier overhead, not an estimate of recoverable hardware stall cycles.
"""
from pathlib import Path
import json
import statistics

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"artifacts/warp_spectral"


def decode(path):
    raw=np.load(path)["prof"].view(np.uint32).astype(np.int64)
    mask=raw!=0
    anchor=int(raw[0,0,0])
    # Signed modular distance, valid for this single-CTA trace shorter than 2^31.
    relative=(raw-anchor+(1<<31))%(1<<32)-(1<<31)
    relative-=relative[mask].min()
    relative[~mask]=-1
    return relative


def spans(a,role,b,e):
    return [(int(a[role,b,i]),int(a[role,e,i])) for i in range(64)
            if a[role,b,i]>=0 and a[role,e,i]>=a[role,b,i]]


def analyze(a):
    total=int(a.max())
    work=[spans(a,r,2,3) for r in range(2)]
    wait=[spans(a,r,0,1) for r in range(2)]
    overlap=sum(max(0,min(y,v)-max(x,u)) for x,y in work[0] for u,v in work[1])
    iterations=int(np.sum(a[0,0]>=0))
    return dict(total_cycles=total,iterations=iterations,
        wait_pct_total=[100*sum(e-b for b,e in s)/total for s in wait],
        work_cycles=[sum(e-b for b,e in s) for s in work],
        overlap_cycles=overlap,
        overlap_pct_loader_work=100*overlap/sum(e-b for b,e in work[0]),
        # Exclude fill/drain; interval between work starts, not work duration.
        steady_cycles_per_iteration=statistics.median(np.diff(a[1,2,2:iterations-2]).tolist()),
        median_work_cycles=[statistics.median([e-b for b,e in s[2:-2]]) for s in work],
        publication_marker_reorder_count=int(sum(a[0,4,i]>a[1,1,i] for i in range(iterations))))


def main():
    paths=sorted(OUT.glob("trace_*.npz"))
    traces=[decode(p) for p in paths]
    stats=[analyze(a) for a in traces]
    median=statistics.median(s["total_cycles"] for s in stats)
    chosen=min(range(len(stats)),key=lambda i:abs(stats[i]["total_cycles"]-median))
    a=traces[chosen]; total=stats[chosen]["total_cycles"]
    fig,axes=plt.subplots(2,1,figsize=(13,6.6))
    colors=["#2674b5","#d88720"]
    for panel,ax in enumerate(axes):
        for role in range(2):
            y=1-role
            stamps=a[role][a[role]>=0]
            lo,hi=int(stamps.min()),int(stamps.max())
            ax.broken_barh([(lo,hi-lo)],(y-.24,.48),facecolors="#eeeeee")
            ax.broken_barh([(b,e-b) for b,e in spans(a,role,2,3)],(y-.24,.48),facecolors=colors[role])
            ax.broken_barh([(b,e-b) for b,e in spans(a,role,0,1)],(y-.24,.48),
                          facecolors="#bcbcbc",edgecolors="#777777",hatch="////",linewidth=.4)
        if panel:
            for i in range(stats[chosen]["iterations"]):
                if a[0,4,i]>=0 and a[1,1,i]>=0:
                    note=ax.annotate("",xy=(a[1,1,i],.25),xytext=(a[0,4,i],.75),
                                     arrowprops=dict(arrowstyle="->",color="#2674b5",lw=.9))
                    note.arrow_patch.set_clip_path(ax.patch)
                if i+2<stats[chosen]["iterations"]:
                    note=ax.annotate("",xy=(a[0,1,i+2],.75),xytext=(a[1,5,i],.25),
                                     arrowprops=dict(arrowstyle="->",color="#d88720",lw=.9))
                    note.arrow_patch.set_clip_path(ax.patch)
            ax.set_xlim(total*.3,total*.65)
        else:
            ax.set_xlim(0,total)
        ax.set_yticks([1,0],labels=["Warp 0: load","Warp 1: compute / store"])
        ax.set_ylim(-.5,1.5)
        ax.set_xlabel("SM clock cycles relative to first recorded event")
        ax.grid(axis="x",alpha=.25)
        ax.set_title("Single CTA, 16 tiles, double buffer" if not panel else
                     "Steady-state zoom: blue = filled tile i; orange = reuse slot for tile i+2")
    fig.suptitle("Converse2D scale=2 — measured warp-specialized prototype",fontsize=14)
    fig.legend(handles=[Patch(facecolor=colors[0],label="Load work region"),
                        Patch(facecolor=colors[1],label="Compute / store region"),
                        Patch(facecolor="#bcbcbc",hatch="////",label="Measured wait region"),
                        Patch(facecolor="#eeeeee",label="Untracked (not classified as wait)")],
               loc="lower center",ncol=4,fontsize=9)
    fig.tight_layout(rect=(0,.055,1,.95))
    fig.savefig(OUT/"timeline.png",dpi=160)
    result=dict(traces=[dict(file=p.name,**s) for p,s in zip(paths,stats)],
                plotted_trace=paths[chosen].name,
                span_cv_pct=100*statistics.pstdev(s["total_cycles"] for s in stats)/statistics.mean(s["total_cycles"] for s in stats),
                interpretation="Instrumented intervals, not instruction throughput or recoverable stalls; post-arrive markers can lag actual publication.")
    (OUT/"timeline_stats.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2))


if __name__=="__main__": main()
