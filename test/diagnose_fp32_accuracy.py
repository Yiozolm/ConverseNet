"""Follow up the accuracy audit with exact stage pairs and FFT ablations."""
import collections
import json
import math

import torch
import torch.nn.functional as F
import accuracy_fp32_versions as audit


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    versions, nearest, ops=audit.load_all()
    data=json.loads((audit.OUT/'results.json').read_text(encoding='utf-8'))
    cases=[r['case'] for r in data['rows'] if r['group']=='forward' and r['version']=='python_fp32']
    pairs=[('pre_spectral_io_v7','checkout_v7'),('checkout_v7','pre_batch_fft'),
           ('pre_batch_fft','nearest_before_c2r'),('nearest_before_c2r','snapshot_before_training'),
           ('snapshot_before_training','snapshot_after_training'),
           ('checkout_v7','nearest_cpp'),('nearest_cpp','nearest_spectral'),
           ('nearest_before_c2r_nearest','snapshot_nearest_before'),
           ('snapshot_nearest_before','snapshot_nearest_after'),('nearest_spectral','snapshot_nearest_after')]
    paired=[]
    with torch.no_grad():
        for case in cases:
            a=audit.arguments(case['shape'],case['scale'],case['seed'],case['kernel'],case['broadcast'],case['prior'],case['bias'])
            s,e=case['scale'],case['eps']
            selected={**versions,**(nearest if case['prior']=='nearest' else {})}
            outputs={name:fn(a,s,e) for name,fn in selected.items() if name in set(sum(([a,b] for a,b in pairs),[]))}
            for first,second in pairs:
                if first in outputs and second in outputs:
                    paired.append(dict(first=first,second=second,case=case,bitwise=torch.equal(outputs[first].view(torch.int32),outputs[second].view(torch.int32)),**audit.metrics(outputs[second],outputs[first])))
            for op in ops:op.clear_cache()
    # Reproduce the largest observed forward error and independently vary FFT
    # precision versus spectral solve precision. These are diagnosis-only paths.
    worst=max((r for r in data['rows'] if r['group']=='forward'),key=lambda r:r['max_abs'])['case']
    ablations=[]
    for eps in (1e-3,1e-5,1e-7):
        for bias_value in (0.,-12.):
            a=audit.arguments(worst['shape'],1,worst['seed'],worst['kernel'],worst['broadcast'],worst['prior'],bias_value)
            x,p,k,b=a
            ref=audit.converse2d_reference(*(t.double() for t in a),1,eps)
            for name in ('python_fp32','checkout_v7'):
                with torch.no_grad():out=versions[name](a,1,eps)
                ablations.append(dict(version=name,eps=eps,bias=bias_value,**audit.metrics(out,ref)))
            for mode in ('rfft32_solve32','rfft32_solve64','kernel_fft64_only','all_fft64_cast32'):
                kh,kw=k.shape[-2:]
                psf=F.pad(k,(0,x.size(-1)-kw,0,x.size(-2)-kh))
                psf=torch.roll(psf,(-(kh//2),-(kw//2)),(-2,-1))
                fk=torch.fft.rfft2(psf.double() if mode in ('kernel_fft64_only','all_fft64_cast32') else psf).to(torch.complex64)
                y=torch.fft.rfft2(x.double() if mode=='all_fft64_cast32' else x).to(torch.complex64)
                prior=torch.fft.rfft2(p.double() if mode=='all_fft64_cast32' else p).to(torch.complex64)
                lam=torch.sigmoid(b-9)+eps
                if mode=='rfft32_solve64':fk,y,prior,lam=fk.to(torch.complex128),y.to(torch.complex128),prior.to(torch.complex128),lam.double()
                spectrum=prior+fk.conj()*(y-fk*prior)/(fk.real.square()+fk.imag.square()+lam)
                out=torch.fft.irfft2(spectrum.to(torch.complex64),s=x.shape[-2:])
                ablations.append(dict(version=mode,eps=eps,bias=bias_value,**audit.metrics(out,ref)))
    groups=collections.defaultdict(list)
    for row in paired:groups[row['first'],row['second']].append(row)
    summary=[dict(first=a,second=b,count=len(rows),bitwise_equal=sum(r['bitwise'] for r in rows),
        max_abs=max(r['max_abs'] for r in rows),max_relative_l2=max(r['relative_l2'] for r in rows)) for (a,b),rows in groups.items()]
    result=dict(pair_summary=summary,pairs=paired,worst_case=worst,ablations=ablations)
    (audit.OUT/'diagnosis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(pair_summary=summary,worst_case=worst,ablations=ablations),indent=2))


if __name__=='__main__':main()
