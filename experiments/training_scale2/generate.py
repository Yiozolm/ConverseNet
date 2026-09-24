from pathlib import Path
import importlib.util,json,hashlib,subprocess,re
from locations import HERE,ROOT
if (HERE/'before_manifest.json').exists():
    raise RuntimeError('Use a new artifacts directory for a new study, or --phase compile to resume a frozen build.')
spec=importlib.util.spec_from_file_location('config',ROOT/'Converse2D/build_config.py')
config=importlib.util.module_from_spec(spec);spec.loader.exec_module(config)
source=config.PACKAGE
before=config.legacy_sources()
frozen=HERE/'before';frozen.mkdir(exist_ok=True)
hashes={}
for name in (*config.dependency_names(), 'converse2d_training.h'):
    path=source/name;dest=frozen/'Converse2D/torch_converse2d'/name
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    hashes[path.relative_to(ROOT).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
for name in ['Converse2D/build_config.py','Converse2D/setup.py','models/converse_core.py','models/util_converse.py','models/converse_usrnet.py']:
    path=ROOT/name;dest=frozen/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
manifest=dict(head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),
              status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),source_sha256=hashes)
(HERE/'before_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
(HERE/'before_sources.json').write_text(json.dumps(before),encoding='utf-8')
math=(source/'training/detail/math.cuh').read_text()
math=math.split('namespace converse2d::training_detail {',1)[1].rsplit('}',1)[0]
generic=(source/'training/training_generic.cu').read_text()
generic=generic[generic.index('namespace {'):]
names=['solve_alias','solve_output','adjoint_q','alias_adjoint','adjoint_inputs','adjoint_filter']
for name in names:
    match=re.search(r'void '+name+r'\([^)]*\)\s*\{',generic)
    assert match,name
    signature=match[0]
    assert 'I s' in signature
    replacement=signature.replace('I s','I runtime_scale')+'\n    constexpr I s = 2;'
    generic=generic[:match.start()]+replacement+generic[match.end():]
generic=generic.replace('launch_training_generic_', 'launch_training_scale2_')
prototypes='\n'.join(line.replace('launch_training_generic_','launch_training_scale2_')
                     for line in (source/'training/launchers.cuh').read_text().splitlines()
                     if line.startswith('void launch_training_generic_'))
new_sources={}
for mode in ('const64','int32'):
    math_mode=math.replace('using I = int64_t;', 'using I = int32_t;') if mode=='int32' else math
    candidate='''#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/complex.h>
namespace converse2d::scale2 {
'''+math_mode+generic+'\n}\n'
    (HERE/f'training_scale2_{mode}.cu').write_text(candidate,encoding='utf-8')
    texts=dict(before)
    text=texts['converse2d_training.cu']
    fwd='    else launch_training_generic_forward(y,p,k,l,out,q,d,H,W,s,stream);'
    bwd='    else launch_training_generic_backward(g,p,k,q,d,r,gd,gp,gk,H,W,s,need_p,need_k,no_broadcast,reduce_filter,stream);'
    gate='s==2' if mode=='const64' else 's==2 && p.numel()<=INT32_MAX-256 && H<=INT32_MAX/2 && W<=INT32_MAX/2'
    assert text.count(fwd)==text.count(bwd)==1
    text=text.replace(fwd,f'    else if({gate}) converse2d::scale2::launch_training_scale2_forward(y,p,k,l,out,q,d,H,W,s,stream);\n'+fwd)
    text=text.replace(bwd,f'    else if({gate}) converse2d::scale2::launch_training_scale2_backward(g,p,k,q,d,r,gd,gp,gk,H,W,s,need_p,need_k,no_broadcast,reduce_filter,stream);\n'+bwd)
    text='#include <ATen/ATen.h>\n#include <cuda_runtime.h>\n#include <cstdint>\nnamespace converse2d::scale2 {\n'+prototypes+'\n}\n'+text+'\n'+candidate
    texts['converse2d_training.cu']=text
    new_sources[mode]=texts
new_sources['before']=before
candidate=ROOT/'Converse2D/torch_converse2d/training/training_scale2.cu'
clean=candidate.read_text(encoding='utf-8')
(HERE/'candidate.cu').write_bytes(candidate.read_bytes())
manifest['candidate_source']=dict(path=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
(HERE/'before_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
fused=dict(new_sources['int32'])
old=(HERE/'training_scale2_int32.cu').read_text(encoding='utf-8')
assert fused['converse2d_training.cu'].count(old)==1
fused['converse2d_training.cu']=fused['converse2d_training.cu'].replace(old,clean)
new_sources['fused']=fused

(HERE/'variants.json').write_text(json.dumps(new_sources),encoding='utf-8')
print('Frozen current source and generated constant-scale / int32 variants.')
