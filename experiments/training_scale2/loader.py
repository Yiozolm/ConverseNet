from pathlib import Path
import json,hashlib,os
import torch
from torch.utils import cpp_extension
from locations import HERE,ROOT
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load_all(warm=False):
    if warm:
        manifest=json.loads((HERE/'builds.json').read_text())
        assert set(manifest)==set(json.loads((HERE/'variants.json').read_text())), 'Incomplete variant build manifest'
        result={}
        for mode,row in manifest.items():
            assert row['torch']==str(torch.__version__) and row['cuda']==torch.version.cuda
            assert sha(Path(row['library']))==row['binary_sha256']
            for name,digest in row['files'].items():assert sha(Path(row['directory'])/name)==digest
            torch.ops.load_library(row['library']);result[mode]=getattr(torch.ops,row['namespace'])
        return result
    cpp_extension.SUBPROCESS_DECODE_ARGS=('utf-8','replace')
    if 'TORCH_CUDA_ARCH_LIST' not in os.environ:
        major,minor=torch.cuda.get_device_capability()
        os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    variants=json.loads((HERE/'variants.json').read_text())
    records={};result={}
    for mode,texts in variants.items():
        fingerprint=hashlib.sha256(json.dumps(dict(texts=texts,torch=str(torch.__version__),cuda=torch.version.cuda,arch=os.environ.get('TORCH_CUDA_ARCH_LIST'),platform=os.name),sort_keys=True).encode()).hexdigest()[:12]
        prefix=f's2_{mode}_{fingerprint}_converse'
        namespace=prefix+'2d';name=f's2_{mode}_{fingerprint}'
        folder=HERE/'build'/name;folder.mkdir(parents=True,exist_ok=True)
        sources=[];files={}
        for old,content in texts.items():
            filename=old.replace('converse',prefix)
            dest=folder/filename;value=content.replace('converse',prefix)
            if not dest.exists() or dest.read_text()!=value:dest.write_text(value,encoding='utf-8')
            files[filename]=sha(dest)
            if dest.suffix in ('.cpp','.cu'):sources.append(str(dest))
        flags=(["/O2","/std:c++17"] if os.name=='nt' else ["-O3","-std=c++17"])+['-DCONVERSE2D_WITH_CUDA=1']
        cpp_extension.load(name=name,sources=sources,extra_cflags=flags,
            extra_cuda_cflags=['-O3','-lineinfo'],with_cuda=True,is_python_module=False,
            build_directory=str(folder),verbose=True)
        lib=folder/(name+('.pyd' if os.name=='nt' else '.so'))
        records[mode]=dict(directory=str(folder),namespace=namespace,library=str(lib),binary_sha256=sha(lib),
                           files=files,cxx_flags=flags,cuda_flags=['-O3','-lineinfo'],torch=str(torch.__version__),cuda=torch.version.cuda,arch=os.environ.get('TORCH_CUDA_ARCH_LIST'))
        result[mode]=getattr(torch.ops,namespace)
        (HERE/'builds.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    return result
if __name__=='__main__':load_all()
