"""Checked, isolated loader for a future inference-only pad-and-upcast study.

No CUDA implementation is supplied or selected by this file. Intended binding:
``pad_cast(x: FP16|BF16 CUDA, padding: int, mode: str) -> FP32 padded real``.
The remainder of Converse2D stays in its existing FP32/complex64 solver.

After baseline profiling admits an experiment, provide bindings.cpp and
pad_cast.cu (and any local headers) beside this loader. Build into a fresh path
under .build/mixed_fusion, with TORCH_CUDA_ARCH_LIST=12.0. The conservative
initial block-size alternatives are 128, 256 and 512; no performance choice is
made here. A process may load production and one research variant together.

Import and local source-closure inspection require only the standard library.
Torch/CUDA/toolchain loading happens only through explicit build/load calls.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD_ROOT = ROOT / '.build/mixed_fusion'
ENTRYPOINTS = ('bindings.cpp', 'pad_cast.cu')
BLOCK_CONFIGURATIONS = (128, 256, 512)
RESEARCH_ARCH = '12.0'
_INCLUDE_LINE = re.compile(r'^\s*#\s*include\s+(.+?)\s*$', re.M)
_LITERAL_INCLUDE = re.compile(r'^["<]([^">]+)[">]')
_loaded = None
_loaded_key = None
_loaded_manifest = None
_production_loader = None


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_closure(source_root=HERE, entrypoints=ENTRYPOINTS):
    """Hash the complete conservative local include closure, retaining paths.

    Quoted includes must resolve within the research directory. Angle includes
    are treated as local when a corresponding local file exists; otherwise
    their names are retained as external toolchain/Torch dependencies. Macro
    includes are rejected because their closure cannot be verified statically.
    """
    source_root = Path(source_root).resolve()
    found, external = set(), set()
    def visit(path):
        path = path.resolve()
        if not path.is_relative_to(source_root):
            raise ValueError('Local include escapes isolated research sources: ' + str(path))
        if path in found:
            return
        if not path.is_file():
            raise FileNotFoundError('Missing research source/header: ' + str(path))
        found.add(path)
        for directive in _INCLUDE_LINE.findall(path.read_text(encoding='utf-8')):
            match = _LITERAL_INCLUDE.match(directive)
            if match is None:
                raise ValueError('Macro/nonliteral includes cannot be checked: ' + str(path))
            target = match.group(1)
            candidates = ((path.parent / target).resolve(), (source_root / target).resolve())
            local = next((candidate for candidate in candidates if candidate.is_file()), None)
            if local is not None:
                visit(local)
            elif directive.startswith('"'):
                raise FileNotFoundError('Unresolved quoted include ' + target + ' in ' + str(path))
            else:
                external.add(target)
    for name in entrypoints:
        visit(source_root / name)
    hashes = {path.relative_to(source_root).as_posix(): sha256(path) for path in sorted(found)}
    return dict(local_sources=hashes, external_include_names=sorted(external),
                closure_policy='All reachable local quoted/angle includes, including inactive preprocessor branches')


def _build_config():
    path = ROOT / 'Converse2D/build_config.py'
    spec = importlib.util.spec_from_file_location('mixed_pad_cast_build_config', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _check_precision_flags(cxx_flags, cuda_flags):
    inputs = [*cxx_flags, *cuda_flags, *(os.environ.get(key, '') for key in
              ('CL', '_CL_', 'CXXFLAGS', 'CFLAGS', 'CUDAFLAGS', 'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS'))]
    joined = ' '.join(inputs).lower()
    forbidden = (r'--use_fast_math', r'-ffast-math', r'-ofast', r'/fp:fast',
                 r'-ffp-contract=fast', r'--fmad(?:=|\s+)true', r'--ftz(?:=|\s+)true',
                 r'--prec-div(?:=|\s+)false', r'--prec-sqrt(?:=|\s+)false')
    if any(re.search(pattern, joined) for pattern in forbidden):
        raise RuntimeError('Fast/relaxed floating-point compiler options are forbidden')


def _compiler_details(toolchain):
    details = {}
    for name, info in toolchain.items():
        if name == 'environment':
            continue
        if info is None:
            raise RuntimeError('Compiler is unavailable in the current build shell: ' + name)
        command = [info['path'], '--version' if name == 'nvcc' else '/Bv' if os.name == 'nt' else '--version']
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', timeout=30, check=False)
        details[name] = dict(**info, sha256=sha256(info['path']), version_command=command,
                             version_exit_code=completed.returncode, version_output=completed.stdout.strip())
    return details


def build_identity(*, block_threads=256):
    if block_threads not in BLOCK_CONFIGURATIONS:
        raise ValueError('Use a declared block configuration: ' + str(BLOCK_CONFIGURATIONS))
    if os.environ.get('TORCH_CUDA_ARCH_LIST') != RESEARCH_ARCH:
        raise RuntimeError('Set TORCH_CUDA_ARCH_LIST=12.0 explicitly; PTX/multi-architecture builds are separate experiments')
    closure = source_closure()
    config = _build_config()
    cxx, cuda = config.compile_flags()
    define = '-DMIXED_FUSION_BLOCK_THREADS=' + str(block_threads)
    cxx_flags = [*cxx, define]
    cuda_flags = [*cuda, '--fmad=false', '--ftz=false', '--prec-div=true', '--prec-sqrt=true', define]
    _check_precision_flags(cxx_flags, cuda_flags)
    toolchain = config.toolchain_identity(True)
    compilers = _compiler_details(toolchain)
    import torch
    if torch.version.cuda is None:
        raise RuntimeError('This research extension requires a CUDA-enabled Torch build')
    return dict(schema_version=1, candidate_kind='inference_low_storage_pad_cast_to_fp32',
        entrypoints=list(ENTRYPOINTS), local_sources=closure['local_sources'],
        external_include_names=closure['external_include_names'], closure_policy=closure['closure_policy'],
        loader_sha256=sha256(__file__), build_config_sha256=sha256(ROOT / 'Converse2D/build_config.py'),
        block_threads=block_threads, architecture=RESEARCH_ARCH, torch=str(torch.__version__),
        torch_git=torch.version.git_version, torch_cuda=torch.version.cuda,
        torch_cxx11_abi=getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', None),
        python=sys.version, platform=sys.platform, toolchain=toolchain, compiler_binaries=compilers,
        additional_flag_environment={key: os.environ.get(key, '') for key in ('CXXFLAGS', 'CFLAGS', 'CUDAFLAGS')},
        cxx_flags=cxx_flags, cuda_flags=cuda_flags,
        runtime_policy=dict(inference_only=True, autocast=False, tf32=False,
                            input_storage=['float16', 'bfloat16'], output='float32',
                            other_solver_compute='Existing FP32/complex64 solver unchanged'))


def _artifact_path(artifacts):
    path = Path(artifacts).resolve()
    root = BUILD_ROOT.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError('Use a named fresh run directory inside ' + str(root))
    return path


def _verify_snapshot(artifacts, identity):
    for name, digest in identity['local_sources'].items():
        if sha256(artifacts / 'sources' / name) != digest:
            raise RuntimeError('Frozen source/header changed: ' + name)
    if sha256(artifacts / 'sources/loader.py') != identity['loader_sha256']:
        raise RuntimeError('Frozen loader changed')
    if sha256(artifacts / 'support/build_config.py') != identity['build_config_sha256']:
        raise RuntimeError('Frozen production build configuration changed')
    saved = json.loads((artifacts / 'build_inputs.json').read_text(encoding='utf-8'))
    if saved != identity:
        raise RuntimeError('Saved research build inputs differ from manifest identity')


def load_checked(artifacts, *, build=False, block_threads=256, verbose=False):
    """Explicit fresh build or verified reload; never touches release binaries."""
    global _loaded, _loaded_key, _loaded_manifest
    artifacts = _artifact_path(artifacts)
    if os.name == 'nt':
        if os.environ.get('CONVERSE2D_BUILD_PATH'):
            os.environ['PATH'] = os.environ['CONVERSE2D_BUILD_PATH']
        os.environ.setdefault('VSLANG', '1033')
    identity = build_identity(block_threads=block_threads)
    token = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    key = str(artifacts), token
    if _loaded is not None:
        if _loaded_key != key:
            raise RuntimeError('Research variant/build inputs changed; use a fresh process')
        _verify_snapshot(artifacts, identity)
        if sha256(_loaded_manifest['binary']) != _loaded_manifest['binary_sha256']:
            raise RuntimeError('Loaded research binary changed on disk')
        return _loaded, _loaded_manifest

    manifest_path = artifacts / 'manifest.json'
    if build:
        if artifacts.exists():
            raise FileExistsError('Build directory already exists; preserve it and choose a fresh run name')
        sources, binary_dir = artifacts / 'sources', artifacts / 'build'
        sources.mkdir(parents=True)
        binary_dir.mkdir()
        for name in identity['local_sources']:
            target = sources / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(HERE / name, target)
        shutil.copyfile(Path(__file__), sources / 'loader.py')
        (artifacts / 'support').mkdir()
        shutil.copyfile(ROOT / 'Converse2D/build_config.py', artifacts / 'support/build_config.py')
        (artifacts / 'build_inputs.json').write_text(json.dumps(identity, indent=2), encoding='utf-8')
        _verify_snapshot(artifacts, identity)
        from torch.utils import cpp_extension
        if os.name == 'nt':
            cpp_extension.SUBPROCESS_DECODE_ARGS = ('utf-8', 'replace')
        name = 'converse_v4_mixed_pad_cast_' + token[:16]
        module = cpp_extension.load(name=name, sources=[str(sources / entry) for entry in ENTRYPOINTS],
            extra_cflags=identity['cxx_flags'], extra_cuda_cflags=identity['cuda_flags'],
            extra_include_paths=[str(sources)], with_cuda=True, build_directory=str(binary_dir), verbose=verbose)
        binary = Path(module.__file__).resolve()
        manifest = dict(identity=identity, module_name=name, binary=str(binary), binary_sha256=sha256(binary))
        with manifest_path.open('x', encoding='utf-8') as stream:
            json.dump(manifest, stream, indent=2)
    else:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['identity'] != identity:
            raise RuntimeError('Stale research source/compiler/flags; create a fresh explicit build')
        binary = Path(manifest['binary']).resolve()
        if not binary.is_relative_to((artifacts / 'build').resolve()):
            raise RuntimeError('Research binary escapes its isolated build directory')
        if sha256(binary) != manifest['binary_sha256']:
            raise RuntimeError('Research binary SHA mismatch')
        _verify_snapshot(artifacts, identity)
        spec = importlib.util.spec_from_file_location(manifest['module_name'], binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if build_identity(block_threads=block_threads) != identity:
        raise RuntimeError('Research source/compiler inputs changed during build/load')
    _loaded, _loaded_key, _loaded_manifest = module, key, manifest
    return module, manifest


@contextmanager
def _production_environment(manifest):
    previous = {key: os.environ.get(key) for key in ('TORCH_CUDA_ARCH_LIST', 'CONVERSE2D_SKIP_BUILD')}
    arch = manifest['inputs']['toolchain']['environment']['TORCH_CUDA_ARCH_LIST']
    try:
        if arch:
            os.environ['TORCH_CUDA_ARCH_LIST'] = arch
        else:
            os.environ.pop('TORCH_CUDA_ARCH_LIST', None)
        os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_production_checked():
    """Load baseline first or safely after research ARCH=12.0; restore env."""
    global _production_loader
    path = ROOT / '.build/cuda/source_manifest.json'
    manifest_hash = sha256(path)
    manifest = json.loads(path.read_text(encoding='utf-8'))
    with _production_environment(manifest):
        if _production_loader is None:
            spec = importlib.util.spec_from_file_location('mixed_pad_cast_production_loader', ROOT / 'test/extension_loader.py')
            _production_loader = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_production_loader)
        _production_loader.load_extension()
    if sha256(path) != manifest_hash:
        raise RuntimeError('Checked baseline loading modified the production manifest')
    binary = ROOT / '.build/cuda' / manifest['library']
    if sha256(binary) != manifest['binary_sha256']:
        raise RuntimeError('Production baseline binary mismatch')
    return dict(checked_manifest=manifest, manifest_sha256=manifest_hash,
                production_source_sha256=_production_loader.production_source_hashes(),
                architecture_restored=os.environ.get('TORCH_CUDA_ARCH_LIST', ''))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--block-threads', type=int, choices=BLOCK_CONFIGURATIONS, default=256)
    parser.add_argument('--load-production-first', action='store_true')
    parser.add_argument('--inspect-sources', action='store_true', help='Standard-library-only local include closure inspection')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    if args.inspect_sources:
        print(json.dumps(source_closure(), indent=2))
        return
    if args.artifacts is None:
        parser.error('--artifacts .build/mixed_fusion/<fresh-run> is required')
    if args.load_production_first:
        load_production_checked()
    _, manifest = load_checked(args.artifacts, build=args.build, block_threads=args.block_threads, verbose=args.verbose)
    print(json.dumps(dict(binary=manifest['binary'], binary_sha256=manifest['binary_sha256'],
                         block_threads=manifest['identity']['block_threads']), indent=2))


if __name__ == '__main__':
    main()
