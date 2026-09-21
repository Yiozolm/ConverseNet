"""Verified frozen source/model baseline for the next training refinement round."""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import types

import torch
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SNAPSHOT = ROOT / "artifacts/training_refinements/source_before"
SOURCE = "Converse2D/torch_converse2d"
SOURCE_NAMES = ("converse2d.cpp", "converse2d_kernels.cu",
                "converse2d_training.cu", "converse2d_training.h")
MODEL_NAMES = ("converse_core", "util_converse", "converse_usrnet")
_extensions = {}
_models = {}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_snapshot(snapshot=DEFAULT_SNAPSHOT):
    snapshot = Path(snapshot).resolve()
    manifest_path = snapshot / "manifest.json"
    expected = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    required = {f"{SOURCE}/{name}" for name in SOURCE_NAMES}
    required.update(f"models/{name}.py" for name in MODEL_NAMES)
    if not required.issubset(expected):
        raise RuntimeError(f"Frozen manifest lacks {sorted(required.difference(expected))}")
    actual = {}
    for name, wanted in expected.items():
        path = (snapshot / name).resolve()
        if not path.is_relative_to(snapshot):
            raise RuntimeError(f"Frozen manifest path escapes snapshot: {name}")
        actual[name] = sha256(path)
        if actual[name] != wanted.lower():
            raise RuntimeError(f"Frozen source differs from manifest: {name}")
    fingerprint = hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest()[:16]
    return snapshot, actual, fingerprint


def load_baseline(snapshot=DEFAULT_SNAPSHOT, verbose=False):
    snapshot, hashes, fingerprint = verify_snapshot(snapshot)
    if fingerprint in _extensions:
        return _extensions[fingerprint]
    prefix = f"refinement_before_{fingerprint}_converse"
    namespace = prefix + "2d"
    build = ROOT / ".build/training_refinement_baseline" / fingerprint
    build.mkdir(parents=True, exist_ok=True)
    sources = []
    for name in SOURCE_NAMES:
        content = (snapshot / SOURCE / name).read_text(encoding="utf-8")
        content = content.replace("converse", prefix)
        target = build / name.replace("converse", prefix)
        if not target.exists() or target.read_text(encoding="utf-8") != content:
            target.write_text(content, encoding="utf-8")
        if target.suffix != ".h":
            sources.append(str(target))
    if os.name == "nt":
        if os.environ.get("CONVERSE2D_BUILD_PATH"):
            os.environ["PATH"] = os.environ["CONVERSE2D_BUILD_PATH"]
        os.environ.setdefault("VSLANG", "1033")
        cpp_extension.SUBPROCESS_DECODE_ARGS = ("utf-8", "replace")
    flags = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]
    flags += ["-DCONVERSE2D_WITH_CUDA=1", "-DREFINEMENT_FROZEN_REV=0x" + fingerprint[:12]]
    extension_name = "training_refinement_before_" + fingerprint
    # Always invoke the compiler's dependency validation. There is deliberately
    # no load_library/skip-build path that could accept an unrelated binary.
    cpp_extension.load(name=extension_name, sources=sources,
                       extra_include_paths=[str(build)], extra_cflags=flags,
                       extra_cuda_cflags=["-O3", "-lineinfo"], with_cuda=True,
                       is_python_module=False, build_directory=str(build), verbose=verbose)
    manifest = dict(ref="frozen-refinement-before:" + fingerprint,
                    source_sha256=hashes, snapshot=str(snapshot), namespace=namespace,
                    manifest_sha256=sha256(snapshot / "manifest.json"),
                    extension_name=extension_name, build_directory=str(build),
                    cflags=flags, cuda_flags=["-O3", "-lineinfo"])
    _extensions[fingerprint] = (getattr(torch.ops, namespace), manifest)
    return _extensions[fingerprint]


def load_frozen_models(snapshot=DEFAULT_SNAPSHOT):
    """Load verified original forwards, changing only their import package names."""
    snapshot, _, fingerprint = verify_snapshot(snapshot)
    if fingerprint in _models:
        return _models[fingerprint]
    package_name = "_refinement_before_models_" + fingerprint
    package = types.ModuleType(package_name)
    package.__path__ = [str(snapshot / "models")]
    sys.modules[package_name] = package
    modules = {}
    for name in MODEL_NAMES:
        path = snapshot / "models" / (name + ".py")
        module = types.ModuleType(package_name + "." + name)
        module.__file__ = str(path)
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        setattr(package, name, module)
        # Do not patch forward bodies or resolve imports against current models.
        source = re.sub(r"\bfrom models(?=[. ])", "from " + package_name,
                        path.read_text(encoding="utf-8"))
        exec(compile(source, str(path), "exec"), module.__dict__)
        modules[name] = module
    _models[fingerprint] = types.SimpleNamespace(**modules)
    return _models[fingerprint]
