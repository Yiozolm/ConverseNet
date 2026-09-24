"""Build-input and compatibility contracts; does not compile or require CUDA."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

def config_at(root):
    spec = importlib.util.spec_from_file_location('layout_config', root / 'Converse2D/build_config.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

class BuildLayout(unittest.TestCase):
    def test_explicit_sources_exclude_candidates_and_facades(self):
        config = config_at(ROOT)
        self.assertFalse(any('shared_s1' in p or 'low_precision' in p for p in config.source_names()))
        self.assertNotIn('converse2d_training.cu', config.source_names())
        self.assertNotIn('converse2d_kernels.cu', config.source_names())
        self.assertEqual(len(config.source_names()), len({Path(p).name for p in config.source_names()}))
        self.assertFalse(any(p.endswith('.cu') for p in config.source_names(False)))

    def test_transitive_header_changes_invalidate_hashes(self):
        config = config_at(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in config.dependency_names():
                dest = root / 'Converse2D/torch_converse2d' / name
                dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(config.PACKAGE/name,dest)
            for name in ('build_config.py','setup.py'):
                shutil.copyfile(ROOT/'Converse2D'/name,root/'Converse2D'/name)
            copied = config_at(root)
            before = copied.source_hashes()
            header = copied.PACKAGE/'inference/detail/math.cuh'
            header.write_text(header.read_text()+'\n// dependency probe\n')
            after = copied.source_hashes()
            self.assertNotEqual(before, after)
            self.assertEqual([p for p in before if before[p] != after[p]], ['inference/detail/math.cuh'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
