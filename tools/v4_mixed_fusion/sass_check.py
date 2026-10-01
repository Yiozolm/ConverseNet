"""Inspect the exact checked research binary; never infer SASS from source."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuobjdump', type=Path,
                        default=Path(r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin\cuobjdump.exe'))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh SASS evidence directory')
    manifest = json.loads((args.artifacts / 'manifest.json').read_text(encoding='utf-8'))
    binary = Path(manifest['binary'])
    if sha(binary) != manifest['binary_sha256']:
        raise RuntimeError('Binary differs from checked manifest')
    args.output.mkdir(parents=True)
    target = args.output / 'kernel.sass'
    command = [str(args.cuobjdump), '--dump-sass', str(binary)]
    with target.open('w', encoding='utf-8') as out, (args.output / 'stderr.log').open('w', encoding='utf-8') as err:
        result = subprocess.run(command, stdout=out, stderr=err)
    text = target.read_text(encoding='utf-8')
    kernels = re.findall(r'Function\s*:\s*(\S+)', text)
    opcodes = re.findall(r'/\*[0-9a-fA-F]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)', text)
    double_arithmetic = sorted({op for op in opcodes if op.split('.')[0] in ('DADD', 'DMUL', 'DFMA', 'DSETP', 'DMNMX')})
    loads16 = [op for op in opcodes if op.startswith('LDG') and '.U16' in op]
    stores = [op for op in opcodes if op.startswith('STG')]
    conversions = sorted({op for op in opcodes if op.startswith(('F2F', 'F2FP', 'I2F'))})
    passed = (result.returncode == 0 and bool(kernels) and all('pad_cast_kernel' in name for name in kernels)
              and bool(loads16) and bool(stores) and not double_arithmetic)
    record = dict(kind='mixed_fusion_sass', passed=passed, command=command, returncode=result.returncode,
                  source_sha256=sha(__file__), binary_sha256=sha(binary), manifest=manifest,
                  sass_path=str(target.resolve()), sass_sha256=sha(target), kernel_names=kernels,
                  static_instruction_count=len(opcodes), global_u16_loads=len(loads16), global_stores=len(stores),
                  conversion_opcodes=conversions, fp64_arithmetic=double_arithmetic,
                  scope='Confirms 16-bit global reads, global writes and no FP64 arithmetic; FP32 buffer/value contract is independently checked by the numerical gate.')
    (args.output / 'summary.json').write_text(json.dumps(record, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: record[k] for k in ('passed', 'global_u16_loads', 'global_stores', 'conversion_opcodes', 'fp64_arithmetic')}))
    if not passed:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
