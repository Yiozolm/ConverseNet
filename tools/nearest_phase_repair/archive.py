"""Losslessly archive this completed, numerically rejected phase experiment."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'artifacts/nearest_phase_repair')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/nearest_phase_repair_evidence')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve earlier archives; use a new directory')
    gate = json.loads((args.source/'gate_v1.json').read_text(encoding='utf-8'))
    if (gate['status'] != 'candidate_failed_numeric_gate' or gate['tensor_count'] != 96
            or gate['old_failures'] != 45 or gate['new_failures'] != 45 or not gate['sources_unchanged']
            or gate['performance_allowed'] or gate['production_admitted']):
        parser.error('Expected the closed, fully evaluated numerical rejection')
    args.output.mkdir(parents=True)
    records = []
    for path in sorted(args.source.iterdir()):
        if not path.is_file() or path.suffix not in ('.json', '.log', '.md'):
            continue
        raw = path.read_bytes()
        compressed = gzip.compress(raw, compresslevel=9, mtime=0)
        target = args.output/(path.name+'.gz')
        target.write_bytes(compressed)
        if gzip.decompress(target.read_bytes()) != raw:
            raise RuntimeError('Archive round-trip mismatch')
        records.append(dict(source=path.relative_to(ROOT).as_posix(), archive=target.name,
                            raw_bytes=len(raw), raw_sha256=digest(raw),
                            gzip_bytes=len(compressed), gzip_sha256=digest(compressed)))
    result = dict(kind='nearest_phase_repair_lossless_evidence',
                  archive_tool_sha256=digest(Path(__file__).read_bytes()),
                  old_failures=45, new_failures=45, production_admitted=False,
                  provenance='Tensor bytes/hashes were measured in this experiment, not retroactively added to the historical report.',
                  files=records)
    (args.output/'index.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    (args.output/'README.md').write_text(
        '# nearest phase repair evidence\n\n'
        'Decompress each gzip once to recover its original bytes. The index binds raw and gzip sizes/SHA256. '
        'CPU/CUDA minimal examples pass; the full candidate still fails 45/96 tensor gates, including three '
        'new failures. Exit code4 and the wrapper task_failed status are preserved. No performance or '
        'production admission follows from the successful reproduction/audit.\n', encoding='utf-8')
    print(json.dumps(dict(files=len(records), raw_bytes=sum(r['raw_bytes'] for r in records),
                          gzip_bytes=sum(r['gzip_bytes'] for r in records))))


if __name__ == '__main__':
    main()
