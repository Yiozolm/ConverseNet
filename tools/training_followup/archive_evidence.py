"""Preserve closed follow-up reports losslessly; large fixtures stay local."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts/training_followup')
    parser.add_argument('--output', type=Path, default=ROOT/'docs/training_followup_evidence')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Use a fresh archive destination')
    campaign = args.artifacts/'training_pairs_v1/campaign.json'
    if json.loads(campaign.read_text(encoding='utf-8'))['status'] != 'complete':
        parser.error('Do not read a live or incomplete training campaign')
    candidates = sorted(p for p in args.artifacts.rglob('*') if p.is_file())
    archived, local_only = [], []
    args.output.mkdir(parents=True)
    for path in candidates:
        relative = path.relative_to(args.artifacts).as_posix()
        record = dict(source=relative, raw_bytes=path.stat().st_size, raw_sha256=sha(path))
        if path.suffix in ('.pt', '.pth'):
            local_only.append(record)
            continue
        if path.suffix not in ('.json', '.jsonl', '.log', '.md', '.csv', '.gz', '.py'):
            continue
        name = relative.replace('/', '__')
        target = args.output/(name if path.suffix == '.gz' else name+'.gz')
        if target.exists():
            raise RuntimeError('Archive name collision')
        if path.suffix == '.gz':
            shutil.copyfile(path, target)
            record['encoding'] = 'original gzip copied verbatim'
            if target.read_bytes() != path.read_bytes():
                raise RuntimeError('Copy mismatch')
        else:
            target.write_bytes(gzip.compress(path.read_bytes(), compresslevel=9, mtime=0))
            record['encoding'] = 'gzip of original bytes'
            if gzip.decompress(target.read_bytes()) != path.read_bytes():
                raise RuntimeError('Round-trip mismatch')
        record.update(archive=target.name, archive_bytes=target.stat().st_size, archive_sha256=sha(target))
        archived.append(record)
    index = dict(kind='lossless_training_followup_evidence', tool_sha256=sha(__file__),
                 archived=archived, local_only_large_tensor_fixtures=local_only,
                 scope='Reports, logs, inputs, telemetry and diagnostic traces are preserved byte-for-byte. Tensor checkpoints and captured activations remain in the local artifacts tree with hashes; datasets and production binaries are not duplicated.')
    (args.output/'index.json').write_text(json.dumps(index, indent=2), encoding='utf-8')
    (args.output/'README.md').write_text(
        '# Training follow-up evidence\n\n'
        'The index records SHA256 and byte counts for original files and lossless archives. '
        'For `gzip of original bytes`, decompress once to recover the exact original; existing '
        'trace `.gz` files were copied verbatim. Successful gates, performance rejection, '
        'toolchain loading failure and all paired runs remain separate.\n\n'
        'Large local checkpoints and activation fixtures are indexed without being committed. '
        'The standalone nearest-spectral input packet is also available as '
        '`../training_followup_route_b_inputs.json`. No production candidate is admitted by an archive status.\n', encoding='utf-8')
    print(json.dumps(dict(archived=len(archived), local_only=len(local_only),
                         archive_bytes=sum(r['archive_bytes'] for r in archived))))


if __name__ == '__main__':
    main()
