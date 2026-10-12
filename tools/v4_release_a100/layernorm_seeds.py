"""Fused LayerNorm channel-vector gradients over independent seeds, per geometry.

`test_layernorm_training.test_geometries_alignment_tails_and_eps` applies the
single-seed budget to dweight and dbias. Those are cancelling sums over all
pixels (the production-shape test gates them over eight seeds for that reason).
This sweep runs every geometry of that test over several seeds and reports, per
geometry and vector, the single-run pass count and the seeded verdict, so a
single-seed miss can be told apart from a systematic excess on a given GPU.

    python tools/v4_release_a100/layernorm_seeds.py --output <new json> [--seeds 8]
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--seeds', type=int, default=8)
parser.add_argument('--seed-base', type=int, default=7100)
args = parser.parse_args()
ROOT = args.root.resolve()
if args.output.exists():
    raise SystemExit(f'refusing to overwrite {args.output}')
sys.path[:0] = [str(ROOT / 'test'), str(ROOT)]
os.environ['CONVERSE2D_SKIP_BUILD'] = '1'
os.environ.pop('CONVERSE2D_BACKEND', None)

import torch  # noqa: E402
from extension_loader import load_extension  # noqa: E402
from numerical_policy import comparison, seeded_comparison  # noqa: E402
import test_layernorm_training as T  # noqa: E402

load_extension()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# The exact case list of the geometry test, plus the production shape as a control.
CASES = [((1, 64, 20, 24), 0, 1e-5), ((2, 64, 5, 7), 0, 1e-5), ((1, 64, 96, 96), 1, 1e-6),
         ((1, 128, 8, 8), 2, 1e-5), ((1, 128, 7, 10), 1, .2), ((1, 64, 1, 7), 0, 1e-5),
         ((2, 128, 7, 1), 0, 1e-5), ((3, 64, 33, 31), 3, 1e-5), ((4, 64, 96, 96), 0, 1e-5)]
NEEDS = (True, True, True)

report = dict(gpu=torch.cuda.get_device_name(0), capability='%d.%d' % torch.cuda.get_device_capability(),
              torch=torch.__version__, cuda=torch.version.cuda, seeds=args.seeds, seed_base=args.seed_base,
              started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), cases={})
all_seeded = True
for index, (shape, offset, eps) in enumerate(CASES):
    rows = {'dx': [], 'dweight': [], 'dbias': []}
    outputs_identical = True
    for seed in range(args.seeds):
        raw = T.fixture(shape, args.seed_base + 100 * index + seed, offset)
        assert raw[0].data_ptr() % 16 == offset * 4
        high, _ = T.evaluate(raw, NEEDS, eps, dtype=torch.float64)
        baseline, _ = T.evaluate(raw, NEEDS, eps)
        actual, active = T.evaluate(raw, NEEDS, eps, production=True)
        assert active, 'fused training node not active'
        outputs_identical &= bool(torch.equal(actual['output'], baseline['output']))
        for name in rows:
            rows[name].append(comparison(actual[name], baseline[name], high[name]))
    key = f'{"x".join(map(str, shape))} off{offset} eps{eps:g}'
    entry = dict(output_bytes_identical=outputs_identical, vectors={})
    for name, seeds in rows.items():
        verdict = seeded_comparison(seeds)
        ratios = [r['ratio']['rel_l2'] for r in seeds]
        entry['vectors'][name] = dict(
            single_run_failures=verdict['single_run_failures'], seeded_passed=verdict['passed'],
            rel_l2_ratio=dict(min=min(ratios), max=max(ratios),
                              geomean=float(torch.tensor(ratios, dtype=torch.float64).log().mean().exp())),
            candidate_rel_l2_max=max(r['candidate']['rel_l2'] for r in seeds),
            baseline_rel_l2_max=max(r['baseline']['rel_l2'] for r in seeds),
            seeded=verdict)
        all_seeded &= verdict['passed']
        print(f"{key:28s} {name:7s} single fails {verdict['single_run_failures']}/{args.seeds} "
              f"seeded {'pass' if verdict['passed'] else 'FAIL'} ratio geomean "
              f"{entry['vectors'][name]['rel_l2_ratio']['geomean']:.3f} max {max(ratios):.3f}", flush=True)
    report['cases'][key] = entry
report['all_seeded_passed'] = all_seeded
report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print('all seeded passed', all_seeded, 'wrote', args.output)
