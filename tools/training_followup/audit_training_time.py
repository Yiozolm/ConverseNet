"""Hash-bound, non-overlapping accounting of completed paired training logs."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
WORKER = ROOT / 'tools/roadmap_quality/train_usrnet_dataset.py'
STEP_PHASES = ('h2d', 'forward_backward', 'finite_check', 'optimizer')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line]


def positive(value, label):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'Invalid duration {label}: {value}')
    return value


def step_phases(row):
    result = dict(h2d=row['h2d_wall_ms'], forward_backward=row['forward_backward_wall_ms'],
                  finite_check=row['finite_check']['wall_ms'], optimizer=row['optimizer']['wall_ms'])
    result['unattributed_step'] = row['training_step_wall_ms'] - sum(result.values())
    for key, value in result.items():
        positive(value, key)
    return result


def partition(process, setup, initial_checkpoint, checkpoints, data, training, evaluation):
    """Initial checkpoint is nested in setup; include it exactly once."""
    result = dict(setup_excluding_initial_checkpoint=setup - initial_checkpoint,
                  data_prepare_and_hash=data, training_step=training,
                  evaluation_body=evaluation, checkpoint=checkpoints)
    result['unattributed_process'] = process - sum(result.values())
    for key, value in result.items():
        positive(value, key)
    if not math.isclose(sum(result.values()), process, rel_tol=1e-12, abs_tol=1e-8):
        raise ValueError('Outer accounting did not close')
    return result


def distribution(values):
    values = sorted(values)
    return dict(count=len(values), sum=math.fsum(values), mean=statistics.mean(values),
                median=statistics.median(values), p90=values[math.ceil(.9 * len(values)) - 1])


def assert_matched(before, current):
    if len(before) != len(current) or [r['step'] for r in before] != list(range(1, len(before) + 1)):
        raise ValueError('Missing, duplicate, or unmatched updates')
    for b, c in zip(before, current):
        for key in ('step', 'data_step', 'batch_sha256', 'samples', 'microbatches', 'optimizer_steps', 'loss', 'grad_l2_norm'):
            if b[key] != c[key]:
                raise ValueError(f"Mismatched update {b['step']}: {key}")


def session(directory, hashes):
    for name, checksum in hashes.items():
        if sha(directory / name) != checksum:
            raise ValueError(f'Evidence changed: {directory / name}')
    report, training, evaluations = read(directory / 'run.json'), rows(directory / 'training.jsonl'), rows(directory / 'evaluations.jsonl')
    if report['source_sha256']['tools/roadmap_quality/train_usrnet_dataset.py'] != sha(WORKER):
        raise ValueError('Timing-scope worker changed since capture')
    timing = report['timing']
    totals = {name: math.fsum(step_phases(row)[name] for row in training) / 1000
              for name in (*STEP_PHASES, 'unattributed_step')}
    step_sum = math.fsum(row['training_step_wall_ms'] for row in training) / 1000
    data_sum = math.fsum(row['data_prepare_and_hash_wall_ms'] for row in training) / 1000
    eval_sum = math.fsum(row['wall_s'] for row in evaluations)
    checks = ((step_sum, timing['total_training_step_wall_ms'] / 1000),
              (data_sum, timing['data_prepare_and_hash_wall_ms'] / 1000),
              (eval_sum, timing['evaluation_wall_s']))
    if not all(math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-6) for a, b in checks):
        raise ValueError(f'Log and cumulative timer mismatch: {directory}')
    initial = report['checkpoints']['initial']['wall_s']
    outer = partition(report['process_elapsed_wall_s'], timing['setup_wall_s'], initial,
                      timing['checkpoint_wall_s'], data_sum, step_sum, eval_sum)
    return dict(name=directory.name, seed=report['config']['seed'], variant=report['config']['variant'],
                status=report['status'], session_start_step=report['session_start_step'],
                updates=len(training), evaluations=len(evaluations), process_seconds=report['process_elapsed_wall_s'],
                partition_seconds=outer, step_partition_seconds=totals,
                initial_checkpoint_nested_in_setup_seconds=initial,
                first_update={k: training[0][k] for k in ('step', 'training_step_wall_ms', 'forward_backward_wall_ms')},
                comparison_recipe_sha256=report['comparison_recipe_sha256'],
                origin_initial_state_tensor_sha256=report['origin_initial_state_tensor_sha256'],
                validation_payload_sha256=report['validation_payload_sha256'],
                deterministic_algorithms=report['config']['deterministic_algorithms'],
                input_sha256=hashes), training, evaluations


def analyze(artifacts):
    identity_path = artifacts / 'quality_final_numeric_audit.json'
    identities = read(identity_path)
    audit_path = artifacts / 'quality_final_audit.json'
    if sha(audit_path) != identities['audit_sha256']:
        raise ValueError('Final quality audit changed')
    audit = read(audit_path)
    if audit['status'] != 'quality_gates_passed' or audit['errors'] or audit['missing']:
        raise ValueError('Completed paired audit is not admissible')
    sessions, data, evaluations = [], {}, {}
    for original, hashes in identities['input_sha256'].items():
        directory = artifacts / original.replace('\\', '/').split('/')[-1]
        record, train_rows, eval_rows = session(directory, hashes)
        sessions.append(record)
        key = (record['seed'], record['variant'])
        data.setdefault(key, []).extend(train_rows)
        evaluations.setdefault(key, []).extend(eval_rows)
    pairs = []
    for seed in (17, 29, 43):
        before, current = [sorted(data[(seed, variant)], key=lambda x: x['step']) for variant in ('before', 'current')]
        assert_matched(before, current)
        matching = [s for s in sessions if s['seed'] == seed]
        for key in ('comparison_recipe_sha256', 'origin_initial_state_tensor_sha256', 'validation_payload_sha256', 'deterministic_algorithms'):
            if len({s[key] for s in matching}) != 1:
                raise ValueError(f'Mismatched trajectory identity: {seed}, {key}')
        outer, inner = {}, {}
        for variant in ('before', 'current'):
            selected = [s for s in matching if s['variant'] == variant]
            outer[variant] = {key: math.fsum(s['partition_seconds'][key] for s in selected) for key in selected[0]['partition_seconds']}
            inner[variant] = {key: math.fsum(s['step_partition_seconds'][key] for s in selected) for key in selected[0]['step_partition_seconds']}
        be, ce = [sorted(evaluations[(seed, v)], key=lambda x: x['optimizer_steps']) for v in ('before', 'current')]
        if [r['optimizer_steps'] for r in be] != list(range(0, 4001, 250)) or [r['optimizer_steps'] for r in ce] != list(range(0, 4001, 250)):
            raise ValueError('Unmatched or duplicate complete evaluations')
        blocks = []
        for offset in range(0, len(before), 250):
            b, c = before[offset:offset+250], current[offset:offset+250]
            blocks.append(dict(first_step=offset+1, last_step=offset+len(b),
                               before_forward_backward_ms=distribution([r['forward_backward_wall_ms'] for r in b]),
                               current_forward_backward_ms=distribution([r['forward_backward_wall_ms'] for r in c]),
                               paired_current_over_before_median=statistics.median(y['forward_backward_wall_ms'] / x['forward_backward_wall_ms'] for x, y in zip(b, c))))
        pairs.append(dict(seed=seed, matched_updates=len(before), matched_evaluations=len(be),
                          exclusive_process_seconds=outer, exclusive_step_seconds=inner,
                          current_minus_before_seconds={k: outer['current'][k] - outer['before'][k] for k in outer['before']},
                          step_current_minus_before_seconds={k: inner['current'][k] - inner['before'][k] for k in inner['before']},
                          process_current_minus_before_seconds=sum(outer['current'].values()) - sum(outer['before'].values()),
                          paired_step_wall_ratio=distribution([c['training_step_wall_ms'] / b['training_step_wall_ms'] for b, c in zip(before, current)]),
                          blocks_250=blocks))
    return dict(kind='matched_training_time_accounting', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                scope='Historical 3 seeds x before/current x 4000 identical updates; seven sessions; no rerun or causal hardware attribution.',
                timing_rules=[
                    'Each outer partition closes to the worker process timer, starting in main after module imports. Interpreter/wrapper startup and final report write are excluded.',
                    'Initial checkpoint is subtracted from setup before all checkpoints are included once. Failed-prefix and recovery process timers both remain included.',
                    'Step subphases are sequential and explicitly synchronized by the worker; their sum is nested in training_step and never added to it.',
                    'CUDA event spans include gaps/CPU submission delays, not only GPU kernels. Wall and event timers are never added together.',
                    'Data is synchronous CPU generation/validation/hash, not an asynchronous DataLoader wait measurement. H2D is inside the step.',
                    'Evaluation body excludes its prechecks/cache setup and cleanup; these, logging, metadata and unmeasured checks remain unattributed.',
                    'The 282.197191 second recovery restart interval is separate from process timers and is not added to this ledger.',
                    'Optimizer work and scheduled evaluation counts match. Recovery adds setup and an initial checkpoint, retained as actual cost.',
                    'Chronological processes were not interleaved benchmarks; stage attribution cannot prove clocks, temperature, contention or a code regression caused the differences.'],
                input_sha256={str(identity_path):sha(identity_path), str(audit_path):sha(audit_path), str(WORKER):sha(WORKER)},
                tool_sha256=sha(__file__), sessions=sessions, pairs=pairs)


def markdown(report):
    lines = ['# 匹配训练流程耗时审计', '', report['scope'], '',
             '同一种子按数据哈希及更新编号配对；模型初始化、配方、4000 个 loss/梯度范数均一致。历史失败及恢复成本保留。', '',
             '下表单位秒，正值表示 current 更慢。外层各项互斥；训练步子项另表展开，不重复加入总计。', '',
             '| Seed | 初始化（扣初始保存） | 数据准备/哈希 | 训练步 | 评估主体 | 全部保存 | 未归因余量 | 进程总差 |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ('setup_excluding_initial_checkpoint', 'data_prepare_and_hash', 'training_step', 'evaluation_body', 'checkpoint', 'unattributed_process')
    for pair in report['pairs']:
        lines.append('| ' + str(pair['seed']) + ' | ' + ' | '.join(f"{pair['current_minus_before_seconds'][key]:+.6f}" for key in keys) + f" | {pair['process_current_minus_before_seconds']:+.6f} |")
    lines += ['', '| Seed | H2D | 前向+loss+backward | 有限性/梯度范数 | Adam | 步内未归因余量 |', '|---:|---:|---:|---:|---:|---:|']
    for pair in report['pairs']:
        lines.append('| ' + str(pair['seed']) + ' | ' + ' | '.join(f"{pair['step_current_minus_before_seconds'][key]:+.6f}" for key in (*STEP_PHASES,'unattributed_step')) + ' |')
    lines += ['', '## 计时边界', '', *['- ' + rule for rule in report['timing_rules']], '',
              '机器报告包含每个 session 的完整账本、每 250 步分块统计及输入 SHA。本审计解释时间落在哪个阶段，不把进程顺序与时间漂移推断为优化代码的因果效应。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT / 'artifacts/fp32_roadmap')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    md = args.output.with_suffix('.md')
    if args.output.exists() or md.exists():
        parser.error('Refusing to overwrite an earlier report')
    report = analyze(args.artifacts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    md.write_text(markdown(report), encoding='utf-8')
    print(markdown(report))


if __name__ == '__main__':
    main()
