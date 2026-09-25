"""Audit closed ABBA/BAAB runs; separate paired step and full workflow scopes."""
import argparse
import datetime
import json
import math
from pathlib import Path
import statistics
import sys

from audit_training_time import ROOT, WORKER, assert_matched, distribution, partition, read, rows, sha, step_phases


def same_state(a, b, path='root'):
    import torch
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8)):
            raise ValueError(f'Checkpoint tensor differs: {path}')
        return 1
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError(f'Checkpoint keys differ: {path}')
        return sum(same_state(a[k], b[k], f'{path}/{k}') for k in a)
    if isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError(f'Checkpoint sequence differs: {path}')
        return sum(same_state(x, y, f'{path}/{i}') for i, (x, y) in enumerate(zip(a, b)))
    if a != b:
        raise ValueError(f'Checkpoint scalar differs: {path}')
    return 0


def analyze(directory):
    import torch
    campaign = read(directory/'campaign.json')
    if campaign['status'] != 'complete' or len(campaign['runs']) != 8 or sha(WORKER) != campaign['worker_sha256']:
        raise ValueError('Campaign incomplete or worker changed')
    results, logs, checkpoints, evaluation_rows = {}, {}, {}, {}
    for item in campaign['runs']:
        path = directory/item['name']
        for name, checksum in item['input_sha256'].items():
            if sha(path/name) != checksum:
                raise ValueError(f'Changed evidence: {path/name}')
        report, training, evaluations = read(path/'run.json'), rows(path/'training.jsonl'), rows(path/'evaluations.jsonl')
        variant = item['variant']
        expected = campaign['expected_identity_by_variant'][variant]
        if (report['source_sha256'] != expected['source_sha256'] or
                report['backend']['build_manifest'] != expected['build_manifest']):
            raise ValueError(f'Frozen source or checked build changed: {path}')
        config = report['config']
        expected_config = dict(variant=variant, seed=17, steps=campaign['steps'], eval_every=campaign['eval_every'],
                               batch_size=4, microbatch_size=4, patch_size=96, scale=3, lr=1e-5,
                               init='pretrained', loss='mse', deterministic_algorithms=True)
        if any(config[key] != value for key, value in expected_config.items()):
            raise ValueError(f'Unexpected worker recipe: {path}')
        expected_root = campaign['baseline'] if variant == 'before' else campaign['current']
        if Path(config['root']).resolve() != Path(expected_root).resolve() or Path(config['run_dir']).resolve() != path.resolve():
            raise ValueError(f'Wrong source root or output directory: {path}')
        if report['status'] != 'complete' or len(training) != campaign['steps'] or report['optimizer_steps'] != campaign['steps']:
            raise ValueError('Incomplete worker')
        if [r['optimizer_steps'] for r in evaluations] != [0, campaign['eval_every'], campaign['steps']]:
            raise ValueError('Evaluation cadence or count differs')
        affinity = read(directory/f"{item['name']}_affinity.json")
        if (affinity['status'] != 'complete' or affinity['requested_mask_decimal'] != 12598275 or
                int(affinity['verified_current_mask_hex'], 16) != 12598275 or
                affinity['python_child_probe']['mask_decimal'] != 12598275 or affinity['task_exit_code'] != 0):
            raise ValueError('Invalid CPU affinity protocol')
        timing = report['timing']
        checks = [(math.fsum(row[key] for row in training), timing['total_'+key])
                  for key in ('training_step_wall_ms', 'h2d_wall_ms', 'forward_backward_wall_ms')]
        checks += [(math.fsum(row['data_prepare_and_hash_wall_ms'] for row in training), timing['data_prepare_and_hash_wall_ms']),
                   (math.fsum(row['wall_s'] for row in evaluations), timing['evaluation_wall_s'])]
        checks += [(math.fsum(row[key]['wall_ms'] for row in training), timing[key+'_wall_ms'])
                   for key in ('finite_check', 'optimizer')]
        if not all(math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-6) for a,b in checks):
            raise ValueError(f'Row and cumulative timer mismatch: {path}')
        outer = partition(report['process_elapsed_wall_s'], timing['setup_wall_s'], report['checkpoints']['initial']['wall_s'],
                          timing['checkpoint_wall_s'], timing['data_prepare_and_hash_wall_ms']/1000,
                          timing['total_training_step_wall_ms']/1000, timing['evaluation_wall_s'])
        result = dict(name=item['name'], variant=item['variant'], process_seconds=report['process_elapsed_wall_s'],
                      wrapper_seconds=item['wrapper_process_wall_s'], partition_seconds=outer,
                      step_partition_seconds={key:math.fsum(step_phases(r)[key] for r in training)/1000 for key in step_phases(training[0])},
                      step_wall_ms=distribution([r['training_step_wall_ms'] for r in training]),
                      warmed_step_6_onward_wall_ms=distribution([r['training_step_wall_ms'] for r in training[5:]]),
                      peak_allocated=report['training_peak_memory']['allocated_bytes'],
                      source_sha256=report['source_sha256'], build_manifest=report['backend']['build_manifest'],
                      environment=report['environment'], initial_state=report['origin_initial_state_tensor_sha256'],
                      recipe=report['comparison_recipe_sha256'], input_sha256=item['input_sha256'])
        results[item['name']] = result
        logs[item['name']] = training
        evaluation_rows[item['name']] = evaluations
        checkpoints[item['name']] = torch.load(path/'final.pth', map_location='cpu', weights_only=True)
    first = campaign['runs'][0]['name']
    equality = []
    for name, result in results.items():
        assert_matched(logs[first], logs[name])
        for field in ('environment','initial_state','recipe'):
            if result[field] != results[first][field]:
                raise ValueError(f'Identity changed: {name}, {field}')
        for b, c in zip(evaluation_rows[first], evaluation_rows[name]):
            for field in ('optimizer_steps','images','validation_payload_sha256','rgb','y','per_image'):
                if b[field] != c[field]:
                    raise ValueError(f'Evaluation differs: {name}, {field}')
        checks = {field:same_state(checkpoints[first][field],checkpoints[name][field],field)
                  for field in ('state_dict','optimizer_state_dict')}
        equality.append(dict(name=name, all_data_loss_norm_and_evaluation_equal=True, equal_checkpoint_tensor_counts=checks))
    pairs=[]
    for before,current in (('01_before','02_current'),('04_before','03_current'),('06_before','05_current'),('07_before','08_current')):
        b,c=results[before],results[current]
        pairs.append(dict(before=before,current=current,
                          process_before_over_current=b['process_seconds']/c['process_seconds'],
                          wrapper_before_over_current=b['wrapper_seconds']/c['wrapper_seconds'],
                          all_steps_before_over_current=b['partition_seconds']['training_step']/c['partition_seconds']['training_step'],
                          warmed_step_before_over_current=b['warmed_step_6_onward_wall_ms']['mean']/c['warmed_step_6_onward_wall_ms']['mean'],
                          fb_before_over_current=b['step_partition_seconds']['forward_backward']/c['step_partition_seconds']['forward_backward'],
                          process_current_minus_before_seconds={key:c['partition_seconds'][key]-b['partition_seconds'][key] for key in b['partition_seconds']}))
    ratios={key:dict(median=statistics.median(pair[key] for pair in pairs),minimum=min(pair[key] for pair in pairs),maximum=max(pair[key] for pair in pairs))
            for key in ('process_before_over_current','wrapper_before_over_current','all_steps_before_over_current','warmed_step_before_over_current','fb_before_over_current')}
    return dict(kind='interleaved_training_workflow_audit',status='paired_equal',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                campaign_sha256=sha(directory/'campaign.json'),tool_sha256=sha(__file__),
                scope=f"Four paired fresh-process comparisons in fixed ABBA/BAAB order; same seed17, data and frozen worker; {campaign['steps']} updates and 3 full evaluations each. Diagnostic only, not long-run performance or convergence.",
                exclusions='Warmed step means exclude steps1-5 only in explicitly warmed diagnostics; full worker process and all-step sums include them. Wrapper and worker scopes are separate, not added.',
                ratios=ratios,pairs=pairs,equality=equality,runs=list(results.values()))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Refusing to overwrite evidence')
    result=analyze(args.directory)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],ratios=result['ratios']),indent=2))


if __name__=='__main__':
    main()
