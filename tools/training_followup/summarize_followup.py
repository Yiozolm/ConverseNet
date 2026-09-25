"""Combine completed diagnostics without widening their admission scopes."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'artifacts/training_followup')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Preserve earlier summaries; choose a new output')
    names = dict(historical='training_time_v1.json', paired='training_pairs_audit_v1.json',
                 wgrad='wgrad_summary.json', route_b='route_b_cuda.json', inference='inference_deployment_v1.json')
    reports = {key:json.loads((args.artifacts/name).read_text(encoding='utf-8')) for key,name in names.items()}
    historical, paired, wgrad, route, inference = (reports[k] for k in names)
    if (paired['status'] != 'paired_equal' or route['status'] != 'counterexample_reproduced'
            or inference['status'] != 'complete' or len(inference['cases']) != 16
            or not all(case['strict_numerical_lifecycle_gates_passed'] for case in inference['cases'].values())):
        raise ValueError('Required diagnostics are incomplete')
    summary = dict(kind='focused_training_followup_results', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   branch_base_commit='1d804b1', production_source_commit='ff7f8ba', historical_before_commit='a33e3af',
                   production_changes=False, new_training_candidate_admitted=False, new_inference_support_added=False,
                   tool_sha256=sha(__file__), input_sha256={names[k]:sha(args.artifacts/names[k]) for k in names})
    summary['historical_training'] = dict(scope=historical['scope'], timing_rules=historical['timing_rules'],
        pairs=[{k:p[k] for k in ('seed','matched_updates','matched_evaluations','current_minus_before_seconds',
                                'step_current_minus_before_seconds','process_current_minus_before_seconds')} for p in historical['pairs']],
        conclusion='Most extra recorded time is in forward/loss/backward. Sequential long-run stage attribution does not identify the cause of temporal variability. Historical full training remains slower.')
    summary['interleaved_short_workflow'] = dict(scope=paired['scope'], runs=8, paired_comparisons=4, updates_per_run=64,
        complete_evaluations_per_run=3, ratios=paired['ratios'], pairs=paired['pairs'], equality=paired['equality'],
        limitation='Evaluation every32 updates is denser than the historical every250. Small short-step gains and short-workflow gains are not proof of long-training acceleration, stability, or convergence.')
    profile=wgrad['profiles']['dataset']
    summary['wgrad'] = dict(decision=wgrad['decision'], numeric=wgrad['numerical'],
        kernel_count=profile['kernel_count'], kernel_sum_us=profile['kernel_sum_us'],
        wgrad_gpu_us=profile['mutually_exclusive_name_categories']['cuDNN_wgrad']['gpu_us'],
        wgrad_kernel_fraction=profile['mutually_exclusive_name_categories']['cuDNN_wgrad']['gpu_us']/profile['kernel_sum_us'],
        layernorm_gpu_us=profile['layernorm']['union_gpu_us'],
        layernorm_kernel_fraction=profile['layernorm']['union_gpu_us']/profile['kernel_sum_us'],
        candidate_scope='One B4 C128->64 HR96 module across five real invocations driven by synthetic inputs; full forward and VJP includes layout copies. Dataset profile is attribution only.',
        performance=wgrad['performance'], diagnostic_profile=wgrad['diagnostic_profile'], scope_limits=wgrad['scope_limits'])
    minimal=route['cases']['minimal_identity_impulse']
    summary['route_b'] = dict(candidate=route['candidate'], input_packet_sha256=route['input_packet_sha256'],
        minimality=route['minimality'], minimal_output_gate=minimal['output_gate'],
        first_common_difference=minimal['first_different_common_stage'], single_variable_interventions=minimal['single_variable_interventions'],
        candidate_admitted=False, scope=route['scope'])
    ratios={name:{kind:statistics.median(case['paired_ratios'][kind]['wall_ms']) for kind in case['paired_ratios']}
            for name,case in inference['cases'].items()}
    summary['inference'] = dict(cases=16, all_numerical_lifecycle_gates_passed=True,
        scope=inference['scope'], timing_scope=inference['timing_scope'], cold_scope=inference['cold_scope'],
        paired_wall_ratio_medians=ratios,
        per_case_median_ranges={kind:dict(minimum=min(row[kind] for row in ratios.values()), maximum=max(row[kind] for row in ratios.values())) for kind in next(iter(ratios.values()))},
        model_state_unchanged=inference['model_state_unchanged'], tensor_versions_unchanged=inference['tensor_versions_unchanged'],
        bindings_restored=inference['bindings_restored'], source_integrity_passed=inference['source_integrity_passed'],
        checked_binary_integrity_passed=inference['checked_binary_integrity_passed'], all_round_affinities_match=inference['all_round_affinities_match'],
        deployment_distribution='Representative supported model shapes, not production traffic samples or a measured deployment distribution.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), sha256=sha(args.output), inference_cases=16, new_candidate_admitted=False)))


if __name__ == '__main__':
    main()
