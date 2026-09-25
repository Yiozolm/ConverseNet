"""Collect immutable roadmap evidence without launching CUDA or relaxing gates.

The detailed raw reports remain the authority. Missing studies stay missing;
failed candidates are never made eligible by favorable timings or other cases.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_records(value):
    if isinstance(value, dict):
        if 'sha256' in value and 'shape' in value and 'dtype' in value:
            return 1
        return sum(tensor_records(child) for child in value.values())
    if isinstance(value, list):
        return sum(tensor_records(child) for child in value)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, default=Path('artifacts/fp32_roadmap'))
    parser.add_argument('--research', type=Path, default=Path('.build/roadmap-research/research'))
    parser.add_argument('--quality-report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output; prior evidence must be preserved')
    inputs, missing = {}, []

    def read(path, *, required=True):
        if not path.is_file():
            if required:
                missing.append(str(path))
            return None
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        inputs[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        return value

    comparisons = {}
    for path in sorted(args.artifacts.glob('compare_*.json')):
        value = read(path)
        if 'timings' not in value:
            continue
        source_reports = [read(Path(name)) for name in value.get('files', [])]
        comparisons[path.stem] = {
            'settings_match': value.get('settings_match'),
            'harness_match': value.get('harness_match'),
            'helpers_match': value.get('helpers_match'),
            'missing_cases': value.get('missing_cases'),
            'tensor_hash_difference_count': len(value['tensor_hash_differences']),
            'tensor_hash_differences': value['tensor_hash_differences'],
            'noninferiority_failures': value['noninferiority_failures'],
            'prepared_state_equal': value.get('common_prepared_states_equal'),
            'timings': value['timings'],
            'source_reports': value.get('files'),
            'snapshot_tensor_counts': [sum(tensor_records(case.get('snapshot')) for case in report['cases'].values())
                                       if report else None for report in source_reports],
            'note': 'Complete callable medians. Earlier unpinned timings varied; fixed-affinity protocol, when used, is recorded in a separate per-process manifest. No significance or convergence claim.'}

    inference = {}
    for path in sorted(args.artifacts.glob('peripheral_inference_*.json')):
        value = read(path)
        inference[path.stem] = {
            'peripheral_cases': {name: {key: case[key] for key in
                ('shape', 'mode', 'precision', 'selected_helper_speedup')} for name, case in value['peripheral_cases'].items()},
            'graph_same_model': {name: {key: case[key] for key in
                ('runner_outputs_byte_equal', 'selected_runner_speedup')}
                for name, case in value.get('graph_runner_pair', {}).items()},
            'automatic_policy': 'Production enables alpha/affine fusion only without GradMode and with at least 2**21 elements. Private-helper tests include smaller shapes.'}

    algorithms = read(args.artifacts / 'research_numeric.json')
    research = {}
    if algorithms:
        for name, item in algorithms['candidates'].items():
            failures = [dict(case=index, spec=case['spec'], tensor=tensor, comparison=record)
                        for index, case in enumerate(item['cases'])
                        for tensor, record in case['tensors'].items() if not record['passed']]
            assert len(failures) == item['failures'], (name, len(failures), item['failures'])
            research[name] = dict(status=item['status'], cases=len(item['cases']),
                                 tensors=item['tensor_count'], failure_count=len(failures), failures=failures,
                                 failure_unit='output_or_requested_VJP_tensor', scope=algorithms['scope'],
                                 source_sha256=algorithms['source_sha256'])
    for name in ('fftfree_numeric', 'nearest_coefficient_numeric', 'cufftdx2d_32x40',
                 'cufftdx2d_36x44', 'cufftdx2d_100x100', 'pointwise_numeric', 'pointwise_split_numeric',
                 'fixed_transfer_numeric', 'full_layernorm_numeric', 'full_ln_parallel_numeric'):
        value = read(args.artifacts / (name + '.json'))
        if value is None:
            continue
        failures = value.get('failed_cases', value.get('failures'))
        pointwise = name.startswith('pointwise_')
        layernorm = name in ('full_layernorm_numeric', 'full_ln_parallel_numeric')
        provenance = value.get('provenance', {})
        research[name] = {
            'passed': value.get('passed', value.get('all_gpu_cases_passed', value.get('all_output_gates_passed'))),
            'cases': len(value.get('cases', [])), 'failures': failures,
            'failure_count': len(failures) if failures is not None else None,
            'failure_unit': 'output_or_requested_VJP_tensor' if pointwise else 'case',
            'tensor_count': sum(len(case.get('noninferiority', {})) for case in value['cases'].values()) if pointwise else None,
            'case_failure_count': sum(not case['passed'] for case in value['cases'].values()) if pointwise else (len(failures) if failures is not None else None),
            'timing_status': value.get('timing_status', value.get('timing_admission')),
            'scope': value.get('scope', provenance.get('scope',
                'Pointwise operator/VJP gate, including captured model tensors; no complete-model replacement' if pointwise else
                'Complete channel-first LayerNorm inference statistics and affine; original ATen GradMode fallback' if layernorm else None)),
            'contracts': value.get('contracts'), 'failed_contracts': value.get('failed_contracts'),
            'numeric_passed': value.get('numeric_passed'), 'cache_contracts_passed': value.get('cache_contracts_passed'),
            'provenance': {key: value[key] for key in ('api_sha256', 'harness_sha256', 'build_manifest', 'candidate_dependency_sha256') if key in value},
            'source_sha256': value.get('source_sha256', value.get('candidate_source_sha256',
                value.get('source_and_checkpoint_sha256', provenance.get('source_sha256'))))}
    layernorm_models = {}
    model_reports = [*args.artifacts.glob('full_ln*_model_*.json'),
                     *args.artifacts.glob('production_ln_ablation_*.json')]
    for path in sorted(model_reports):
        value = read(path)
        if 'cases' not in value:
            continue
        layernorm_models[path.stem] = {
            key: value[key] for key in ('status', 'settings', 'environment', 'scope', 'cold_definition',
                'timing_inclusions', 'source_model_object_shared', 'all_eager_outputs_byte_equal',
                'model_state_unchanged', 'all_instance_bindings_restored', 'gate_sha256',
                'tensor_versions_unchanged', 'public_marker_restored',
                'all_capture_context_eager_routes_byte_equal') if key in value}
        layernorm_models[path.stem]['cases'] = value['cases']
    fft = {}
    for name in ('lto_gpu_01', 'dx64_gpu_02', 'dx36_gpu_02', 'dx44_gpu_02', 'dx100_gpu_02'):
        value = read(args.research / 'fft_backends' / (name + '.json'))
        if value:
            fft[name] = value

    evidence = {}
    for name in ('resume_cuda_check', 'profile_summary', 'cpu_topology', 'long_campaign_plan'):
        evidence[name] = read(args.artifacts / (name + '.json'))
    quality = read(args.quality_report) if args.quality_report else None
    result = {
        'kind': 'fp32_roadmap_evidence_index',
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'release_commit_at_collection': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'collector_sha256': sha(__file__), 'inputs': inputs, 'missing': missing,
        'comparisons': comparisons, 'peripheral_inference': inference,
        'research_candidates': research, 'fft_primitive_probes': fft,
        'same_model_layernorm_studies': layernorm_models,
        'quality_report': quality, 'other_evidence': evidence,
        'interpretation': [
            'Research-candidate failures apply to these implementations and fixtures, not to every future algorithm in that direction.',
            'A primitive FFT or local operator speedup does not establish complete-model or training-quality speedup.',
            'The three paired long trajectories, stability decision and fixed-quality time must be read separately from numerical preservation.',
            'Historical failures and earlier noisy or rejected measurements remain in the immutable source reports.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), comparisons=len(comparisons),
                         research=len(research), missing=missing)))


if __name__ == '__main__':
    main()
