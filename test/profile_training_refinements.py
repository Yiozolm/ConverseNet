"""Separate diagnostic profiles of refinement training; never a speed benchmark.

FFT/roll GPU shares use summed CUDA event durations, not elapsed wall time.
Kernel FFT attribution prefers the explicit prepare_training_kernel profiler
range. Frozen code without that range uses observed pad/roll/FFT preparation
inside a public forward, never tensor-shape guesses. Cache-hit
forwards contain no such preparation and their signal FFTs remain unassigned.
Backward attribution is reported
only when profiler autograd sequence numbers provide an observed match.
"""
import argparse
import collections
import json

import torch

import benchmark_fp32_training as common
from benchmark_training_refinements import load_variants, model_templates, report_metadata
from training_refinement_baseline import ROOT, DEFAULT_SNAPSHOT


PROFILE_CASES = {
    "usrnet-b16-s3": ("usrnet", (16, 3, 16, 20), 3, 1),
    "usrnet-default-tiny": ("usrnet_full", (1, 3, 8, 8), 2, 1),
}
FFT_CORE = {"aten::_fft_r2c", "aten::_fft_c2r", "aten::_fft_c2c"}
FFT_FRONTEND = {"aten::fft_rfft2", "aten::fft_rfft", "aten::fft_fft", "aten::fft_fft2"}


def ancestors(event):
    event = event.cpu_parent
    while event is not None:
        yield event
        event = event.cpu_parent


def inside(event, parent):
    return any(value.id == parent.id for value in ancestors(event))


def top_level(events, names):
    return [event for event in events if event.name in names and
            not any(parent.name in names for parent in ancestors(event))]


def sequence_ids(events, roots, names):
    result = set()
    for event in events:
        if event.name not in names:
            continue
        if any(event.id == root.id or inside(event, root) for root in roots):
            sequence = getattr(event, "sequence_nr", -1)
            if sequence >= 0:
                result.add(sequence)
    return result


def matched_backward(events, sequences, name_fragment):
    return [event for event in events
            if event.name.startswith("autograd::engine::evaluate_function:")
            and name_fragment in event.name
            and getattr(event, "sequence_nr", -1) in sequences]


def profile_summary(events, steps):
    gpu_events = [event for event in events
                  if event.device_type == torch.autograd.DeviceType.CUDA]
    total_gpu_us = sum(event.time_range.elapsed_us() for event in gpu_events)
    public = [event for event in events
              if "converse2d" in event.name and event.name.endswith("::forward")]
    explicit_prepare = [event for event in events
                        if event.name.endswith("::_prepare_training_kernel")]
    marked_prepare = [event for event in events
                      if event.name.endswith("::prepare_training_kernel")]
    prepared_forward = [event for event in events
                        if event.name.endswith("::_forward_prepared")]
    kernel_ffts, kernel_rolls = {}, {}
    scopes_without_preparation = []
    forward_ffts = top_level(events, FFT_FRONTEND)
    rolls = top_level(events, {"aten::roll"})
    pads = [event for event in events if event.name == "aten::constant_pad_nd"]
    if marked_prepare:
        method = "record_function_prepare_training_kernel_range"
        for event in forward_ffts:
            if any(inside(event, scope) for scope in marked_prepare):
                kernel_ffts[event.id] = event
        for event in rolls:
            if any(inside(event, scope) for scope in marked_prepare):
                kernel_rolls[event.id] = event
        # This capture has explicit instrumentation: a public forward without
        # a prepare range is a cache hit, and must never use first-FFT inference.
        scopes_without_preparation = [scope.name for scope in public
                                      if not any(inside(marked, scope) for marked in marked_prepare)]
    else:
        method = "frozen_source_pad_roll_fft_sequence_inference"
    for scope in [] if marked_prepare else [*public, *explicit_prepare]:
        candidates = sorted((event for event in forward_ffts if inside(event, scope)),
                            key=lambda event: event.time_range.start)
        previous_end, matched = scope.time_range.start, False
        for fft in candidates:
            preceding_rolls = [event for event in rolls if inside(event, scope)
                               and previous_end <= event.time_range.start < fft.time_range.start]
            preceding_pads = [event for event in pads if inside(event, scope)
                              and previous_end <= event.time_range.start < fft.time_range.start]
            explicit = scope.name.endswith("::_prepare_training_kernel")
            # Dense preparation has pad -> roll -> rfft2; separable preparation
            # has pad -> roll -> rfft followed by pad -> roll -> fft. Neither
            # signal transform follows this sequence inside the public solver.
            if explicit or (preceding_rolls and preceding_pads and
                            min(event.time_range.start for event in preceding_pads) <
                            max(event.time_range.start for event in preceding_rolls)):
                kernel_ffts[fft.id] = fft
                matched = True
                for event in preceding_rolls:
                    kernel_rolls[event.id] = event
            previous_end = fft.time_range.end
        if not matched:
            scopes_without_preparation.append(scope.name)
    kernel_ffts, kernel_rolls = list(kernel_ffts.values()), list(kernel_rolls.values())
    fft_sequences = sequence_ids(events, kernel_ffts, FFT_FRONTEND | FFT_CORE)
    roll_sequences = sequence_ids(events, kernel_rolls, {"aten::roll"})
    fft_backward = matched_backward(events, fft_sequences, "Fft")
    roll_backward = matched_backward(events, roll_sequences, "RollBackward")

    def group(values):
        gpu_us = sum(event.device_time_total for event in values)
        return dict(calls_per_step=len(values) / steps,
                    cpu_inclusive_us_per_step=sum(event.cpu_time_total for event in values) / steps,
                    gpu_inclusive_us_per_step=gpu_us / steps,
                    fraction_of_summed_gpu_events=gpu_us / total_gpu_us if total_gpu_us else None)

    kernels = collections.defaultdict(lambda: dict(calls=0, us=0.0))
    for event in gpu_events:
        kernels[event.name]["calls"] += 1
        kernels[event.name]["us"] += event.time_range.elapsed_us()
    top_kernels = sorted((dict(name=name, calls_per_step=row["calls"] / steps,
                               us_per_step=row["us"] / steps)
                          for name, row in kernels.items()), key=lambda row: -row["us_per_step"])
    fft_ops = top_level(events, FFT_CORE)
    return dict(
        summed_gpu_event_us_per_step=total_gpu_us / steps,
        groups={
            "all_fft_forward_and_backward": group(fft_ops),
            "all_roll_forward_and_backward": group(rolls),
            "kernel_fft_forward_attributed": group(kernel_ffts),
            "kernel_roll_forward_attributed": group(kernel_rolls),
            "kernel_prepare_marked_range": group(marked_prepare),
            "kernel_fft_backward_sequence_matched": group(fft_backward),
            "kernel_roll_backward_sequence_matched": group(roll_backward),
        },
        attribution=dict(method=method, public_forward_calls=len(public),
                         explicit_prepare_calls=len(explicit_prepare), marked_prepare_calls=len(marked_prepare),
                         prepared_forward_calls=len(prepared_forward),
                         kernel_fft_sequence_ids=sorted(fft_sequences),
                         kernel_roll_sequence_ids=sorted(roll_sequences),
                         fft_backward_matches=len(fft_backward), roll_backward_matches=len(roll_backward),
                         scopes_without_observed_preparation=scopes_without_preparation,
                         rule="Prefer prepare_training_kernel range ancestors for all FFT/roll attribution. Only uninstrumented frozen code falls back to observed pad -> roll -> FFT sequence. Cache hits never assign signal FFTs. Supports dense/separable FFTs; backward uses observed sequence_nr matches, never shapes.",
                         limitation="Zero backward matches means attribution unavailable, not zero cost. Aggregated FFT/roll groups remain valid; inspect the exported trace. Groups overlap and must not be added."),
        top_cuda_events=top_kernels[:50],
        fft_events=[dict(name=event.name, shapes=event.input_shapes,
                         sequence_nr=getattr(event, "sequence_nr", -1),
                         cpu_parent=event.cpu_parent.name if event.cpu_parent else None,
                         gpu_inclusive_us=event.device_time_total)
                    for event in fft_ops],
    )


def run_profile(template, batches, targets, ops, args, trace):
    with common.production_namespace(ops):
        runner = common.TrainingRunner(template, batches, targets)
        for _ in range(args.warmup):
            runner.step()
        runner.reset()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=False, with_stack=False,
        ) as profiler:
            for _ in range(args.steps):
                with torch.profiler.record_function("refinement_profile/full_training_step"):
                    runner.step()
            torch.cuda.synchronize()
        profiler.export_chrome_trace(str(trace))
        return profile_summary(profiler.events(), args.steps)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT))
    parser.add_argument("--case", choices=tuple(PROFILE_CASES), default="usrnet-b16-s3")
    parser.add_argument("--variant", choices=("before", "after", "both"), default="both")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--reuse-spectra", action="store_true")
    parser.add_argument("--output-dir", default="artifacts/training_refinements/profile")
    args = parser.parse_args()
    if min(args.steps, args.warmup) < 1:
        parser.error("steps and warmup must be positive")
    variants, baseline, frozen, dispatch = load_variants(args.snapshot, args.verbose_build)
    config = PROFILE_CASES[args.case]
    templates, batches, targets = model_templates(config, frozen, args.reuse_spectra)
    report = report_metadata(args, baseline, dispatch)
    report["timing_scope"] = "Diagnostic profiler capture only; do not use as benchmark timings or combine with independent latency results"
    report["config"] = config
    report["model_size"] = dict(iterations=5 if config[0] == "usrnet_full" else 2,
                                blocks=7 if config[0] == "usrnet_full" else 1)
    report["profiles"] = {}
    directory = ROOT / args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    for label, name in (("before", "dev"), ("after", "current")):
        if args.variant not in (label, "both"):
            continue
        common.release_cuda()
        trace = directory / f"{args.case}.{label}.trace.json"
        result = run_profile(templates[name], batches, targets, variants[name], args, trace)
        result["trace"] = str(trace)
        report["profiles"][label] = result
        common.release_cuda()
        print(json.dumps(dict(variant=label, groups=result["groups"], trace=str(trace))), flush=True)
        output = directory / f"{args.case}.summary.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", output, flush=True)


if __name__ == "__main__":
    main()
