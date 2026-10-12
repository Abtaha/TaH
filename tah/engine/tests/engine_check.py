"""Bounded eager engine gates and workload/profile reports.

Run one checkpoint/gate per process. Production parity uses WP2's immutable
self-noise, with runtime/reference/checkpoint hashes recorded in every report.
"""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import yaml

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.fixtures import FixtureSuite
from tah.engine.parity import assert_gtf, compare_output, derive_decision_limits, gtf_summary, production_noise, require_production
from tah.engine.model import TeacherForcedOutput
from tah.engine.reference_capture import capture, divergence
from tah.engine.generate_fixtures import load_reference


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trace_profile_totals(trace):
    """Exclusive physical CUDA work and CPU scope duration from a saved trace.

    GPU user-annotation envelopes and nested CPU operation totals are excluded.
    This also permits auditing a report without executing the model again.
    """
    steps = [e for e in trace if e.get('cat') == 'user_annotation'
             and e.get('name') == 'engine_decode_step']
    spans = [(e['ts'], e['ts'] + e['dur']) for e in steps]
    work = [e for e in trace if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
            and any(a <= e['ts'] and e['ts'] + e['dur'] <= b for a, b in spans)]
    return dict(steps=len(steps), wall_us=sum(e['dur'] for e in steps),
                kernel_us=sum(e['dur'] for e in work if e['cat'] == 'kernel'),
                copy_us=sum(e['dur'] for e in work if e['cat'] != 'kernel'), activities=len(work))


class Observer:
    """Bounded device->pinned-host tracing; decode adds NO synchronization."""
    def __init__(self, engine):
        self.engine, self.records = engine, {}

    def host(self, tensor):
        out = torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=tensor.is_cuda)
        out.copy_(tensor, non_blocking=True)
        return out

    def prefill(self, state, start, output):
        rows = self.records.setdefault(state.index, [])
        for pos in range(output.logits.shape[0]):
            rows.append((start + pos, self.host(output.logits[pos]),
                         self.host(output.visited[:, pos]), self.host(output.decision[:, pos]),
                         self.host(output.decision_logits[:, pos]), self.host(output.continue_probability[:, pos])))

    def decode(self, states, bucket, status):
        for row, state in enumerate(states):
            self.records[state.index].append((state.position, self.host(bucket.logits[row]),
                                              state.depth, self.host(status[row])))

    def output(self, index, length):
        n, v = self.engine.model.max_iter, self.engine.model.config['vocab_size']
        logits = torch.empty(length, v, dtype=self.engine.model.dtype)
        visited = torch.zeros(n, length, dtype=torch.bool)
        decisions = torch.zeros_like(visited)
        raw = torch.full((n, length), float('nan'), dtype=self.engine.model.dtype)
        prob = torch.full_like(raw, float('nan'))
        for record in self.records[index]:
            pos, values = record[:2]
            logits[pos] = values
            if len(record) == 6:
                visited[:, pos], decisions[:, pos], raw[:, pos], prob[:, pos] = record[2:]
            else:
                depth, status = record[2:]
                visited[depth - 1, pos] = True
                decisions[depth - 1, pos] = bool(status[0])
                raw[depth - 1, pos], prob[depth - 1, pos] = status[2], status[3]
        return TeacherForcedOutput(logits, visited.sum(0), visited, decisions, raw, prob,
            torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(n, 0, 0, 0))


def fixture_setup(name):
    suite = FixtureSuite('data/engine/fixtures/reference')
    require_production(suite)
    meta = suite.manifest['checkpoints'][name]
    root = Path(meta['path'])
    for file, info in meta['files'].items():
        assert digest(root / file) == info['sha256'], f'checkpoint changed: {file}'
    for file, value in suite.manifest['reference_source_sha256'].items():
        assert digest(file) == value, f'reference changed: {file}'
    eval_config = Path(suite.manifest['eval_config']['path'])
    assert digest(eval_config) == suite.manifest['eval_config']['sha256']
    override = yaml.safe_load(eval_config.read_text()) if name == 'real' else None
    noise = production_noise(suite, name, json.loads(Path('data/engine/wp2/tolerances.json').read_text()))
    records = [r for r in suite.records(kind='greedy', checkpoint=name) if r['variant'] == 'single']
    assert len(records) == 6
    prompts = [suite.load(r)['prompt_ids'].tolist() for r in records]
    return suite, root, override, noise, records, prompts, eval_config


def parity(args):
    suite, root, override, noise, records, prompts, eval_config = fixture_setup(args.checkpoint)
    engine = TaHEngine(root, override, max_batch_size=6, prefill_chunk_size=128, kv_capacity=20000)
    observer = Observer(engine) if args.gate == 'tf' else None
    engine._observer = observer
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95) if args.gate == 'tf' else SamplingParams(temperature=0)
    results = engine.generate([Request(p, 256, i) for i, p in enumerate(prompts)], params)
    report = dict(gate='G-TF' if observer else 'G-GEN', checkpoint=args.checkpoint,
        production_gates_eligible=True, engine_run=engine.last_run,
        results=[asdict(r) for r in results], manifest_sha256=digest(suite.root / 'manifest.json'),
        wp2_tolerances_sha256=digest('data/engine/wp2/tolerances.json'),
        decision_limits=derive_decision_limits(noise))
    failure = None
    if observer:
        outputs = [observer.output(i, len(prompt) + max(0, len(result.token_ids) - (result.finish_reason == 'length')))
                   for i, (prompt, result) in enumerate(zip(prompts, results))]
        engine._observer = None
        del observer
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        oracle = load_reference(SimpleNamespace(device='cuda', eval_config=eval_config), args.checkpoint, root)
        rows = []
        for i, (prompt, result, output) in enumerate(zip(prompts, results, outputs)):
            sequence = prompt + result.token_ids[:output.logits.shape[0] - len(prompt)]
            # Replay actual engine-generated sequences with WP2 tooling. No
            # engine-vs-engine/cross-layout proxy can establish this gate.
            boundaries = list(range(0, len(prompt), 128)) + list(range(len(prompt), len(sequence) + 1))
            trace, logits = capture(oracle, [sequence], pad_id=oracle.config.eos_token_id,
                                    chunk_boundaries=boundaries)
            row = compare_output(trace[0], output, logits[0], suite.manifest['checkpoints'][args.checkpoint]['threshold'])
            row.update(sequence_id=records[i]['sequence_id'], generated_sequence=True)
            rows.append(row)
            print(f'G-TF {args.checkpoint}: replay {i + 1}/6', flush=True)
        report.update(summary=gtf_summary(rows), comparisons=rows, noise_limits=noise,
                      reference_layout='singleton prompt chunks128, then cached one-token chunks')
        try:
            assert_gtf(rows, noise)
        except AssertionError as exc:
            failure = str(exc)
    else:
        rows = []
        for record, result in zip(records, results):
            baseline = suite.load(record)['generated_ids'].tolist()
            d = divergence(baseline, result.token_ids)
            rows.append(dict(sequence_id=record['sequence_id'], first_divergence=d,
                             generation_length=len(baseline), effective_divergence=len(baseline) if d is None else d))
        baseline_lengths = {r['sequence_id']: len(suite.load(r)['generated_ids']) for r in records}
        reference = suite.manifest['self_noise'][args.checkpoint]['greedy']['comparisons']
        ref_positions = [baseline_lengths[r['sequence_id']] if r['first_divergence'] is None else r['first_divergence'] for r in reference]
        candidate_median, reference_median = statistics.median(r['effective_divergence'] for r in rows), statistics.median(ref_positions)
        report.update(comparisons=rows, median_first_divergence=candidate_median,
            reference_median_first_divergence=reference_median, median_limit=.5 * reference_median)
        if candidate_median < .5 * reference_median or (any(r['effective_divergence'] == 0 for r in rows) and 0 not in ref_positions):
            failure = 'G-GEN first-divergence bound failed'
    report['status'] = 'FAIL' if failure else 'PASS'
    if failure:
        report['failure'] = failure
    return report


def workload_prompts():
    baseline = Path('data/engine/wperf_bs16/math500_tah/20261011_155619/TASK_0_64/samples.jsonl')
    rows = [json.loads(line) for line in baseline.read_text().splitlines()]
    questions = {}
    for row in rows:
        questions[row['id']] = row['problem']
    assert len(questions) == 64
    # Exactly the reference evaluation chat template, rendered with its saved
    # tokenizer only in offline workload preparation (not engine runtime).
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('models/TaH-plus-1.7B')
    return [tokenizer.apply_chat_template([dict(role='user', content=questions[f'math500_{i}'])],
                                        tokenize=True, add_generation_prompt=True)
            for i in range(64) for _ in range(2)]


def workload(args):
    prompts = workload_prompts()[args.start:args.end]
    if args.gate == 'long-prompt':
        prompts = [(prompts[0] * (16384 // len(prompts[0]) + 1))[:16384]] + prompts[1:33]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine('models/TaH-plus-1.7B', override, max_batch_size=args.batch_size,
                      prefill_chunk_size=128, profile_steps=0, memory_fraction=args.memory_fraction)
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    # Load/warmup excluded; exercise actual maximum-B linear and bounded sampler.
    engine.generate([Request(prompts[i % len(prompts)], 4, i) for i in range(args.batch_size)], params)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    requests = [Request(p, args.max_new_tokens, i + args.start) for i, p in enumerate(prompts)]
    kv_high_water = engine.pool.capacity - engine.pool.free_slots
    allocate = engine.pool.allocate
    def measured_allocate(*args, **kwargs):
        nonlocal kv_high_water
        result = allocate(*args, **kwargs)
        kv_high_water = max(kv_high_water, engine.pool.capacity - engine.pool.free_slots)
        return result
    engine.pool.allocate = measured_allocate
    completed = []
    def done(result):
        completed.append(result)
        print(f'{args.gate} completed {len(completed)}/{len(requests)}, tokens={sum(len(r.token_ids) for r in completed)}', flush=True)
    results = engine.generate(requests, params, done)
    report = dict(status='PASS', workload='W-perf' if args.gate == 'perf' else 'G-ROBUST',
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, request_range=[args.start, args.end],
        requests=len(requests), total_prompt_tokens=sum(map(len, prompts)),
        memory_budget=engine.memory_budget, run=engine.last_run,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        kv_audit=engine.pool.audit(), host_swap_used_bytes=engine.swap.used_bytes,
        kv_pool_high_water_entries=kv_high_water,
        wp0_baseline_tokens_per_second=78.26, speedup=engine.last_run['tokens_per_second'] / 78.26,
        results=[asdict(r) for r in results])
    assert report['kv_audit']['free'] == engine.pool.capacity and report['host_swap_used_bytes'] == 0
    return report


def swap_check(args):
    """Production-weight forced swap with the same batch composition."""
    prompts = workload_prompts()[:4]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine('models/TaH-plus-1.7B', override, max_batch_size=4, kv_capacity=10000,
                      swap_chunk_entries=7, prefill_chunk_size=128)
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    requests = [Request(p, 128, i) for i, p in enumerate(prompts)]
    expected = engine.generate(requests, params)
    checks = []
    def hook(engine, running, swapped, free_rows):
        if engine.last_run['steps'] != 32:
            return
        state = running[-1]
        original_state = (state.position, state.depth, state.generated.copy(), state.generated_counts.copy())
        engine._swap_out(state, running, swapped, free_rows)
        host_bytes = engine.swap.used_bytes
        assert host_bytes > 0 and state.cache is None
        engine._restore(running, swapped, free_rows)
        assert not swapped and engine.swap.used_bytes == 0
        assert original_state == (state.position, state.depth, state.generated, state.generated_counts)
        checks.append(dict(position=state.position, depth=state.depth, host_bytes=host_bytes))
    engine._step_hook = hook
    actual = engine.generate(requests, params)
    assert len(checks) == 1
    assert [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in actual] == [
        (r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in expected]
    return dict(status='PASS', gate='G-ROBUST forced swap identity', requests=len(requests),
                output_tokens=sum(len(r.token_ids) for r in actual), checks=checks,
                run=engine.last_run, kv_audit=engine.pool.audit(), host_swap_used_bytes=engine.swap.used_bytes)


def profile(args):
    prompts = workload_prompts()
    if args.profile_context:
        prompts = [(p * (args.profile_context // len(p) + 1))[:args.profile_context] for p in prompts]
    override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
    engine = TaHEngine('models/TaH-plus-1.7B', override, max_batch_size=args.batch_size,
                      prefill_chunk_size=128, profile_steps=0, admission_headroom=0)
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    requests = [Request(prompts[i % len(prompts)], 12, i) for i in range(args.batch_size)]
    engine.generate(requests, params)
    original = engine._decode
    from torch.profiler import record_function
    def decode(running):
        with record_function('engine_decode_step'):
            return original(running)
    engine._decode = decode
    # Uninstrumented timing samples provide the step wall envelope before the
    # profiler's CPU instrumentation increases launch overhead.
    engine.profile_steps = 100
    engine.generate(requests, params)
    unprofiled = engine.last_run['profile']
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
    started = False
    closed = False
    count = 0
    def profiled_decode(running):
        nonlocal started, closed, count
        # Starting at the first decode avoids recording the large, irrelevant
        # prefill trace (and keeps the profiler's host memory bounded).
        if not started:
            prof.__enter__()
            started = True
        result = decode(running)
        count += 1
        if count == 11:
            prof.__exit__(None, None, None)
            closed = True
            engine._decode = decode
        return result
    engine._decode = profiled_decode
    try:
        engine.generate(requests, params)
    finally:
        if started and not closed:
            prof.__exit__(None, None, None)
    steps = [e for e in prof.events() if e.name == 'engine_decode_step'
             and e.device_type == torch.autograd.DeviceType.CPU]
    def covered(start, end, scopes):
        return any(a <= start and end <= b for a, b in scopes)
    scopes = [(e.time_range.start, e.time_range.end) for e in steps]
    # CPU inclusive device totals can duplicate activities (including an
    # "Unrecognized" CPU event) and include GPU user-annotation envelopes.
    # Count actual CUDA events once; annotation spans are NOT kernels.
    device_events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
                     and not e.is_user_annotation and covered(e.time_range.start, e.time_range.end, scopes)]
    gpu_us = sum(e.device_time_total for e in device_events)
    def is_copy(name):
        return 'memcpy' in name.lower() or 'memset' in name.lower()
    kernel_us = sum(e.device_time_total for e in device_events if not is_copy(e.name))
    transfer_us = gpu_us - kernel_us
    wall_us = sum(s.cpu_time_total for s in steps)
    assert 0 < kernel_us <= gpu_us <= wall_us
    unprofiled = unprofiled[:len(steps)]
    observed = engine.last_run['profile'][:len(steps)]
    assert [p['batch_size'] for p in unprofiled] == [p['batch_size'] for p in observed]
    unprofiled_mean = statistics.mean(p['total_ms'] for p in unprofiled)
    grouped = {}
    for event in device_events:
        grouped[event.name] = grouped.get(event.name, 0.) + event.device_time_total
    trace_path = args.output.with_suffix('.trace.json')
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())['traceEvents']
    trace_steps = [(e['ts'], e['ts'] + e['dur']) for e in trace
                   if e.get('cat') == 'user_annotation' and e.get('name') == 'engine_decode_step']
    assert len(trace_steps) == len(steps)
    trace_device = [e for e in trace if e.get('cat') in ('kernel', 'gpu_memcpy', 'gpu_memset')
                    and covered(e['ts'], e['ts'] + e['dur'], trace_steps)]
    trace_gpu_us = sum(e['dur'] for e in trace_device)
    assert len(trace_device) == len(device_events)
    assert abs(trace_gpu_us - gpu_us) < .1, (trace_gpu_us, gpu_us)
    audit = trace_profile_totals(trace)
    assert audit['steps'] == len(steps)
    assert abs(audit['kernel_us'] - kernel_us) < .1
    assert abs(audit['wall_us'] - wall_us) < .1
    report = dict(status='PASS', batch_size=args.batch_size, context=args.profile_context,
        context_source='repeated real workload prompts; actual model prefill', admission_headroom=0,
        observed_batch_sizes=[p['batch_size'] for p in engine.last_run['profile'][:len(steps)]], decode_steps=len(steps),
        mean_step_wall_ms=wall_us / len(steps) / 1000,
        mean_kernel_gpu_ms=kernel_us / len(steps) / 1000,
        mean_gpu_copy_ms=transfer_us / len(steps) / 1000,
        mean_python_launch_and_gpu_idle_ms=(wall_us - gpu_us) / len(steps) / 1000,
        profiler_overhead_included=True, event_profile=engine.last_run['profile'],
        unprofiled_mean_step_ms=unprofiled_mean,
        estimated_unprofiled_python_launch_idle_ms=unprofiled_mean - gpu_us / len(steps) / 1000,
        unprofiled_event_profile=unprofiled,
        verification=dict(actual_cuda_device_us=gpu_us, trace_cuda_device_us=trace_gpu_us,
                          actual_cuda_activities=len(device_events), kernel_us=kernel_us,
                          cpu_step_scopes=len(steps), source_trace_sha256=digest(trace_path),
                          discarded_cpu_inclusive_device_us=sum(s.device_time_total for s in steps),
                          trace_file=str(trace_path)),
        top_gpu_kernels=[dict(name=name, total_gpu_ms=duration / 1000)
            for name, duration in sorted(grouped.items(), key=lambda pair: pair[1], reverse=True)[:20]],
        note='GPU kernel sum is device execution time; stream events include CPU launch gaps. The unprofiled residual uses device durations from an equivalent profiled run and is an estimate. CPU and GPU overlap; residuals are not additive CPU utilization. The profiled residual includes profiler overhead.')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gate', choices=['gen', 'tf', 'perf', 'robust', 'long-prompt', 'swap', 'profile'], required=True)
    parser.add_argument('--checkpoint', choices=['real', 'mlp3', 'mlp1', 'label3'], default='real')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int, default=4096)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=128)
    parser.add_argument('--memory-fraction', type=float, default=.9)
    parser.add_argument('--profile-context', type=int, default=2048)
    args = parser.parse_args()
    torch.set_num_threads(2)
    start = time.perf_counter()
    report = (parity(args) if args.gate in ('gen', 'tf') else profile(args) if args.gate == 'profile'
              else swap_check(args) if args.gate == 'swap' else workload(args))
    report.update(process_seconds=time.perf_counter() - start, device=torch.cuda.get_device_name(),
        runtime_sha256={str(p): digest(p) for p in Path('tah/engine').glob('*.py')})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print({k: v for k, v in report.items() if k not in ('runtime_sha256', 'results', 'comparisons', 'noise_limits', 'engine_run')}, flush=True)
    if report['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
