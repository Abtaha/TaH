"""CUDA engine state, unsplit static plans and the single-status-sync bound."""
import json
import os

import pytest
import torch

from tah.engine import TaHEngine, Request, SamplingParams
from tah.engine.attention import AttentionRow, DecodeBucket
from tah.engine.kv_cache import SlotPool
from tah.engine.tests.test_model import reference

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get('CUDA_VISIBLE_DEVICES') == '' or not torch.cuda.is_available(), reason='CUDA required')]


def test_static_plan_grows_ragged_context_without_replan():
    pool = SlotPool(4096, 1, 2, 128, 3, device='cuda')
    requests = [pool.create_request(i, context_limit=1024) for i in range(3)]
    bucket = DecodeBucket(pool, 3, pool.capacity, 4, static_plan=True)
    assert bucket.wrapper._int_workspace_buffer.numel() == 4096
    assert bucket.wrapper._pin_memory_int_workspace_buffer.numel() == 4096
    def fail(*args, **kwargs):
        raise AssertionError('decode replanned a static bucket')
    bucket.wrapper.plan = fail
    torch.manual_seed(5)
    pool.values.normal_()
    q = torch.randn(3, 4, 128, device='cuda', dtype=torch.bfloat16)
    for length in [1, 3, 127, 256, 1024]:
        rows = []
        for i, req in enumerate(requests):
            target = max(1, length - 17 * i)
            for d in range(3):
                positions = range(req.streams[d].size, target) if d == 0 else []
                pool.allocate(req, d, positions)
            rows.append(AttentionRow(req, target - 1, 1))
        bucket.prepare(rows)
        actual = bucket.run(q, 0).clone()
        # Component oracle uses independent paged decode plans, already
        # established against dense SDPA in WP4; retain its numeric tolerance.
        oracle = DecodeBucket(pool, 3, pool.capacity, 4)
        oracle.prepare(rows)
        torch.testing.assert_close(actual, oracle.run(q, 0), atol=.015, rtol=.02)
        del oracle
    for req in requests:
        pool.release(req)
    assert pool.audit()['free'] == pool.capacity


def test_cuda_engine_forced_swap_and_single_sync(tmp_path, monkeypatch):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    # Long enough tiny context for short generation, without changing runtime.
    cfg = json.loads((tmp_path / 'config.json').read_text())
    cfg['max_position_embeddings'] = 256
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, max_batch_size=3, kv_capacity=128,
                    prefill_chunk_size=2, swap_chunk_entries=2, eos_token_id=None)
    requests = [Request([3, 7], 8, 81), Request([5, 8], 8, 12)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    eng.cuda_graphs = False
    expected = eng.generate(requests, params)
    eng.cuda_graphs = True
    exercised = []
    def hook(engine, running, swapped, free_rows):
        if engine.last_run['steps'] == 2:
            state = running[-1]
            engine._swap_out(state, running, swapped, free_rows)
            assert engine.swap.used_bytes > 0
            engine._restore(running, swapped, free_rows)
            assert not swapped
            exercised.append(True)
    eng._step_hook = hook
    from tah.engine.engine import StepBucket
    original_run = StepBucket.run
    def run(bucket, engine, rotary):
        def fail(*args, **kwargs):
            raise AssertionError('host scalar/read or dynamic shape inside CUDA step')
        with monkeypatch.context() as m:
            for name in ['item', 'tolist', 'cpu', 'numpy', '__bool__', '__int__', 'nonzero']:
                m.setattr(torch.Tensor, name, fail)
            return original_run(bucket, engine, rotary)
    monkeypatch.setattr(StepBucket, 'run', run)
    actual = eng.generate(requests, params)
    assert exercised == [True]
    assert [(r.token_ids, r.generated_pass_counts) for r in actual] == [(r.token_ids, r.generated_pass_counts) for r in expected]
    assert eng.swap.used_bytes == 0 and eng.pool.audit()['owned'] == 0
    # Audit the ENTIRE decode including planning for hidden local_scalar syncs.
    eng._step_hook = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        eng.generate(requests, params)
    scalar = [e for e in prof.events() if e.name == 'aten::_local_scalar_dense']
    # Prefill's validation/ragged bookkeeping is outside the decode bound.
    # Explicit per-decode inspection below includes host->device allocation,
    # static planning, dynamic RoPE, all layers, sampling and status transfer.
    from torch.profiler import record_function
    original_decode = eng._decode
    def decode(running):
        with record_function('engine_decode_step'):
            return original_decode(running)
    eng._decode = decode
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
        eng.generate(requests, params)
    def in_decode(event):
        parent = event.cpu_parent
        while parent:
            if parent.name == 'engine_decode_step':
                return True
            parent = parent.cpu_parent
        return False
    scalars = [e for e in prof.events() if e.name == 'aten::_local_scalar_dense' and in_decode(e)]
    assert not scalars
    # The sole blocking status copy produces one cudaStreamSynchronize each.
    steps = [e for e in prof.events() if e.name == 'engine_decode_step']
    for step in steps:
        syncs = [e for e in prof.events() if e.name in ('cudaStreamSynchronize', 'cudaDeviceSynchronize') and in_decode(e) and step.time_range.start <= e.time_range.start < step.time_range.end]
        assert len(syncs) <= 1, [(e.name, e.time_range) for e in syncs]
    assert eng.last_run['graph_steps'] == eng.last_run['steps'] > 0


def test_compile_fullgraph_eager_and_cuda_graph_identity(tmp_path):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, max_batch_size=3, kv_capacity=256,
                    prefill_chunk_size=2, eos_token_id=None,
                    graph_batch_sizes=[1, 2, 3], compile=True,
                    compile_cache_dir=tmp_path / 'inductor-cache')
    assert eng.compiled_step.compile_graphs >= 1
    assert eng.memory_budget['graphs']['compile_seconds'] >= 0
    assert (tmp_path / 'inductor-cache').is_dir()
    requests = [Request([3, 7], 8, 81), Request([5, 8], 8, 12)]
    params = SamplingParams(temperature=.6, top_k=20, top_p=.95)
    eng.cuda_graphs = False
    eager = eng.generate(requests, params)
    eager_trace = [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in eager]
    assert eng.last_run['compile']['enabled'] is True
    assert eng.last_run['eager_steps'] > 0
    eng.cuda_graphs = True
    graphed = eng.generate(requests, params)
    graph_trace = [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in graphed]
    assert graph_trace == eager_trace
    assert eng.last_run['graph_steps'] == eng.last_run['steps'] > 0
    assert eng.pool.audit()['owned'] == 0 and eng.swap.used_bytes == 0
    general = SamplingParams(temperature=.6, top_k=0, top_p=.95)
    eng.cuda_graphs = False
    general_eager = eng.generate(requests, general)
    eng.cuda_graphs = True
    general_fallback = eng.generate(requests, general)
    assert [(r.token_ids, r.generated_pass_counts) for r in general_fallback] == [
        (r.token_ids, r.generated_pass_counts) for r in general_eager]
    assert eng.last_run['eager_steps'] > 0  # general sampling has no WP7 CUDA graph capture
    assert eng.last_run['compile']['graphs'] > 0


@pytest.mark.parametrize('rope', [None, {'rope_type': 'dynamic', 'factor': 2.},
                                 {'rope_type': 'longrope', 'factor': 2.,
                                  'short_factor': [1.] * 64, 'long_factor': [2.] * 64}])
def test_graph_shared_pool_arbitrary_order_rotary_params_and_fallback(tmp_path, rope):
    reference(tmp_path, max_iter=3, label=False, head_dim=128, rope=rope)
    cfg = json.loads((tmp_path / 'config.json').read_text())
    cfg['max_position_embeddings'] = 128
    if rope and rope['rope_type'] == 'longrope':
        cfg['original_max_position_embeddings'] = 4
    (tmp_path / 'config.json').write_text(json.dumps(cfg))
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, max_batch_size=3, kv_capacity=256, prefill_chunk_size=2,
                    eos_token_id=None, graph_batch_sizes=[1, 3])
    assert set(eng.graphs.info['batch_sizes']) == {1, 3}
    assert eng.memory_budget['reservations']['cuda_graphs'] >= eng.graphs.info['private_reserved_bytes']
    trace = []
    decode = eng._decode
    def observed(running):
        result = decode(running)
        trace.extend((s.index, s.position, s.depth, bool(row[0])) for s, row in zip(running, result))
        return result
    eng._decode = observed
    for size, params in [(1, dict(temperature=0)), (3, dict(temperature=.6, top_k=20)),
                         (2, dict(temperature=.8, top_k=7, top_p=.7)),
                         (3, dict(temperature=.6, top_k=0)), (1, dict(temperature=0)),
                         (3, dict(temperature=.3, top_k=12, min_p=.1)),
                         (3, [dict(temperature=0), dict(temperature=.4, top_k=7),
                              dict(temperature=.8, top_k=20)])]:
        requests = [Request([3 + i, 7], 7 - i, 91 + i) for i in range(size)]
        eng.cuda_graphs = False
        trace.clear()
        expected = eng.generate(requests, params)
        expected_trace = trace.copy()
        trace.clear()
        eng.cuda_graphs = True
        actual = eng.generate(requests, params)
        assert trace == expected_trace
        assert [(r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in actual] == [
            (r.token_ids, r.prompt_pass_counts, r.generated_pass_counts) for r in expected]
        if size == 2 or isinstance(params, dict) and params.get('top_k') == 0:
            assert eng.last_run['eager_steps'] > 0
        else:
            assert eng.last_run['graph_steps'] > 0
        assert eng.pool.audit()['owned'] == 0 and eng.swap.used_bytes == 0


def test_graph_natural_swap_fifo_admission_identity(tmp_path):
    reference(tmp_path, max_iter=3, label=False, head_dim=128)
    torch.set_num_threads(2)
    eng = TaHEngine(tmp_path, max_batch_size=3, kv_capacity=50, prefill_chunk_size=2,
                    swap_chunk_entries=3, eos_token_id=None)
    requests = [Request([3, 7], 12, i) for i in range(4)]
    params = SamplingParams(temperature=0)
    eng.cuda_graphs = False
    expected = eng.generate(requests, params)
    restored, admissions = [], []
    restore, prefill = eng.swap.restore, eng._prefill
    def observed_restore(state, *args):
        restored.append(state.index)
        return restore(state, *args)
    def observed_prefill(state):
        admissions.append((state.index, eng.swap.used_bytes))
        return prefill(state)
    eng.swap.restore, eng._prefill = observed_restore, observed_prefill
    eng.cuda_graphs = True
    actual = eng.generate(requests, params)
    assert [(r.token_ids, r.generated_pass_counts) for r in actual] == [(r.token_ids, r.generated_pass_counts) for r in expected]
    assert restored[0] == 2 and all(used == 0 for _, used in admissions)
    assert eng.last_run['blocked_admission_steps'] > 0 and eng.last_run['graph_steps'] > 0
    assert eng.pool.audit()['free'] == 50 and eng.swap.used_bytes == 0
