"""TaH generation: bounded prefill, graphed mixed-depth steps and swap scheduling.

DecodeBucket and step buffers have fixed addresses/shapes for each exact B.
Admission, slot allocation, descriptor staging and RoPE frequency planning happen
outside the device-only step. One compact status transfer completes each step.
"""
from collections import deque
import json
import math
from pathlib import Path
import time

import torch

from .attention import AttentionRow, DecodeBucket, static_plan_integer_bytes
from .kv_cache import KVCapacityError, SlotPool
from .model import EagerTaHModel
from .sampling import GraphSampler, RequestRNG, SamplingParams
from .scheduler import HostSwap, Request, Result, State
from .tensors import upload
from .graphs import DecodeGraphs, batch_sizes
from .compiled import CompiledDecodeStep, register_attention_plan


class StepBucket:
    def __init__(self, engine, size):
        self.size = size
        self.attention = DecodeBucket(engine.pool, size, engine.pool.capacity,
            engine.model.config['num_attention_heads'], workspace=engine.decode_workspace,
            static_plan=True, buffers=engine.attention_buffers)
        self.rows = engine.step_rows[:size]
        self.positions = engine.step_positions[:size]
        self.depths = engine.step_depths[:size]
        self.slots = engine.step_slots[:size]
        self.embeddings = engine.step_embeddings[:size]
        self.logits = engine.step_logits[:size]
        self.rng = RequestRNG(engine.step_seeds[:size], engine.step_counters[:size])
        self.params = None
        self.sampler = None
        self.sampling_buckets = {}
        self.rotary = tuple(torch.empty(1, size, engine.model.config['head_dim'],
            device=engine.device, dtype=engine.model.dtype) for _ in range(2))
        self.attention_key = register_attention_plan(self.attention) if engine.compile_enabled else None

    def set_rotary(self, rotary):
        for target, source in zip(self.rotary, rotary):
            target.copy_(source)

    def sampling(self, engine, params):
        # Only the immutable sampling path determines a bucket specialization.
        key = (all(p.greedy for p in params), all(p.greedy or 1 <= p.top_k <= engine.sampling_topk for p in params))
        if engine.cuda_graphs and self.size in engine.graph_batch_sizes and key[1]:
            # The bounded sampler already handles greedy rows with argmax and
            # zero RNG advance. One capture per B avoids duplicating every
            # model node/executable for a separate greedy specialization.
            key = (False, True)
        self.sampling_key = key
        if key not in self.sampling_buckets:
            sampler = GraphSampler(engine.model.config['vocab_size'],
                max_top_k=engine.sampling_topk if key[1] else None, greedy_only=key[0])
            batch = sampler.make_batch(params, device=engine.device)
            self.sampling_buckets[key] = (sampler, batch)
        else:
            self.sampling_buckets[key][1].update(params)
        self.sampler, self.params = self.sampling_buckets[key]

    @torch.inference_mode()
    def run(self, engine, rotary):
        """The device-only step boundary WP7 can capture without rescheduling."""
        if engine.compile_enabled:
            return engine.compiled_step.run(self, engine, rotary)
        model = engine.model
        output = model.forward_cached(self.embeddings, self.positions, self.depths,
                                      self.slots, engine.pool, self.attention, rotary=rotary)
        self.logits.add_(output.logits)  # BF16 cumulative additions
        keep, raw, probability = engine.decider(self.logits, self.depths, output.hidden_states,
                                               hidden_indices=engine.decider_indices)
        keep = keep & (self.depths < model.max_iter)
        probability = torch.where(self.depths < model.max_iter, probability, 0.)
        sampled = self.sampler.sample(self.logits, self.params, self.rng, active=~keep)
        if model.max_iter > 1:
            values, indices = self.logits.topk(min(model.input_topk, self.logits.shape[-1]), dim=-1)
            soft = (values.softmax(-1)[..., None] * model.model.embed_tokens.weight[indices]).sum(-2).to(model.dtype)
            next_input = torch.where(keep[:, None], soft,
                model.model.embed_tokens(sampled.clamp_min(0), False))
        else:
            next_input = model.model.embed_tokens(sampled.clamp_min(0), False)
        engine.logits.index_copy_(0, self.rows, torch.where(keep[:, None], self.logits, 0.))
        engine.embeddings.index_copy_(0, self.rows, next_input)
        engine.counters.index_copy_(0, self.rows, self.rng.counters)
        # One transfer for control/output/observations; integers fit FP64 exactly.
        status = torch.stack((keep.double(), sampled.double(), raw.double(), probability.double()), -1)
        return status


class TaHEngine:
    def __init__(self, model_path, config_override=None, *, device='cuda',
                 max_batch_size=128, prefill_chunk_size=128, memory_fraction=.9,
                 kv_capacity=None, admission_headroom=.1, host_swap_bytes=8 * 1024**3,
                 swap_chunk_entries=256, max_new_tokens=1024, seed=0,
                 eos_token_id='tokenizer', sampling_topk=128, compile=False,
                 profile_steps=0, cuda_graphs=True, graph_batch_sizes=None,
                 compile_cache_dir=None):
        if not isinstance(compile, bool):
            raise ValueError('compile must be boolean')
        if compile and torch.device(device).type != 'cuda':
            raise ValueError('compile=True requires a CUDA device; use compile=False for CPU development')
        for name, value in [('max_batch_size', max_batch_size), ('prefill_chunk_size', prefill_chunk_size),
                            ('swap_chunk_entries', swap_chunk_entries), ('sampling_topk', sampling_topk)]:
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not 0 < memory_fraction <= 1 or not 0 <= admission_headroom < 1:
            raise ValueError('memory_fraction must be in (0,1], admission_headroom in [0,1)')
        if max_new_tokens < 0 or not isinstance(max_new_tokens, int) or profile_steps < 0:
            raise ValueError('max_new_tokens and profile_steps must be nonnegative integers')
        if not isinstance(cuda_graphs, bool):
            raise ValueError('cuda_graphs must be boolean')
        self.graph_batch_sizes = batch_sizes(max_batch_size, graph_batch_sizes)
        self.model = EagerTaHModel.from_pretrained(model_path, config_override, device=device)
        self.device = self.model.device
        self.compile_enabled = compile
        self.cuda_graphs = cuda_graphs and self.device.type == 'cuda' and bool(self.graph_batch_sizes)
        self.graphs = DecodeGraphs()
        self.max_batch_size, self.prefill_chunk_size = max_batch_size, prefill_chunk_size
        self.max_new_tokens, self.seed, self.sampling_topk = max_new_tokens, seed, sampling_topk
        self.admission_headroom, self.profile_steps = admission_headroom, profile_steps
        self.decider = self.model.eval_iter_decider or self.model.iter_decider
        self.decider_indices = upload(self.decider.selected_layers(len(self.model.model.layers)), self.device) if hasattr(self.decider, 'selected_layers') else None
        self.compiled_step = CompiledDecodeStep(self, compile_cache_dir) if compile else None
        self.context_limit = self.model.config['max_position_embeddings']
        self.tokenizer = None
        root = Path(model_path)
        if (root / 'tokenizer.json').exists():
            from tokenizers import Tokenizer
            self.tokenizer = Tokenizer.from_file(str(root / 'tokenizer.json'))
        if eos_token_id == 'tokenizer':
            cfg = json.loads((root / 'tokenizer_config.json').read_text()) if (root / 'tokenizer_config.json').exists() else {}
            eos = cfg.get('eos_token')
            if isinstance(eos, dict):
                eos = eos.get('content')
            self.eos_token_id = self.tokenizer.token_to_id(eos) if self.tokenizer is not None and eos else None
        else:
            self.eos_token_id = eos_token_id
        if self.eos_token_id is not None and (not isinstance(self.eos_token_id, int) or not 0 <= self.eos_token_id < self.model.config['vocab_size']):
            raise ValueError('eos_token_id must be a vocabulary ID, None, or "tokenizer"')
        cfg, b, chunk = self.model.config, max_batch_size, prefill_chunk_size
        v, h, n, inter = cfg['vocab_size'], cfg['hidden_size'], cfg['num_hidden_layers'], cfg['intermediate_size']
        slot_bytes = 4 * n * cfg['num_key_value_heads'] * cfg['head_dim']
        # Every future exact-B attention object's private FlashInfer integer
        # workspace is budgeted; float scratch and ragged staging are shared.
        static_integer_bytes = sum(static_plan_integer_bytes(size) for size in range(1, b + 1))
        reserves = dict(flashinfer_float=2 * 128 * 1024**2 if self.device.type == 'cuda' else 0,
            flashinfer_integer=8 * 1024**2 + static_integer_bytes if self.device.type == 'cuda' else 0,
            state_and_step=4 * b * (v + h) + 256 * b,
            model_transients=max(b, chunk) * (8 * v + 2 * (n + 16) * h + 12 * inter + 4 * min(v, self.model.input_topk) * h),
            general_sampler=80 * b * v,
            swap_scratch=min(swap_chunk_entries, kv_capacity or swap_chunk_entries) * slot_bytes,
            allocator_margin=512 * 1024**2 if self.device.type == 'cuda' else 0)
        self._busy = False
        self._observer = None
        self._step_hook = None
        self.last_run = {}
        probe = None
        if self.cuda_graphs:
            # Measure the same model/step shapes with a tiny scratch KV pool.
            # Destroy these graphs before allocating the real KV pool: captured
            # pointers must refer to its final, immovable storage.
            self._allocate_runtime(b, host_swap_bytes, swap_chunk_entries)
            self.graphs.capture(self)
            probe = self.graphs.info.copy()
            before_close = torch.cuda.mem_get_info(self.device)[0]
            self.graphs.close()
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            probe['released_device_bytes'] = max(0, torch.cuda.mem_get_info(self.device)[0] - before_close)
            self._release_runtime()
            self.graphs = DecodeGraphs()
            # Driver/module allocations retained after reset are already
            # excluded by the fresh free-memory reading below. Reserve only
            # storage/executables that will be allocated again for final graphs.
            reserves['cuda_graphs'] = max(probe['released_device_bytes'], probe['private_reserved_bytes'])
        free = None
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            free, _ = torch.cuda.mem_get_info(self.device)
            # Metadata costs 20 bytes/entry (slots,pos,depth,indices) + geometric
            # stream tables <=24 bytes/entry. No longest-request padding.
            available = int(free * memory_fraction) - sum(reserves.values())
            maximum = available // (slot_bytes + 44)
            if kv_capacity is None:
                kv_capacity = maximum
            if kv_capacity > maximum:
                raise KVCapacityError(f'KV capacity {kv_capacity} exceeds measured safe capacity {maximum}; '
                                      'reduce capacity/batch/chunk sizes or free GPU memory')
        elif kv_capacity is None:
            raise ValueError('CPU development engine requires an explicit kv_capacity')
        if not isinstance(kv_capacity, int) or isinstance(kv_capacity, bool) or kv_capacity < b:
            raise KVCapacityError('KV budget is too small; decrease max_batch_size or reserve less workspace')
        self.memory_budget = dict(measured_free_bytes=free, memory_fraction=memory_fraction,
            reservations=reserves, kv_capacity=kv_capacity, kv_bytes=kv_capacity * slot_bytes,
            metadata_budget_bytes=kv_capacity * 44, host_flashinfer_workspace_bytes=static_integer_bytes if self.device.type == 'cuda' else 0)
        self._allocate_runtime(kv_capacity, host_swap_bytes, swap_chunk_entries)
        if self.cuda_graphs:
            self.graphs.capture(self)
            self.memory_budget['graph_probe'] = probe
            self.memory_budget['graphs'] = self.graphs.info.copy()
            self.memory_budget['total_capture_seconds'] = probe['capture_seconds'] + self.graphs.info['capture_seconds']
            overhead = max(self.graphs.info['device_overhead_bytes'], self.graphs.info['private_reserved_bytes'])
            if overhead > reserves['cuda_graphs'] + reserves['allocator_margin']:
                raise KVCapacityError('graph capture exceeded its measured memory reservation; '
                                      'reduce memory_fraction or use cuda_graphs=False')
        if self.compiled_step is not None:
            self.memory_budget['compile'] = dict(enabled=True,
                cache_dir=self.compiled_step.cache_dir,
                seconds=self.compiled_step.compile_seconds,
                graphs=self.compiled_step.compile_graphs)

    def _release_runtime(self):
        self.buckets.clear()
        for name in self._runtime_attributes:
            setattr(self, name, None)
        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()

    def _allocate_runtime(self, kv_capacity, host_swap_bytes, swap_chunk_entries):
        before = set(self.__dict__)
        cfg, b, n = self.model.config, self.max_batch_size, self.model.config['num_hidden_layers']
        v, h = cfg['vocab_size'], cfg['hidden_size']
        self.pool = SlotPool(kv_capacity, n, cfg['num_key_value_heads'], cfg['head_dim'],
                             self.model.max_iter, device=self.device)
        self.swap = HostSwap(self.pool, host_swap_bytes, swap_chunk_entries)
        self.decode_workspace = self.prefill_workspace = None
        if self.device.type == 'cuda':
            self.decode_workspace = torch.empty(128 * 1024**2, dtype=torch.uint8, device=self.device)
            self.prefill_workspace = torch.empty_like(self.decode_workspace)
        def empty(*shape, dtype=torch.long):
            return torch.empty(*shape, device=self.device, dtype=dtype)
        self.logits = empty(b, v, dtype=self.model.dtype)
        self.embeddings = empty(b, h, dtype=self.model.dtype)
        self.seeds, self.counters = empty(b), empty(b)
        self.device_state = (self.logits, self.embeddings, self.seeds, self.counters)
        self.step_logits, self.step_embeddings = torch.empty_like(self.logits), torch.empty_like(self.embeddings)
        self.step_rows, self.step_positions, self.step_depths, self.step_slots = [empty(b) for _ in range(4)]
        self.step_seeds, self.step_counters = empty(b), empty(b)
        cap = self.pool.capacity
        self.attention_buffers = {name: empty(cap, dtype=dtype) for name, dtype in
            [('source_slots', torch.int32), ('source_pos', torch.long), ('source_depth', torch.int32), ('indices', torch.int32)]}
        self.attention_buffers.update({name: empty(b, dtype=dtype) for name, dtype in
            [('starts', torch.int32), ('lengths', torch.int32), ('query_pos', torch.long), ('query_depth', torch.int32), ('counts', torch.int32)]})
        self.attention_buffers.update(indptr=empty(b + 1, dtype=torch.int32),
            output=empty(b, cfg['num_attention_heads'], cfg['head_dim'], dtype=self.model.dtype),
            descriptors=empty(b, 3 * self.model.max_iter + 4))
        self.buckets = {}
        if not hasattr(self, '_runtime_attributes'):
            self._runtime_attributes = set(self.__dict__) - before

    def _bucket(self, size):
        if size not in self.buckets:
            self.buckets[size] = StepBucket(self, size)
        return self.buckets[size]

    def _requests(self, requests, sampling):
        params = sampling if isinstance(sampling, (list, tuple)) else [sampling] * len(requests)
        if len(params) != len(requests):
            raise ValueError('sampling must be shared or have one entry per request')
        states = []
        for index, (item, param) in enumerate(zip(requests, params)):
            req = item if isinstance(item, Request) else Request(item, self.max_new_tokens, self.seed + index)
            if isinstance(req.max_new_tokens, bool) or not isinstance(req.max_new_tokens, int) or req.max_new_tokens < 0:
                raise ValueError('request max_new_tokens must be a nonnegative integer')
            RequestRNG.from_seeds([req.seed])  # host-only seed validation
            param = SamplingParams(**param) if isinstance(param, dict) else param
            if not isinstance(param, SamplingParams):
                raise TypeError('sampling requires SamplingParams or parameter dictionaries')
            if isinstance(req.prompt, str):
                if self.tokenizer is None:
                    raise ValueError('string requests require tokenizer.json in the checkpoint')
                prompt = self.tokenizer.encode(req.prompt).ids
            else:
                prompt = list(req.prompt)
            if not prompt or any(isinstance(t, bool) or not isinstance(t, int) or not 0 <= t < self.model.config['vocab_size'] for t in prompt):
                raise ValueError('requests require nonempty vocabulary token ID lists')
            if len(prompt) + max(0, req.max_new_tokens - 1) > self.context_limit:
                raise KVCapacityError('request prompt + generated context exceeds model context limit')
            prefill_bound = len(prompt) * self.model.max_iter
            if prefill_bound + math.ceil(self.pool.capacity * self.admission_headroom) > self.pool.capacity:
                raise KVCapacityError('request worst-case prefill cannot fit KV pool with admission headroom')
            states.append(State(index, prompt, param, req.max_new_tokens, req.seed, time.perf_counter()))
        return states

    def _prefill(self, state):
        start_time = time.perf_counter()
        state.cache = self.pool.create_request(state.index, context_limit=self.context_limit)
        state.admitted = start_time
        self.seeds[state.row] = state.seed
        self.counters[state.row] = 0
        for start in range(0, len(state.prompt), self.prefill_chunk_size):
            end = min(start + self.prefill_chunk_size, len(state.prompt))
            # Admission reserved the ENTIRE worst-case prompt before any pass.
            output = self.model.forward_chunk(state.prompt[start:end], self.pool, state.cache,
                start_position=start, workspace=self.prefill_workspace)
            state.prompt_counts.extend(output.iter_count.cpu().tolist())
            if self._observer is not None:
                self._observer.prefill(state, start, output)
            if end == len(state.prompt):
                self.logits[state.row].copy_(output.logits[-1])
            del output
        state.prefill_seconds = time.perf_counter() - start_time
        state.position = len(state.prompt)
        if not state.max_new_tokens:
            return 'length'
        bucket = self._bucket(1)
        bucket.sampling(self, [state.params])
        bucket.rng.seeds.copy_(self.seeds[state.row:state.row + 1])
        bucket.rng.counters.copy_(self.counters[state.row:state.row + 1])
        token = int(bucket.sampler.sample(self.logits[state.row:state.row + 1], bucket.params, bucket.rng)[0].cpu())
        self.counters[state.row].copy_(bucket.rng.counters[0])
        reason = self._sampled(state, token, state.prompt_counts[-1])
        if reason is None:
            self.logits[state.row].zero_()
            self.embeddings[state.row].copy_(self.model.model.embed_tokens(torch.tensor(token, device=self.device), False))
        return reason

    def _sampled(self, state, token, count):
        if token == self.eos_token_id:
            return 'eos'
        state.generated.append(token)
        state.generated_counts.append(count)
        return 'length' if len(state.generated) >= state.max_new_tokens else None

    def _finish(self, state, reason, results, callback):
        finished = time.perf_counter()
        result = Result(state.index, state.generated.copy(),
            self.tokenizer.decode(state.generated, skip_special_tokens=False) if self.tokenizer else '',
            state.prompt_counts.copy(), state.generated_counts.copy(), reason,
            dict(total_seconds=finished - state.submitted, queue_seconds=state.admitted - state.submitted,
                 prefill_seconds=state.prefill_seconds, swaps=state.swaps))
        self.pool.release(state.cache)
        state.cache = None
        results[state.index] = result
        if callback is not None:
            callback(result)

    def _swap_out(self, state, running, swapped, free_rows):
        started = time.perf_counter()
        self.swap.out(state, self.device_state)
        running.remove(state)
        free_rows.append(state.row)
        state.row = -1
        swapped.append(state)
        self.last_run['swap_outs'] += 1
        self.last_run['time_seconds']['swap'] += time.perf_counter() - started

    def _restore(self, running, swapped, free_rows):
        # FIFO, and reserve growth for every resident row before restoration.
        while swapped and free_rows:
            state = swapped[0]
            needed = sum(map(len, state.snapshot.positions)) + len(running) + 1
            if needed > self.pool.free_slots:
                break
            state.row = free_rows.pop()
            started = time.perf_counter()
            self.swap.restore(state, self.device_state, self.context_limit)
            self.last_run['time_seconds']['swap'] += time.perf_counter() - started
            swapped.popleft()
            running.append(state)
            running.sort(key=lambda s: s.index)
            self.last_run['swap_ins'] += 1

    @torch.inference_mode()
    def _decode(self, running):
        start = time.perf_counter()
        size = len(running)
        bucket = self._bucket(size)
        profiled = self.device.type == 'cuda' and self.last_run['steps'] < self.profile_steps
        events = [torch.cuda.Event(enable_timing=True) for _ in range(4)] if profiled else None
        if events:
            events[0].record()
        rows, slots = [], []
        for state in running:
            slots.append(self.pool.allocate(state.cache, state.depth - 1, [state.position]))
            rows.append(AttentionRow(state.cache, state.position, state.depth))
        for target, values in [(bucket.rows, [s.row for s in running]),
                               (bucket.positions, [s.position for s in running]),
                               (bucket.depths, [s.depth for s in running])]:
            target.copy_(upload(values, self.device))
        bucket.slots.copy_(torch.cat(slots))
        torch.index_select(self.embeddings, 0, bucket.rows, out=bucket.embeddings)
        torch.index_select(self.logits, 0, bucket.rows, out=bucket.logits)
        torch.index_select(self.seeds, 0, bucket.rows, out=bucket.rng.seeds)
        torch.index_select(self.counters, 0, bucket.rows, out=bucket.rng.counters)
        bucket.sampling(self, [s.params for s in running])
        bucket.attention.prepare(rows)
        rotary = self.model.model.rotary_emb(bucket.embeddings[None], bucket.positions,
                                             sequence_length=max(s.position for s in running) + 1)
        planned = time.perf_counter()
        if events:
            events[1].record()
        status = self.graphs.run(bucket, self, rotary)
        if self._observer is not None:
            self._observer.decode(running, bucket, status)
        if events:
            events[2].record()
        launched = time.perf_counter()
        host_status = status.cpu().tolist()  # exactly ONE device->host sync
        if events:
            # Event 2 precedes the blocking status transfer, already complete.
            gpu_ms = events[0].elapsed_time(events[2])
            self.last_run['profile'].append(dict(batch_size=size,
                total_ms=(time.perf_counter() - start) * 1000,
                prepare_host_ms=(planned - start) * 1000,
                launch_host_ms=(launched - planned) * 1000,
                wait_host_ms=(time.perf_counter() - launched) * 1000,
                gpu_stream_ms=gpu_ms, prepare_gpu_ms=events[0].elapsed_time(events[1]),
                forward_update_sample_gpu_ms=events[1].elapsed_time(events[2])))
        self.last_run['steps'] += 1
        return host_status

    @torch.inference_mode()
    def generate(self, requests, sampling=None, on_result=None):
        """Generate results in input order; callback receives each Result on finish.

        Requests are raw strings (already chat-formatted if desired), token lists,
        or Request objects. SamplingParams/dicts may be shared or per request.
        This blocking call continuously admits queued inputs as capacity opens.
        """
        if self._busy:
            raise RuntimeError('TaHEngine.generate is not reentrant')
        states = self._requests(list(requests), sampling or SamplingParams())
        self._busy = True
        waiting, swapped, running = deque(states), deque(), []
        free_rows = list(range(self.max_batch_size - 1, -1, -1))
        results = [None] * len(states)
        started = time.perf_counter()
        self.last_run = dict(steps=0, graph_steps=0, eager_steps=0, swap_outs=0, swap_ins=0,
                             blocked_admission_steps=0, profile=[],
                             compile=dict(enabled=self.compile_enabled,
                                cache_dir=self.compiled_step.cache_dir if self.compiled_step else None,
                                seconds=self.compiled_step.compile_seconds if self.compiled_step else 0.,
                                graphs=self.compiled_step.compile_graphs if self.compiled_step else 0),
                             time_seconds=dict(decode=0., prefill=0., swap=0., host=0.))
        try:
            while waiting or swapped or running:
                self._restore(running, swapped, free_rows)
                if swapped and waiting:
                    self.last_run['blocked_admission_steps'] += 1
                while waiting and not swapped and free_rows:
                    state = waiting[0]
                    bound = len(state.prompt) * self.model.max_iter
                    reserve = math.ceil(self.pool.capacity * self.admission_headroom) + len(running)
                    if self.pool.free_slots < bound + reserve:
                        break
                    waiting.popleft()
                    state.row = free_rows.pop()
                    phase_started = time.perf_counter()
                    reason = self._prefill(state)
                    self.last_run['time_seconds']['prefill'] += time.perf_counter() - phase_started
                    if reason:
                        self._finish(state, reason, results, on_result)
                        free_rows.append(state.row)
                    else:
                        running.append(state)
                if not running:
                    if not waiting and not swapped:
                        break
                    raise KVCapacityError('scheduler cannot make progress: request cannot fit KV pool')
                if self._step_hook is not None:
                    self._step_hook(self, running, swapped, free_rows)
                # Reserve ALL current rows' growth atomically before execution.
                for state in running:
                    if state.cache.size + 1 > self.pool.capacity:
                        raise KVCapacityError('request cannot fit KV pool alone at its next pass')
                while len(running) > self.pool.free_slots:
                    victim = max(running, key=lambda s: s.index)
                    if len(running) == 1:
                        raise KVCapacityError('request cannot fit KV pool alone at its next pass')
                    self._swap_out(victim, running, swapped, free_rows)
                phase_started = time.perf_counter()
                status = self._decode(running)
                self.last_run['time_seconds']['decode'] += time.perf_counter() - phase_started
                for state, (keep, token, raw, probability) in zip(running.copy(), status):
                    if keep:
                        state.depth += 1
                    else:
                        reason = self._sampled(state, int(token), state.depth)
                        if reason:
                            self._finish(state, reason, results, on_result)
                            running.remove(state)
                            free_rows.append(state.row)
                        else:
                            state.depth = 1
                            state.position += 1
            self.last_run.update(wall_seconds=time.perf_counter() - started,
                generated_tokens=sum(len(r.token_ids) for r in results),
                host_swap_peak_bytes=self.swap.peak_bytes)
            if self.compiled_step is not None:
                self.last_run['compile'] = dict(enabled=True, cache_dir=self.compiled_step.cache_dir,
                    seconds=self.compiled_step.compile_seconds, graphs=self.compiled_step.compile_graphs)
            self.last_run['tokens_per_second'] = self.last_run['generated_tokens'] / max(self.last_run['wall_seconds'], 1e-9)
            phases = self.last_run['time_seconds']
            phases['host'] = self.last_run['wall_seconds'] - sum(phases.values())
            self.last_run['time_share'] = {k: v / max(self.last_run['wall_seconds'], 1e-9) for k, v in phases.items()}
            return results
        finally:
            for state in states:
                if state.cache is not None:
                    self.pool.release(state.cache)
                    state.cache = None
                self.swap.discard(state)
            self._busy = False
            assert self.swap.used_bytes == 0
            assert self.pool.free_slots == self.pool.capacity
