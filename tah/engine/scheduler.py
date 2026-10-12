"""Host request ledger and bounded pinned-host swapping (no recomputation)."""
from dataclasses import dataclass, field
import torch

from .kv_cache import KVCapacityError
from .sampling import SamplingParams


@dataclass(frozen=True)
class Request:
    """Optional per-request controls; plain strings/token lists also work."""
    prompt: object
    max_new_tokens: int = 1024
    seed: int = 0


@dataclass
class Result:
    request_id: int
    token_ids: list
    text: str
    prompt_pass_counts: list
    generated_pass_counts: list
    finish_reason: str
    timing: dict


@dataclass
class State:
    index: int
    prompt: list
    params: SamplingParams
    max_new_tokens: int
    seed: int
    submitted: float
    cache: object = None
    row: int = -1
    position: int = 0
    depth: int = 1
    generated: list = field(default_factory=list)
    prompt_counts: list = field(default_factory=list)
    generated_counts: list = field(default_factory=list)
    prefill_seconds: float = 0.
    admitted: float = 0.
    swaps: int = 0
    snapshot: object = None


@dataclass
class SwapSnapshot:
    positions: tuple
    kv: torch.Tensor
    logits: torch.Tensor
    embeddings: torch.Tensor
    seeds: torch.Tensor
    counters: torch.Tensor
    nbytes: int
    event: object


class HostSwap:
    """One bounded device scratch, stream-ordered copies and explicit accounting.

    A snapshot owns pinned host tensors only. Scheduler fields stay in State;
    they are frozen while swapped. Slot release is safe on the same CUDA stream
    as copies; restoration waits on the recorded completion event on device.
    """
    def __init__(self, pool, budget_bytes, chunk_entries):
        if budget_bytes < 0 or chunk_entries < 1:
            raise ValueError('host swap budget must be nonnegative; chunk_entries positive')
        self.pool, self.budget_bytes = pool, budget_bytes
        self.used_bytes = self.peak_bytes = 0
        self.chunk_entries = min(chunk_entries, pool.capacity)
        self.scratch = torch.empty(pool.values.shape[0], 2, self.chunk_entries,
                                   *pool.values.shape[3:], dtype=pool.dtype, device=pool.device)

    def _host(self, shape, dtype):
        return torch.empty(shape, dtype=dtype, pin_memory=self.pool.device.type == 'cuda')

    def out(self, state, device_state):
        pool, cache, row = self.pool, state.cache, state.row
        positions = tuple(tuple(t.host_positions) for t in cache.streams)
        ids = [slot for t in cache.streams for slot in t.host_slots]
        per_slot = pool.values.numel() * pool.values.element_size() // pool.capacity
        nbytes = len(ids) * per_slot + sum(t[row].numel() * t.element_size() for t in device_state)
        # Include host coordinate/slot ledger overhead conservatively too.
        nbytes += len(ids) * 80 + 1024
        if self.used_bytes + nbytes > self.budget_bytes:
            raise KVCapacityError(f'host swap budget exceeded: need {nbytes} bytes, '
                                  f'{self.budget_bytes - self.used_bytes} available')
        kv = self._host((pool.values.shape[0], 2, len(ids), *pool.values.shape[3:]), pool.dtype)
        for start in range(0, len(ids), self.chunk_entries):
            part = ids[start:start + self.chunk_entries]
            scratch = self.scratch[:, :, :len(part)]
            # Partial chunks have layer strides: index_select(out=) supports it.
            torch.index_select(pool.values, 2, torch.tensor(part, device=pool.device), out=scratch)
            kv[:, :, start:start + len(part)].copy_(scratch, non_blocking=True)
        payload = []
        for tensor in device_state:
            host = self._host(tensor[row].shape, tensor.dtype)
            host.copy_(tensor[row], non_blocking=True)
            payload.append(host)
        event = None
        if pool.device.type == 'cuda':
            event = torch.cuda.Event()
            event.record()
        snapshot = SwapSnapshot(positions, kv, *payload, nbytes, event)
        self.used_bytes += nbytes
        self.peak_bytes = max(self.used_bytes, self.peak_bytes)
        pool.release(cache)
        state.cache, state.snapshot = None, snapshot
        state.swaps += 1

    def restore(self, state, device_state, context_limit):
        pool, snapshot = self.pool, state.snapshot
        count = sum(map(len, snapshot.positions))
        if count + 1 > pool.free_slots:
            raise KVCapacityError('not enough free slots to restore request and reserve its next pass')
        if snapshot.event is not None:
            torch.cuda.current_stream(pool.device).wait_event(snapshot.event)
        cache = pool.create_request(state.index, context_limit=context_limit)
        try:
            ids = []
            for stream, positions in enumerate(snapshot.positions):
                pool.allocate(cache, stream, positions)
                ids.extend(cache.streams[stream].host_slots)
            for start in range(0, count, self.chunk_entries):
                part = ids[start:start + self.chunk_entries]
                scratch = self.scratch[:, :, :len(part)]
                scratch.copy_(snapshot.kv[:, :, start:start + len(part)], non_blocking=True)
                pool.values.index_copy_(2, torch.tensor(part, device=pool.device), scratch)
            for tensor, host in zip(device_state, (snapshot.logits, snapshot.embeddings, snapshot.seeds, snapshot.counters)):
                tensor[state.row].copy_(host, non_blocking=True)
        except Exception:
            pool.release(cache)
            raise
        # Keep pinned payload alive until DMA finishes. PyTorch's pinned
        # allocator records stream use for copy_; freeing Python refs is safe.
        state.cache, state.snapshot = cache, None
        self.used_bytes -= snapshot.nbytes

    def discard(self, state):
        if state.snapshot is not None:
            self.used_bytes -= state.snapshot.nbytes
            state.snapshot = None
