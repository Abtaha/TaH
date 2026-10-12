"""Static-shape, device-only sampling for depth-mixed decode rows.

Create the sampler, parameter buffers and request RNG buffers at admission.
Copy/reorder those buffers outside the captured step; keep their addresses fixed
for replay. Only finished-token rows should be active: inactive and greedy rows
consume no random draw. Runtime imports contain no reference/HF/PEFT code.

Logits may contain -inf, but each row must contain a finite value and no NaN/+inf.
FP32 filtering is allowed by SPEC. A CUDA vocabulary >4096 uses PyTorch 2.6's
stable large-sort tie order (token index) in the bounded top-k implementation.
Small vocabularies and CPU use the actual reference sort, including its ties.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SamplingParams:
    temperature: float = 1.0
    min_p: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = True

    def __post_init__(self):
        for name in ("temperature", "min_p", "top_p"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 <= self.min_p <= 1 or not 0 <= self.top_p <= 1:
            raise ValueError("min_p and top_p must be in [0, 1]")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 0:
            raise ValueError("top_k must be a nonnegative integer")
        if not isinstance(self.do_sample, bool):
            raise ValueError("do_sample must be bool")

    @property
    def greedy(self) -> bool:
        # The generation API convention is temperature=0 -> greedy. The raw
        # reference helper requires do_sample=False (division by zero otherwise).
        return not self.do_sample or self.temperature == 0


class SamplingBatch:
    """Fixed-address per-row parameters; host validation happens at admission.

    Use update() for new requests, then gather/copy validated buffers when rows
    are reordered. Do not write out-of-contract values directly into tensors.
    """

    def __init__(self, params: Sequence[SamplingParams], *, device, max_top_k=None,
                 greedy_only=False):
        if not params:
            raise ValueError("sampling batch must have at least one row")
        self.max_top_k = max_top_k
        self.greedy_only = greedy_only
        self.temperature = torch.empty(len(params), dtype=torch.float32, device=device)
        self.min_p = torch.empty_like(self.temperature)
        self.top_p = torch.empty_like(self.temperature)
        self.top_k = torch.empty(len(params), dtype=torch.int64, device=device)
        self.do_sample = torch.empty(len(params), dtype=torch.bool, device=device)
        self.update(params)

    def update(self, params: Sequence[SamplingParams]) -> None:
        if len(params) != self.temperature.numel():
            raise ValueError("parameter update must preserve the batch shape")
        for p in params:
            if not isinstance(p, SamplingParams):
                raise TypeError("parameters must be SamplingParams")
            if self.greedy_only and not p.greedy:
                raise ValueError("greedy_only sampler requires every row to be greedy")
            if self.max_top_k is not None and not p.greedy:
                if not 1 <= p.top_k <= self.max_top_k:
                    raise ValueError("bounded sampler requires 1 <= top_k <= max_top_k; "
                                     "use the general sampler for unbounded/large top_k")
        for name in ("temperature", "min_p", "top_p", "top_k", "do_sample"):
            buffer = getattr(self, name)
            values = [getattr(p, name) for p in params]
            # Greedy ignores all filters, including arbitrarily large top_k.
            if name == "top_k":
                values = [1 if p.greedy else p.top_k for p in params]
            buffer.copy_(torch.tensor(values, dtype=buffer.dtype, device=buffer.device))


@dataclass
class RequestRNG:
    """Per-request signed-int64 seeds and draw counters, carried with the request.

    SplitMix64(seed + gamma * (counter + 1)) produces one open-interval uniform
    per draw. It does not use a global torch generator. Greedy/inactive rows do
    not advance. Reset counters to zero to reproduce a request from its seed.
    Counters must stay nonnegative (<2**63-1 draws per request).
    """
    seeds: torch.Tensor
    counters: torch.Tensor

    @classmethod
    def from_seeds(cls, seeds: Sequence[int], *, device="cpu") -> RequestRNG:
        if not seeds or any(isinstance(s, bool) or not isinstance(s, int) or
                            not -(1 << 63) <= s < (1 << 63) for s in seeds):
            raise ValueError("seeds must be a nonempty sequence of signed int64 integers")
        tensor = torch.tensor(seeds, dtype=torch.int64, device=device)
        return cls(tensor, torch.zeros_like(tensor))


def _logical_right_shift(value: torch.Tensor, bits: int) -> torch.Tensor:
    return (value >> bits) & ((1 << (64 - bits)) - 1)


def request_uniform(rng: RequestRNG) -> torch.Tensor:
    """Pure device ops, with defined two's-complement int64 wraparound."""
    z = rng.seeds + (-7046029254386353131) * (rng.counters + 1)
    z = (z ^ _logical_right_shift(z, 30)) * (-4658895280553007687)
    z = (z ^ _logical_right_shift(z, 27)) * (-7723592293110705685)
    z = z ^ _logical_right_shift(z, 31)
    # 52 bits plus a half-bin offset are exactly representable in float64,
    # including the two endpoints: strictly 0 < u < 1.
    return (_logical_right_shift(z, 12).to(torch.float64) + 0.5) * (2.0 ** -52)


class GraphSampler:
    """Distribution-equivalent sampler; no scalar reads or data-dependent shapes.

    max_top_k=None supports all filters with a general full-vocabulary sort.
    A fixed max_top_k (e.g. 20 for W-perf) admits per-row k in [1,max_top_k]
    and avoids a vocabulary sort on CUDA when V>4096. It retains *all* ties,
    including overflow beyond that bound. Selection/RNG work is O(B*V), with
    O(B*max_top_k) shortlist storage. CUDA capture/performance require GPU checks.
    """

    def __init__(self, vocab_size: int, *, max_top_k: int | None = None,
                 greedy_only: bool = False):
        if not isinstance(vocab_size, int) or vocab_size < 1:
            raise ValueError("vocab_size must be a positive integer")
        if max_top_k is not None and (not isinstance(max_top_k, int) or max_top_k < 1):
            raise ValueError("max_top_k must be a positive integer or None")
        self.vocab_size = vocab_size
        self.max_top_k = max_top_k
        self.greedy_only = greedy_only
        self.shortlist_size = min(max_top_k or vocab_size, vocab_size)

    def make_batch(self, params: Sequence[SamplingParams], *, device="cpu") -> SamplingBatch:
        return SamplingBatch(params, device=device, max_top_k=self.max_top_k,
                             greedy_only=self.greedy_only)

    def _check(self, logits, batch):
        # Metadata only; these checks never inspect tensor values.
        if logits.ndim != 2 or logits.shape != (batch.temperature.numel(), self.vocab_size):
            raise ValueError("logits must have shape [batch, vocab_size]")
        if logits.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise ValueError("logits must have floating dtype fp32/bf16/fp16")
        if (batch.temperature.device != logits.device or batch.max_top_k != self.max_top_k
                or batch.greedy_only != self.greedy_only):
            raise ValueError("parameter buffers must match sampler bound and logits device")

    def _min_p(self, logits, batch):
        temperature = torch.where(batch.temperature > 0, batch.temperature, 1.0)
        probs = torch.softmax(logits.float() / temperature[:, None], dim=-1)
        kept = torch.where(probs >= batch.min_p[:, None] * probs.amax(-1, keepdim=True),
                           probs, 0.0)
        # Match the reference's conditional renormalization, even at min_p=0.
        return torch.where(batch.min_p[:, None] > 0,
                           kept / kept.sum(-1, keepdim=True), probs)

    def _top_k(self, probs, batch):
        if self.max_top_k is None:
            ranked = torch.sort(probs, descending=True, dim=-1).values
        else:
            ranked = torch.topk(probs, self.shortlist_size, dim=-1, sorted=True).values
        k = batch.top_k.clamp(1, ranked.shape[-1])
        cutoff = ranked.gather(-1, (k - 1)[:, None])
        kept = torch.where((batch.top_k[:, None] == 0) | (probs >= cutoff), probs, 0.0)
        mass = kept.sum(-1, keepdim=True)
        probs = torch.where(batch.top_k[:, None] > 0, kept / mass, probs)
        ranked = torch.where(ranked >= cutoff, ranked / mass, 0.0)
        return probs, ranked, cutoff

    def _full_probabilities(self, probs, batch):
        probs, _, _ = self._top_k(probs, batch)
        # Sorting AFTER both filters preserves the CPU's unstable tie behavior.
        ranked, indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = ranked.cumsum(-1)
        remove = cumulative > batch.top_p[:, None]
        shifted = torch.cat((torch.zeros_like(remove[:, :1]), remove[:, :-1]), dim=-1)
        remove = torch.zeros_like(shifted).scatter(-1, indices, shifted)
        kept = torch.where(remove, 0.0, probs)
        return torch.where(batch.top_p[:, None] < 1,
                           kept / kept.sum(-1, keepdim=True), probs)

    def _bounded_probabilities(self, probs, batch):
        """Stable-sort nucleus without sorting V. Also CPU-testable in isolation.

        There are at most K-1 entries strictly above the k-th value. Any
        overflow is an equal-valued tail. Find the nucleus boundary in the
        K values, then reconstruct all ties using their token-index ordinal.
        The FP64 prefix expression is rounded once to FP32, as cumsum is on
        CPU; GPU cumsum arithmetic parity is explicitly covered by GPU tests.
        """
        probs, ranked, _ = self._top_k(probs, batch)
        cumulative = ranked.cumsum(-1)
        position = (cumulative <= batch.top_p[:, None]).sum(-1).clamp(max=self.shortlist_size - 1)
        cutoff = ranked.gather(-1, position[:, None])
        # If the nucleus extends beyond K, its boundary is in the bottom tie.
        cutoff = torch.where(cumulative[:, -1:] <= batch.top_p[:, None], ranked[:, -1:], cutoff)
        high_mass = torch.where(ranked > cutoff, ranked, 0.0).sum(
            -1, keepdim=True, dtype=torch.float64)
        tied = (probs == cutoff) & (probs > 0)
        ordinal = tied.cumsum(-1, dtype=torch.int32)
        previous = (high_mass + cutoff.to(torch.float64) * (ordinal - 1)).float()
        keep = (probs > cutoff) | (tied & (previous <= batch.top_p[:, None]))
        kept = torch.where(keep, probs, 0.0)
        return torch.where(batch.top_p[:, None] < 1,
                           kept / kept.sum(-1, keepdim=True), probs)

    def _filtered_probabilities(self, logits, batch):
        probs = self._min_p(logits, batch)
        if self.max_top_k is not None and logits.is_cuda and self.vocab_size > 4096:
            return self._bounded_probabilities(probs, batch)
        return self._full_probabilities(probs, batch)

    @torch.no_grad()
    def probabilities(self, logits: torch.Tensor, batch: SamplingBatch) -> torch.Tensor:
        """Diagnostic filtered target, including one-hot distributions for greedy."""
        self._check(logits, batch)
        if self.greedy_only:
            return torch.zeros_like(logits, dtype=torch.float32).scatter(
                -1, logits.argmax(-1, keepdim=True), 1.0)
        probs = self._filtered_probabilities(logits, batch)
        greedy = (~batch.do_sample) | (batch.temperature == 0)
        index = logits.argmax(-1, keepdim=True)
        one_hot = torch.zeros_like(probs).scatter(-1, index, 1.0)
        return torch.where(greedy[:, None], one_hot, probs)

    @torch.no_grad()
    def sample(self, logits: torch.Tensor, batch: SamplingBatch, rng: RequestRNG,
               active: torch.Tensor | None = None) -> torch.Tensor:
        """Return int64 [B] IDs (-1 for inactive rows), advancing active RNGs.

        rng tensors and active must be on logits.device with shape [B]. No
        host sync, nonzero(), dynamic allocation shape or global RNG operation.
        """
        self._check(logits, batch)
        for state in (rng.seeds, rng.counters):
            if state.shape != logits.shape[:1] or state.device != logits.device or state.dtype != torch.int64:
                raise ValueError("RNG state must be int64 [batch] on logits device")
        if active is None:
            active = torch.ones_like(batch.do_sample)
        if active.shape != logits.shape[:1] or active.device != logits.device or active.dtype != torch.bool:
            raise ValueError("active must be bool [batch] on logits device")
        if self.greedy_only:
            return torch.where(active, logits.argmax(-1), -1)
        probs = self._filtered_probabilities(logits, batch)
        cdf = probs.cumsum(-1)
        target = request_uniform(rng)[:, None] * cdf[:, -1:].to(torch.float64)
        # Equivalent to cdf.double() < target, without a [B,V] FP64 buffer.
        # Resolve rounding onto an exact FP32 CDF boundary using only [B,1].
        rounded = target.float()
        below = (cdf < rounded) | ((cdf == rounded) & (rounded.double() < target))
        token = below.sum(-1)
        greedy = (~batch.do_sample) | (batch.temperature == 0)
        token = torch.where(greedy, logits.argmax(-1), token)
        rng.counters.add_((active & batch.do_sample & (batch.temperature > 0)).to(torch.int64))
        return torch.where(active, token, -1)
