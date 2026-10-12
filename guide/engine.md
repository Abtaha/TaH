# TaH inference engine

The new TaH v1 inference engine in `tah/engine/` is **under development**.
Environment/kernel checks and model, checkpoint, fixture, and sampling
components, persistent KV storage, and cached attention exist, but a complete generation engine and evaluation backend
are not available yet. The evaluation CLI currently accepts only `tah`, `hf`,
and `sglang`.

It is intended to provide fast single-GPU inference for TaH v1 checkpoints,
with selective per-token passes, bounded chunked prefill, continuous batching,
managed KV memory, CUDA graph decode, and sampling with per-request seeds.
Planned results include generated tokens/text, prompt and generation pass
counts, finish reasons, and timing. API details and runnable generation
examples will be documented here when implemented; no generation API is
promised by this placeholder.

Use [installation](installation.md) for the existing `engine` extra and
environment/kernel checks, and [evaluation](evaluation.md) for the shipped
reference `tah` backend. TaH2's bundled `tah2/minisgl/` is a separate engine;
see the main [README](../README.md#tah2-usage) for its usage.

## Cache-backed model tools

The engine includes a persistent KV cache and attention primitives for library
users. CUDA attention uses FlashInfer 0.2.5; CPU attention uses dense SDPA for
development checks. Cached teacher forcing returns logits and traces for one
bounded chunk. It does not sample tokens or manage request admission.

After installing the extras, and **only when the GPU is available**, this runs
two chunks with persistent KV, then verifies that all slots were released:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
import torch
from tah.engine.model import EagerTaHModel
from tah.engine.kv_cache import SlotPool

model = EagerTaHModel.from_pretrained('models/TaH-plus-1.7B', device='cuda')
cfg = model.config
free_bytes, _ = torch.cuda.mem_get_info()
# Example budget: leave most free memory for attention and model workspaces.
capacity = SlotPool.capacity_from_bytes(
    int(free_bytes * .1), cfg['num_hidden_layers'],
    cfg['num_key_value_heads'], cfg['head_dim'], model.dtype)
pool = SlotPool(capacity, cfg['num_hidden_layers'], cfg['num_key_value_heads'],
                cfg['head_dim'], model.max_iter, device=model.device, dtype=model.dtype)
request = pool.create_request('example', context_limit=cfg['max_position_embeddings'])
try:
    first = model.forward_chunk([3, 7, 11], pool, request)
    second = model.forward_chunk([13, 17], pool, request, start_position=3)
    print(first.logits.shape, second.logits.shape, second.iter_count.tolist())
finally:
    pool.release(request)
print(pool.audit())
PY
```

Use `device='cpu'` with `CUDA_VISIBLE_DEVICES=""` for development on small
local checkpoints. The loader does not download models. `from_pretrained`
accepts `config_override=None` and `dtype=torch.bfloat16`; override precedence
matches the reference loader. Saved configuration applies when no override is
supplied. These tools have not yet passed production cached parity or GPU
capture validation.

`model.forward_chunk(input_ids, pool, request, *, start_position=0,
iter_count_labels=None, hidden_positions=None, decode=False)` accepts a nonempty
1D sequence of integer token IDs. Positions start at `start_position`, and
successive chunks must append increasing positions. `iter_count_labels` supplies
optional integer labels for the label decider. `hidden_positions` selects local
chunk indices for the returned hidden trace; the default selects none.
`decode=True` uses the decode attention primitive and requires exactly one
token. The result contains:

| Field | Meaning |
| --- | --- |
| `logits` | Final cumulative logits `[chunk_tokens, vocabulary]`, in model dtype. |
| `iter_count` | Executed passes per chunk token. |
| `visited`, `decision` | Boolean `[max_iter, chunk_tokens]` traces; depth indices start at zero. |
| `decision_logits`, `continue_probability` | Decider diagnostics, with NaNs for unvisited passes. |
| `hidden_positions`, `hidden_layer_indices` | Local token indices and reference-compatible hidden-layer indices. |
| `decider_hidden` | Selected hidden inputs `[max_iter, selected_tokens, selected_layers, hidden_size]`. |

Keep or write each result before processing another chunk if you need a full
sequence trace. The cache retains KV for all executed passes; live vocabulary
buffers are bounded by the current chunk. The eager chunk helper can read device
values on the host and is not intended for capture. Capacity failure during a
later pass leaves earlier passes allocated; admission code must reserve growth
before executing a chunk, then release the request if execution fails.

## KV ownership and contiguous transfers

`SlotPool(capacity, num_layers, num_kv_heads, head_dim, max_iter, *, device,
dtype=torch.bfloat16)` allocates page-size-one KV storage shared by all streams.
Each slot holds K and V for every model layer. `capacity_from_bytes(budget_bytes,
num_layers, num_kv_heads, head_dim, dtype=torch.bfloat16)` computes a slot count;
the caller must reserve memory for weights, logits, attention workspaces and
metadata before choosing the budget.

| Method / property | Behavior |
| --- | --- |
| `create_request(request_id, *, context_limit)` | Creates empty per-stream tables. IDs must be unique among live requests; context limit must be positive. |
| `allocate(request, stream, positions)` | Returns device int64 slot IDs for new entries; stream indices are zero-based. Positions must increase strictly within each stream and fit the context limit. |
| `write(layer, slots, k, v)` | Writes `[rows, kv_heads, head_dim]` K/V into unique live slots for one layer. |
| `slot_ids(request)` | Returns all occupied device slot IDs in stream/position order. |
| `gather(request, *, out=None)` | Returns a `KVSnapshot` with contiguous device KV and host coordinate metadata. An optional output buffer must have the exact contiguous shape/device/dtype. |
| `scatter(request, snapshot)` | Copies device KV to already allocated matching stream/position coordinates. |
| `restore(request_id, snapshot)` | Allocates all snapshot coordinates and scatters the payload, allowing different physical slots. Requires sufficient free slots and matching device/model shape/dtype. |
| `release(request)` | Frees every owned slot and empties its tables. Reusing a released handle raises an error. |
| `free_slots`, `audit()` | Free capacity and explicit host ownership/leak checks, including allocation/free counters. |
| `visible_count(request, position, depth)` | Host planning count from the allocation ledger; depth is one-based. |

`KVSnapshot.positions` is a tuple of occupied positions per stream;
`values` has shape `[layers, 2, occupied_entries, 1, kv_heads, head_dim]`;
`context_limit` preserves the request limit. Transfers include every layer and
stream, and copying/restoring is bit preserving. For pinned host swapping, copy
`values` into a pinned CPU tensor with `non_blocking=True`, record a CUDA event,
and wait for that event before releasing/reusing the source slots. Copy the
payload back to a same-device contiguous tensor before `restore`, maintaining
stream/event ordering. The snapshot covers KV only; the caller must separately
preserve logits, next-pass inputs, RNG and scheduler state.

Pool exhaustion raises `KVCapacityError` before allocation; its message
distinguishes needing a swap from a request exceeding the pool alone. Invalid
coordinates, incompatible snapshots and released handles raise `ValueError`.
Tables grow separately for each occupied request/stream; short requests never
receive tables sized to the longest request.

## Attention and mixed-depth passes

`AttentionRow(request, position, depth)` describes a query with one-based depth.
Allocate its current KV slot before preparing attention. Only entries with
`key_position <= position` and `key_depth <= depth` are visible, including the
current pass and earlier passes of the same token.

`DecodeBucket(pool, batch_size, max_entries, num_q_heads, *,
workspace_bytes=134217728)` reserves fixed-address decode metadata and output
buffers. `max_entries` must cover the sum of real occupied entries for its rows,
including entries that their queries cannot see. Reserve room for growth; if it
is exhausted, create a larger bucket outside capture. `prepare(rows)` stages
stream lists and plans FlashInfer outside capture. Reprepare after allocation,
restore, release, or row position/depth changes. It uses the host allocation
ledger for planning and does not copy GPU context contents to the host.
`build()` independently counts and compacts visible entries on the GPU into the
fixed index/indptr buffers. `run(q, layer, *, out=None)` executes attention into
the bucket's reusable output, or a same-shape/device/dtype output buffer.

`PrefillAttention(pool, rows, num_q_heads, *, workspace_bytes=134217728)` groups
consecutive queries of the same request/depth. It builds device masks over each
group's actual KV entries, including chunks with future positions already
allocated. `run(q, layer)` returns attention output. Recreate it after cache
allocation changes. Prefill planning may synchronize with the host.

Both primitives expect model-dtype queries `[rows, query_heads, head_dim]` and
require query heads divisible by KV heads. Each CUDA primitive reserves its own
workspace; reuse plans across layers and include workspaces in memory sizing.
Attention objects reject changed cache revisions rather than reading stale
slot lists. Release slots only after in-flight attention completes.

`model.forward_cached(embeddings, position_ids, depths, slots, pool, attention,
*, rotary=None)` runs one pass for each row, which can belong to different
requests and depths. Embeddings have shape `[rows, hidden_size]` in model dtype;
positions/depths/slots are model-device int64 vectors. Allocate unique live slots,
prepare attention for the same row order, then call this method. It writes K/V
for every layer, masks LoRA by row depth, and returns pass-local `logits` increments
and `hidden_states` with reference-compatible indexing. The caller accumulates
logits and handles continuation decisions/soft inputs. Keep these inputs and
`DecodeBucket` buffers at fixed addresses for capture. Dynamic/LongRoPE needs
frequency planning outside capture and precomputed `(cos, sin)` via `rotary`.

## Development reports

From the repo root, these CPU tools write JSON reports to the explicit output
path, creating parent directories:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python -m tah.engine.tests.attention_check --device cpu --repeats 10 \
  --output data/engine/attention-cpu.json
python -m tah.engine.tests.metadata_check --mode interpret \
  --output data/engine/metadata-interpreter.json
python -m tah.engine.tests.metadata_check --mode compile \
  --output data/engine/metadata-compile.json
```

`attention_check` accepts `--device cpu|cuda` (default CPU), positive `--repeats`
(default 10), and required `--output`. It records batches 1/8/32 with variable
context lengths, metadata construction, prepare wall time, decode, eight-query
prefill, contiguous gather/scatter and final pool audits. Times are microseconds;
CUDA additionally records metadata+attention graph replay and peak allocated
memory. Select CUDA only when the GPU is free.

`metadata_check` requires empty CUDA visibility, `--mode interpret|compile`,
and `--output`. Interpretation checks exact ragged visibility/compaction on CPU.
Compilation builds both kernels offline for sm80/sm86/sm89 using the installed
Triton/toolchain and never loads a CUDA module. Reports include
`production_gates_eligible=false`; CPU correctness and timing do not establish
production model parity, GPU performance or portability.
