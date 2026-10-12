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

## Reference parity fixtures

The reference fixture tool records teacher-forced depth decisions, probabilities,
pass counts, logits and selected hidden inputs, plus greedy generations. It runs
the existing reference model and generation helper. Reference/model dependencies
are used by this tool, while the fixture loader only needs torch and safetensors.

From the repository root, this prepares benchmark inputs and synthetic checkpoint
files on CPU. The real checkpoint must already exist locally:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES="" python -m tah.engine.generate_fixtures \
  --profile full --prepare-only --output data/engine/fixtures/reference
```

When the GPU is available, run one checkpoint at a time:

```bash
source .venv-tah/bin/activate
for checkpoint in real mlp3 mlp1 label3; do
  CUDA_VISIBLE_DEVICES=0 python -m tah.engine.generate_fixtures \
    --profile full --device cuda --checkpoint "$checkpoint" \
    --output data/engine/fixtures/reference || break
done
```

The package checks the GPU/toolchain on import. Follow
[installation](installation.md) if it reports missing Python headers or CUDA
Toolkit prerequisites. CPU imports require explicitly empty CUDA visibility.
For a small CPU development suite, use `--profile tiny --device cpu` and
`--output data/engine/fixtures/cpu-tiny`; its tokens are remapped into a tiny random
Qwen3 vocabulary and it cannot establish production parity tolerances.

| Option | Default and behavior |
| --- | --- |
| `--profile tiny\|full` | `tiny`; full uses the real checkpoint and three synthetic Qwen3-0.6B-Base checkpoints. |
| `--device cpu\|cuda` | `cpu`; CUDA is an explicit choice and is refused when visibility is empty. Full capture requires CUDA. |
| `--prepare-only` | Prepare/reuse inputs and synthetic models; do not capture. Allows full preparation on CPU. |
| `--output PATH` | `data/engine/fixtures/cpu-tiny`; fixture root. Use a different root for the full profile. |
| `--model-output PATH` | `data/engine/fixture-models`; saves models under `<path>/<profile>/{mlp3,mlp1,label3}`. |
| `--real-model PATH` | `models/TaH-plus-1.7B`; local real checkpoint and benchmark prompt tokenizer. |
| `--base-model ID_OR_PATH` | `Qwen/Qwen3-0.6B-Base`; pretrained synthetic base and its tokenizer. Downloads may be needed during preparation. |
| `--eval-config PATH` | `script/tah/recipes/qwen3_1.7/eval_tah.yaml`; reference TaH override for the real checkpoint. |
| `--checkpoint real\|mlp3\|mlp1\|label3` | All checkpoints in the chosen profile; repeat the flag to select several. Tiny has no real checkpoint. |
| `--seed INTEGER` | `20261012`; random base/adapter/decider preparation seed. |
| `--topk INTEGER` | `32`, positive; final-logit values/indices retained at every position, capped at vocabulary size. |
| `--chunk-size INTEGER` | `256`, nonnegative; reference cached teacher-forced chunk width. `0` uses the complete padded input width. |
| `--generation-tokens INTEGER` | `8` for tiny, `256` for full; greedy generation cap. `0` selects the profile default; negative values are rejected. |
| `--threads INTEGER` | `4`, positive; torch CPU thread count for preparation and metrics. |
| `-h`, `--help` | Print usage and exit. |

The full inputs contain three lengths (256, 1024, 4096 tokens) from each of
math500, GPQA and HumanEval. Prompts match evaluation rendering. Continuations
repeat a recorded dataset solution, canonical solution or answer scaffold to
reach the target length; they are teacher-forced inputs. Each sequence has five
teacher-forced layouts, and six prompts per checkpoint have four greedy layouts.
The MLP3 random decider is calibrated to continue 10–40% of active tokens at
each nonterminal depth, checked against the whole singleton fixture corpus.
IterLabel teacher forcing includes oracle labels; its greedy cases have no labels
and stop after one pass.

Existing inputs/models are reused after profile/seed/source checks. A selected
checkpoint's records are regenerated when that checkpoint is rerun. Capture
settings must match an existing suite with records; choose a new output root to
change them. Interrupted captures remain incomplete. `--prepare-only` marks the
manifest prepared, so use it before production capture. Production eligibility
becomes true only after all four checkpoint sets finish and depth coverage passes.

Outputs under the fixture root:

| File | Contents |
| --- | --- |
| `inputs.json` | Dataset provenance, rendered prompts, continuation descriptions and exact token IDs. |
| `manifest.json` | Schema version, status, production eligibility, effective configs, dependency versions, source/checkpoint/record hashes, file sizes and self-noise statistics. |
| `<checkpoint>/teacher_forced/*.safetensors` | Unpadded inputs and per-depth traces; one file per sequence/layout. |
| `<checkpoint>/greedy/*.safetensors` | Prompt/generated IDs, separate prompt and generation pass counts; EOS excluded. |
| `<checkpoint>/self_noise.json` | Detailed and aggregate teacher-forced errors, per-depth decision/margin counts and greedy first-divergence positions. |

Teacher-forced tensors include `visited`, `decision`, `continue_probability`,
`decision_logits`, `pass_count`, `final_topk_values`, `final_topk_indices`,
`final_top1`, `hidden_positions`, `hidden_layer_indices`, `decider_hidden`,
`logit_positions`, and `final_logits_sample`; oracle cases also include
`iter_count_labels`. Depth axes are zero-based. Unvisited probability/logit/hidden
entries are NaN and must be masked by `visited`. Cap probabilities are zero;
label-decider probabilities are deterministic 0/1. Hidden indices follow HF's
hidden-state tuple, including embeddings at index 0 and final normalization at
the last index. Up to twelve positions retain hidden inputs and four retain
dense vocabulary logits. Every position retains its top-k and actual argmax.

Full-vocabulary self-noise is measured at every position during capture, although
only sampled dense rows are stored. Use reference replay for exhaustive logit
checks, or the separately recorded retained-coordinate metrics for disk-only
checks. Never compare independent top-k rank lists as if they had equal indices.
G-TF limits use the production reference self-noise of the same checkpoint.
Full-logit MAE/RMSE/max error and top-1 disagreement must be within the recorded
1.5× limits, with no epsilon floor. At each depth, decision flips are counted
only on positions visited by both implementations; their rate must be at most
1.5× the reference rate. Pass-count mismatch rate has its own 1.5× reference
limit, covering later visitation differences. A zero reference rate requires
an exact match. Every flip must lie within `max(0.01, 1.5×largest reference flip
margin)` of the effective threshold. Calibration and comparison measure both
probabilities at a flip, so reversing a pair does not change its margin.
Greedy divergence positions are zero-based; `null` means identical IDs and EOS
length. For G-GEN, identical outputs count as their generation length; the
candidate median first divergence must be at least half the reference median,
and position-zero divergence is allowed only if present in reference self-noise.

Load and verify files on CPU without loading a model:

```bash
CUDA_VISIBLE_DEVICES="" .venv-tah/bin/python - <<'PY'
import hashlib
from tah.engine.fixtures import FixtureSuite
suite = FixtureSuite('data/engine/fixtures/reference')
assert suite.manifest['production_gates_eligible']
for record in suite.records():
    tensors = suite.load(record)
    path = suite.root / record['file']
    assert path.stat().st_size == record['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']
print(len(suite.manifest['records']), suite.manifest['total_fixture_bytes'])
PY
```

`FixtureSuite(root, allow_partial=False)` rejects incomplete/failed manifests by
default. `records(kind=None, checkpoint=None)` filters metadata lazily; `load(record)`
returns CPU tensors and checks trace/pass-count shapes. Hash verification is
explicit in the example. Use `allow_partial=True` for inspection only, and require
`profile='full'`, `status='complete'` and `production_gates_eligible=true` for
production gates.

## Full-sequence teacher forcing

`EagerTaHModel` loads a local TaH v1 checkpoint into engine-owned modules and
returns final logits for one complete sequence. It uses dense attention with
temporary per-pass KV history, and keeps no KV between calls. Use this tool for
validation; generation and scheduling are separate APIs still under development.
The model runtime does not import Transformers or PEFT model code.

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python - <<'PY'
from pathlib import Path
import yaml
from tah.engine.model import EagerTaHModel

override = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
model = EagerTaHModel.from_pretrained('models/TaH-plus-1.7B', override, device='cuda')
output = model([3, 7, 11], hidden_positions=[0, 2])
print(output.logits.shape, output.iter_count.tolist())
PY
```

`from_pretrained(model_path, config_override=None, *, device='cpu',
dtype=torch.bfloat16)` accepts a local directory, an optional TaH-config
dataclass or dictionary (including an evaluation YAML dictionary), a torch
device, and BF16/FP16/FP32 dtype. It does not download models. Nonempty override
fields replace saved fields; None and empty dictionaries retain saved values.
LoRA targets must be internal linear modules; embedding and `lm_head` adapters
are rejected. Unsupported model/adapter variants and missing or inconsistent
weights raise explicit errors. The saved adapter config determines rank/scaling.
Default, linear, dynamic NTK, YaRN, LongRoPE and Llama 3 RoPE scaling are
supported; dynamic/LongRoPE growth and short-sequence resets preserve the
reference frequency precision after a model dtype/device cast.

`model(input_ids, *, iter_count_labels=None, position_ids=None,
hidden_positions=None)` requires a nonempty 1D int64 sequence in vocabulary
range. Integer lists are accepted and moved to the model device. Labels are
optional per-token int64 iteration labels; the label decider stops after one
pass when absent. Positions default to `0..T-1`; supplied positions must be
nonnegative int64 with the input shape. Hidden positions select local sequence
indices and default to none. The output has the same fields described for
cached teacher forcing below, with `T` in place of chunk length. All logits and
probabilities retain the model dtype; unvisited diagnostics are NaN.

## Production teacher-forced parity check

After production fixture capture completes, run each checkpoint independently:

```bash
source .venv-tah/bin/activate
for checkpoint in real mlp3 mlp1 label3; do
  CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s -o addopts= -m 'gpu and slow' \
    "tah/engine/tests/test_gtf.py::test_gtf_every_wp2_fixture_set[${checkpoint}]" || break
done
```

This verifies checkpoint/reference/config/record hashes, tests all five saved
teacher-forced layouts for each sequence, and replays the unchanged reference
for full-vocabulary errors. Incomplete or tiny fixtures are rejected. Empty
CUDA visibility skips these GPU tests; `-o addopts=` enables the slow cases.

| Environment variable | Default / behavior |
| --- | --- |
| `TAH_GTF_FIXTURES` | `data/engine/fixtures/reference`; complete production fixture root. |
| `TAH_GTF_TOLERANCES` | `data/engine/wp2/tolerances.json`; reference limits export with matching manifest hash. Its legacy fixed-decision fields are superseded by decision limits derived from saved reference trace pairs. |
| `TAH_GTF_RESULTS` | `data/engine/wp3/gtf`; creates the report directory. |

`<checkpoint>-limits.json` records decision limits and the unchanged logit
limits before model comparison, together with input hashes and the margin
convention. `<checkpoint>.json` records PASS/FAIL, aggregate and per-layout
metrics, common-visited flip counts/rates/margins, pass-count mismatch rate,
noise/decision limits, GPU name, elapsed seconds and peak allocated/reserved
GPU bytes. Read binding limits from `noise_limits` and `decision_limits`;
`g_tf_*_limit` fields inherited in the measured `summary` rescale engine errors
and do not define the binding bounds. A completed comparison writes its metrics even when a gate fails;
an earlier loading/integrity/runtime error may leave only the limits report
and pytest traceback. Retained top-k/dense errors are also checked against
their separately measured reference limits. No CPU development report is a
production gate.

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
