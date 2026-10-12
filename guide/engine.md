# TaH inference engine

`TaHEngine` provides experimental eager generation for local TaH v1 checkpoints on one
CUDA GPU. It runs all prompt passes in bounded chunks, mixes token depths in
decode, and admits queued requests as completed requests free capacity. CUDA
graph generation and the evaluation CLI's `tah_engine` backend are still pending;
the evaluation CLI accepts `tah`, `hf`, and `sglang`.

Generated-sequence reference parity, greedy sanity checks and production swap/
long-context robustness checks pass on the RTX 3090. Eager W-perf with 64 resident
requests measures 962.81 generated tokens/s versus 78.26 for the reference baseline.
End-to-end accuracy evaluation and L40S validation remain later validation steps.

Use [installation](installation.md) for the `engine` extra and prerequisites,
and [evaluation](evaluation.md) for the reference backend. TaH2's bundled
`tah2/minisgl/` is a separate engine.

## Generate with the eager engine

Run from the repository root in the frozen environment:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python - <<'PY_ENGINE'
import yaml
from tah.engine import TaHEngine, Request, SamplingParams

config = yaml.safe_load(open('script/tah/recipes/qwen3_1.7/eval_tah.yaml'))
engine = TaHEngine('models/TaH-plus-1.7B', config_override=config)
requests = [Request('<|im_start|>user\nWhat is 2 + 2?<|im_end|>\n<|im_start|>assistant\n',
                    max_new_tokens=128, seed=42)]
results = engine.generate(requests, SamplingParams(temperature=.6, top_k=20, top_p=.95),
                          on_result=lambda result: print(result.text))
print(results[0].generated_pass_counts)
print(engine.last_run)
PY_ENGINE
```

`generate(requests, sampling=None, on_result=None)` returns `Result` objects in
input order. Each callback receives one `Result` as it finishes; completion order
may differ from input order. The call is blocking and cannot be nested or called
concurrently on the same engine. A callback exception propagates after request
KV and live host swap buffers are released.

Each request may be a token-ID list, a raw string, or `Request(prompt,
max_new_tokens=1024, seed=0)`. Strings are tokenized with the checkpoint's
`tokenizer.json`; apply a chat template yourself, as in the example. No model
code from Transformers or PEFT is imported at generation runtime. A checkpoint
without a tokenizer accepts token IDs and returns empty result text. Plain
strings/lists use the engine's `max_new_tokens` and seed plus input index.

`sampling` accepts one `SamplingParams` or dictionary shared by all requests,
or a list with one entry per request. The default is temperature 1, min_p 0,
top_k 0, top_p 1, do_sample True. Temperature 0 or do_sample False selects greedy
argmax. See the sampling section below for filtering and cutoff rules. A seed
is a signed 64-bit integer; its draw counter belongs to the request and survives
swapping. Repeat a call with the same seeds and batch composition for identical
outputs.

Each `Result` has `request_id` (input index), `token_ids`, `text`,
`prompt_pass_counts`, `generated_pass_counts`, `finish_reason` (`eos` or
`length`), and `timing`. Generated counts describe the context position whose
final logits sampled that output token. EOS and its count are excluded, exactly
as in reference generation; the EOS ID comes from the tokenizer, rather than
an expanded generation-config stop list. `max_new_tokens=0` still computes
prompt counts and returns an empty output. Timing contains `total_seconds`
(including queue time), `queue_seconds`, `prefill_seconds`, and `swaps`.

Constructor options (keyword-only):

| Option | Default | Effect |
|---|---|---|
| `device` | `"cuda"` | GPU device; `"cpu"` is development only and requires explicit KV capacity. |
| `max_batch_size` | 128 | Maximum resident sequences contributing one row per decode step. |
| `prefill_chunk_size` | 128 | Maximum prompt tokens with live vocabulary logits. |
| `memory_fraction` | .9 | Fraction of measured free GPU memory after loading weights available for KV and workspace reservations. |
| `kv_capacity` | `None` | Auto-size page-size-one KV entries across all passes; an explicit larger-than-safe capacity is rejected. |
| `admission_headroom` | .1 | Fraction of KV slots that must remain free after worst-case prefill, in addition to running-row growth. |
| `host_swap_bytes` | 8 GiB | Live pinned-host swap payload/ledger budget; exceeding it raises `KVCapacityError`. |
| `swap_chunk_entries` | 256 | Maximum KV entries in shared device swap scratch; swaps never gather a whole request on device. |
| `max_new_tokens` | 1024 | Generation cap for plain string/token-list requests. |
| `seed` | 0 | Base seed for plain requests; each input adds its index. |
| `eos_token_id` | `"tokenizer"` | Read tokenizer EOS; set an integer explicitly or `None` to disable EOS stopping. |
| `sampling_topk` | 128 | Bounded shortlist limit; sampling with k=0 or larger k uses the general full-vocabulary path. |
| `compile` | `False` | Eager mode; `True` currently raises an actionable error. |
| `profile_steps` | 0 | Collect CUDA stream-event and host timings for the first specified decode steps. Stream events include launch gaps; use the profiler tool below for kernel execution time. |

`config_override` follows the checkpoint loader's reference-compatible TaH
merge rules documented below. KV sizing budgets both 128 MiB attention float
workspaces, private FlashInfer integer workspaces for every possible exact batch
size, shared ragged metadata, geometrically grown request tables, logits/input
state, model/sampling intermediates, bounded swap scratch, and allocator margin.
`engine.memory_budget` exposes the measured free bytes, reservations and resulting
capacity. Memory scales with real request tokens; prefill does not pad short
prompts to the longest request.

Admission reserves the entire prompt's worst-case passes and the next pass of
every running row before execution. If growth runs out of slots, the newest
resident request swaps its KV, cumulative logits, next input embedding and RNG
state to pinned host memory. Its host scheduler state is frozen. Restoration is
FIFO by preemption time; new admission is blocked while any request is swapped.
There is no recompute path. Requests exceeding the model context, the worst-case
prefill capacity/headroom, or the actual single-request growth capacity fail
clearly. Slot and swap ownership are released on success and on exceptions.

`engine.last_run` reports wall seconds (excluding engine construction), generated
tokens, tokens/s, decode steps, swap counts, blocked-admission steps, peak live
host swap bytes, and optional per-step timing samples. Each CUDA decode step
uses one status transfer to the host. Exact batch-size buckets keep fixed buffer
shapes and a reusable unsplit FlashInfer plan; no per-step context-length scan
runs in Python. Low batch sizes can leave the GPU underutilized in eager mode.

## Engine validation and performance tools

Run each checkpoint/gate as a separate process, with a 28-minute limit. These
commands require the completed production fixtures and local checkpoint paths:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m pytest -q tah/engine/tests/test_engine_gpu.py
for checkpoint in real mlp3 mlp1 label3; do
  CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.engine_check \
    --gate gen --checkpoint "$checkpoint" --output "data/engine/wp6/gen-$checkpoint.json"
  CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.generated_gtf \
    --phase generate --checkpoint "$checkpoint"
  CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.generated_gtf \
    --phase noise --checkpoint "$checkpoint"
  CUDA_VISIBLE_DEVICES='' timeout 28m python -m tah.engine.tests.generated_gtf \
    --phase compare --checkpoint "$checkpoint"
done
CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.engine_check \
  --gate perf --batch-size 64 --output data/engine/wp6/wperf.json
CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.engine_check \
  --gate robust --batch-size 64 --max-new-tokens 16384 --output data/engine/wp6/robust.json
CUDA_VISIBLE_DEVICES=0 timeout 28m python -m tah.engine.tests.engine_check \
  --gate long-prompt --max-new-tokens 8 --output data/engine/wp6/long-prompt.json
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.engine_check \
  --gate swap --output data/engine/wp6/forced-swap.json
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.engine_check \
  --gate profile --batch-size 64 --output data/engine/wp6/profile-b64.json
```

Tool options: `--gate` is required (`gen`, `tf`, `perf`, `robust`, `long-prompt`,
`swap`, `profile`); `--output` is the required JSON destination; `--checkpoint` chooses
`real` (default), `mlp3`, `mlp1`, or `label3` for parity; `--batch-size` defaults
to 128 for workload/profile runs; `--max-new-tokens` defaults to 4096 for
workloads; `--start`/`--end` select the expanded 128-prompt workload range
(default 0/128); `--memory-fraction` defaults to .9 for workload runs.
`--profile-context` defaults to 2048: profile with repeated real prompts at that
context length; set 0 to use the original prompt lengths. Greedy parity
uses the six saved prompts with a 256-token cap. `engine_check --gate tf` retains
the earlier fixture-limit diagnostic; use the separate ordered generated-corpus
commands above for the current generated-sequence gate.

`generated_gtf` requires `--phase` (`generate`, `noise`, `compare`) and
`--checkpoint` (`real`, `mlp3`, `mlp1`, `label3`). `--root` defaults to
`data/engine/wp6`; `--min-generated-positions` defaults to 8192 and controls
corpus generation. Comparisons require at least 8192 processed generated
positions per checkpoint. Generation repeats the six prompts in groups of six
with distinct request seeds and the same temperature .6, top_k 20, top_p .95
sampling settings; the original real-checkpoint six sequences are reproduced
and checked against the saved `tf-real.json`.

Generation saves the corpus manifest, token IDs and observed engine traces under
`gtf-corpus/<checkpoint>/`. The noise phase reads token IDs and computes
reference-only layout A versus B before any engine comparison: A uses singleton
prompt chunks of 128 and cached single-token decode; B uses a batched, left-padded
full prompt with aligned cached decode, retaining completed rows with EOS-filled
tails for batch shape and excluding those tails from metrics. It saves layout A
traces and preregisters `gtf-limits/<checkpoint>.json`. The compare phase verifies
corpus/oracle/source hashes, consumes those saved limits, and writes
`tf-<checkpoint>-expanded.json` with PASS/FAIL, decisions, pass-count and
full/retained-logit metrics. Neither calibration nor comparison overwrites an
existing corpus or limit registration; choose a new root for an independent run.
Earlier failing reports remain separate evidence.

Generated-corpus limits are 1.5 times the measured reference-only errors/rates.
MLP decision/pass-count reference counts of zero or one use the specified floor
of two; IterLabel and depth caps remain exact. Margin uses the maximum distance
from threshold on both sides of a reference flip, with the .01 minimum. Fixture
validation continues using its original measured limits.

`reference_capture.capture` accepts optional `chunk_boundaries`: strictly
increasing padded-column boundaries from 0 to the padded input width. Optional
`prompt_lengths` aligns left-padded prompt prefixes and their subsequent decode
positions when replaying generated batches. Invalid values raise `ValueError`;
default fixture capture behavior is unchanged.

`--gate swap` checks production-weight output/pass-count identity across a
forced mid-generation swap and immediate restoration at unchanged batch
composition, then audits KV and host swap cleanup.

The workload tool reconstructs Math500 prompts from the saved reference
W-perf samples and tokenizer's chat template, then repeats each prompt twice.
Construction and a warmup run are excluded from measured wall time. A report
contains request results, memory reservations/peaks, KV audit, live host swap
bytes, tokens/s and the reference 78.26 tokens/s baseline. Split workload ranges
when necessary to keep each invocation bounded; report the sum of tokens divided
by the sum of measured times, and label batching differences. Long-prompt mode
uses one 16,384-token prompt plus 32 short requests. Profile mode counts actual CUDA activities once, cross-checks them against
the saved trace, reports CUDA kernel and copy time separately, and saves
an inspectable `.trace.json` beside the report. It also records unprofiled step
wall times. The profiler records the first eleven decode steps after actual
model prefill, with admission headroom set to zero for this diagnostic; observed
batch sizes are recorded in the report. Prefill is excluded from the trace. The profiled wall-minus-device residual includes Python/launch gaps
and profiler overhead, with overlap caveats.

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
supplied. CUDA component capture/replay and pinned transfers have been
validated on the RTX 3090; production cached parity is checked separately.
Both 256-token chunks and token-by-token replay passed the production
reference comparison for all four fixture checkpoints. The separate comparison
with the full eager forward has one recorded failure: the single-pass
checkpoint's token-by-token retained dense-row max error is 1.625 versus its
1.359375 limit. A focused probe reproduces that difference in the reference's
single-token versus full-sequence execution, with exact matching outputs for
each corresponding layout. The recorded failure remains unresolved.

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
is exhausted, create a larger bucket outside capture. `prepare(rows)` uploads
small table-pointer/length descriptors, stages all real entries in one GPU
kernel, and plans FlashInfer outside capture. Reprepare after allocation,
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
(default 10), `--batches` (one or more positive integers; default 1/8/32),
`--avg-context` (optional integer >=16), `--model-path` (optional local checkpoint,
CUDA only), and required `--output`. Without `--avg-context`, request lengths
are 31+17*row; with it they vary by +/-25% around the chosen mean. It records
metadata construction, prepare wall time, decode, eight-query prefill,
contiguous gather/scatter and final pool audits. Times are microseconds;
CUDA additionally records preparation including completion, separate metadata
and attention graph replay, their combined replay and peak allocated memory.
Select CUDA only when the GPU is free.

For representative contexts and full-model pass timing, after the GPU is free:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.attention_check --device cuda \
  --batches 64 --avg-context 2048 --repeats 100 \
  --output data/engine/attention-context2k.json
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.attention_check --device cuda \
  --model-path models/TaH-plus-1.7B --batches 64 --avg-context 2048 --repeats 20 \
  --output data/engine/cached-model-context2k.json
```

`--model-path` measures all-layer cached passes instead of isolated attention.
It uses randomized old KV, sparse later streams (about 1/16 occupancy in the
second stream) and mixed row depths. The report adds model layer count, KV
bytes, eager/graph pass times and metadata-preparation/pass time ratio. A case
fails clearly if KV would use more than 85% of measured free memory after
loading weights. This times pass-local logits/hidden states; it excludes
continuation decisions, soft-input updates and sampling and is not generation
throughput or an accuracy check.

On the RTX 3090, B=64 with mean context 2048, metadata preparation measured
0.699 ms against a 28.474 ms full-model cached pass graph (2.45%). This proxy
uses 28 layers and 139,354 KV entries, with exact eager/graph logits and hidden
states. It supports the metadata budget assessment for that shape; complete
generation throughput and L40S measurements remain separate checks.

`metadata_check` requires empty CUDA visibility, `--mode interpret|compile`,
and `--output`. Interpretation checks exact ragged visibility/compaction on CPU.
Compilation builds all three kernels offline for sm80/sm86/sm89 using the installed
Triton/toolchain and never loads a CUDA module. Reports include
`production_gates_eligible=false`; CPU correctness and timing do not establish
production model parity, GPU performance or portability.

Cached production validation, when the GPU is available:

```bash
source .venv-tah/bin/activate
for chunk_size in 256 1; do
  for checkpoint in real mlp3 mlp1 label3; do
    CUDA_VISIBLE_DEVICES=0 python -m pytest -q -s -o addopts= -m 'gpu and slow' \
      "tah/engine/tests/test_cached_gtf.py::test_cached_production_gtf[${chunk_size}-${checkpoint}]"
  done
done
```

These are long validation runs: about four minutes per chunked case and
9–13 minutes per token-by-token case on the measured RTX 3090. They require
complete eligible production fixtures, verify hashes, and derive the same
reference decision/margin/pass-count limits as the full-forward gate above.
Reports are written to `data/engine/wp4/gtf/chunk{1,256}/`: `<checkpoint>.json`
compares with the reference; `<checkpoint>-vs-wp3.json` compares with the full
eager model. Each includes PASS/FAIL, comparisons, summary, `noise_limits` and
`decision_limits`. A failed paired retained-coordinate assertion remains FAIL;
neither limits nor reference files are changed by the test.

## Graph-safe sampling

`GraphSampler` implements per-row temperature, min-p, top-k, top-p and greedy
sampling with `RequestRNG` state. A bounded sampler such as
`GraphSampler(vocab_size, max_top_k=20)` selects one top-20 shortlist for each
large-vocabulary CUDA row, then applies the filters on those 20 candidates.
Sampled rows in that bucket must have `1 <= top_k <= 20`; `top_k=0` or a larger
value uses the general full-vocabulary path. Keep these as separate fixed-shape
buckets. Within the selected shortlist, equal scores use token-ID-ascending
ordering. At the outer CUDA top-K boundary, tied membership follows the pinned
PyTorch 2.6 kernel for that shape and dtype; rerun the tie tests when changing
the execution environment. Bounded top-p uses FP64 shortlist arithmetic. A
`top_p` chosen exactly from a full-vocabulary FP32 reference CDF can select an
adjacent nucleus because the reference rounds its full-vocabulary reductions;
strict parity for those deliberately rounded cutoffs is still open.
`temperature=0` is greedy.

Create parameter and request state before graph capture. When requests move,
move their sampling parameters, seed and counter with them; inactive rows return
`-1` and do not advance their counter. Update parameter values in place so the
captured tensor addresses stay fixed:

```python
from tah.engine.sampling import GraphSampler, RequestRNG, SamplingParams

sampler = GraphSampler(vocab_size, max_top_k=20)
batch = sampler.make_batch(per_request_params, device="cuda")
rng = RequestRNG.from_seeds(per_request_seeds, device="cuda")
# Capture sampler.sample(final_logits, batch, rng, active) with fixed buffers.
```

Use `GraphSampler(vocab_size, greedy_only=True)` for a separately captured
all-greedy bucket. Sampling does not create a `generate` API; the engine caller
owns request lifecycle and carries RNG state through scheduling and swaps.
