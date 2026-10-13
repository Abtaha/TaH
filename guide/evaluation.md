# TaH evaluation

The entrypoint is `script/tah/evaluation/eval.py`. From the repository root,
activate `.venv-tah` with the `tah,evaluation,engine` extras installed. CLI help is
safe while another process owns the GPU:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python script/tah/evaluation/eval.py --help
```

The generation examples below are **GPU jobs**; run them only when the selected
GPU is available. `CUDA_VISIBLE_DEVICES=0` explicitly selects GPU 0 even if the
shell previously hid GPUs. The full driver is not a CPU-only command.

## Run a small TaH evaluation

Create a local smoke config from the shipped 1.7B recipe, with smaller batches
and generation cap, then evaluate the first five GSM8K problems:

```bash
source .venv-tah/bin/activate
python - <<'PY'
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path('script/tah/recipes/qwen3_1.7/eval_tah.yaml').read_text())
cfg.update(batch_size=1, repeat_size=1, max_new_tokens=512)
Path('/tmp/tah_eval_smoke.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
PY
CUDA_VISIBLE_DEVICES=0 python script/tah/evaluation/eval.py \
  --eval_config /tmp/tah_eval_smoke.yaml \
  --model_path nics-efc/TaH-plus-1.7B \
  --dataset_name gsm8k --backend tah_engine \
  --job_nums 1 --tp_size_per_job 1 --data_range 5 \
  --output_dir data/eval_smoke
```

`--model_path` accepts a local checkpoint directory or a Hub ID. A Hub ID may
download on first use. The 0.6B recipe is
`script/tah/recipes/qwen3_0.6/eval_tah.yaml`; use it only with a compatible 0.6B
checkpoint. The shipped recipes use **repeat 1**, batch 16, max_new_tokens 4096,
temperature 0.6, top_p 0.95, top_k 20, min_p 0, max_iter 2, threshold 0.9.
For the reference backend (`--backend tah`), batch 16 is not guaranteed to fit long prompts on a 24 GB GPU. The reference
backend left-pads each batch and can allocate large prompt × batch × vocabulary
buffers; reducing batch size is the main available memory control.

## Backends

| `--backend` | Behavior |
| --- | --- |
| `tah_engine` (default, recommended for TaH v1) | Engine-owned BF16 Qwen3/LoRA/decider execution on one GPU, bounded ragged prefill, continuous batching, CUDA graphs and pinned host swap. Receives every job prompt once, writes each completion incrementally, and returns exact generated-token pass counts. |
| `tah` | Correctness reference `TaHForCausalLM` and its iteration-aware generation helper. Loads TaH config/LoRA/decider, uses SDPA, returns generated-token pass statistics. Uses `device_map="auto"`; no SGLang engine is involved. |
| `hf` | Standard `AutoModelForCausalLM.generate`, useful for ordinary base-model baselines. Uses `device_map="auto"`, `trust_remote_code=True`, optional FlashAttention 2. Does not reproduce the TaH selective-iteration wrapper or supply pass counts. |
| `sglang` | `sglang.Engine` with configured dtype, TP size, memory fraction, attention backend, and sampling. Use checkpoints supported by SGLang; this adapter does not load the TaH reference wrapper/decider and does not supply pass counts. |

Use `--backend tah` to run the intact reference model and `TaHForCasualLM_generate`.
Both TaH backends read the same sampling and model overrides from eval YAML.
If engine prerequisites fail, the error points to `--backend tah`. See
[engine options](engine.md#generate-with-the-engine).
HF and SGLang perform a throwaway 100-token generation warmup; the `tah`
adapter does not call that warmup helper. Each job loads one backend for its
assigned problems across all requested datasets, then cleans it up.

## Datasets

Names must appear in
[`dataset_configs.json`](../tah/evaluate/eval_configs/dataset_configs.json).
Hub datasets load through `datasets`; local paths are relative to the repo
root and must already exist. A missing requested split falls back to `train`,
then `test`. All supported benchmarks have a registered grader.

| Name | Source | Configured split | Scoring |
| --- | --- | --- | --- |
| `aime25` | `yentinglin/aime_2025` | `train` | Expression accuracy |
| `aime24` | `Maxwell-Jia/AIME_2024` | `test` (default; fallback if absent) | Expression accuracy |
| `gsm8k` | `pss0204/gsm8k_test`, subset `gsm8k_test` | `test` | Expression accuracy |
| `math500` | `HuggingFaceH4/MATH-500` | `test` | LaTeX/math accuracy |
| `amc23` | `math-ai/amc23` | `test` | LaTeX/math accuracy |
| `olympiadbench` | `realtreetune/olympiadbench` | `test` | LaTeX/math accuracy |
| `minerva` | `data/initial_data/eval/minerva.jsonl` | `test` default; JSON loader falls back to `train` | LaTeX/math accuracy |
| `gpqa` | `data/initial_data/eval/gpqa_diamond.jsonl` | Same local-JSON fallback | String/multiple-choice accuracy |
| `arc_e` | `data/initial_data/eval/arc_e.jsonl` | Same local-JSON fallback | String/multiple-choice accuracy |
| `arc_c` | `data/initial_data/eval/arc_c.jsonl` | Same local-JSON fallback | String/multiple-choice accuracy |
| `mmlu_stem` | `data/initial_data/eval/mmlu_stem.jsonl` | Same local-JSON fallback | String/multiple-choice accuracy |
| `mmlu_redux` | `data/initial_data/eval/mmlu_redux.jsonl` | Same local-JSON fallback | String/multiple-choice accuracy |
| `humaneval` | `data/initial_data/eval/humaneval.jsonl` | Local rows in `train` | EvalPlus base and plus pass@k |
| `mbpp` | `data/initial_data/eval/mbpp.jsonl` | Local rows in `train` | EvalPlus base and plus pass@k |
| `livecodebench` | Registered source `livecodebench/code_generation_lite`, version `v5` | — | **Unsupported**: no repository grader; rejected before loading. |

`brumo25` and `chmath` have math grader entries but no dataset registry entries,
so they are not selectable. Unknown names and duplicate requested names fail.
LiveCodeBench cannot be evaluated by applying HumanEval/MBPP tests.

Math/expression prompts use the tokenizer's user chat template with a generation
prompt. The graders parse/verify model answers using `math_verify`; unextracted
answers are incorrect. Code prompts prime a self-contained Python script in a
markdown code block (`reasoning=False`); outputs are sanitized using each task's
`entry_point`. If a code tokenizer has no chat template, the task prompt is used
directly. Each dataset is scored by its own grader.

## Multi-dataset sessions, ranges, and repeats

Use comma-separated `--dataset_name` to share backend loading across benchmarks.
This example evaluates Math500 rows `[0,8)` and all GPQA rows, with the smoke
config prepared above:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python script/tah/evaluation/eval.py \
  --eval_config /tmp/tah_eval_smoke.yaml \
  --model_path nics-efc/TaH-plus-1.7B \
  --dataset_name math500,gpqa --backend tah_engine \
  --dataset_ranges math500=0:8,gpqa=all \
  --job_nums 1 --tp_size_per_job 1 \
  --output_dir data/eval_multi
```

Selection operates on each dataset's loaded row order, before job sharding:

- `--data_range 8` selects `[0,8)` **in each dataset**.
- `--data_range 8 16` selects `[8,16)` in each dataset; end is exclusive and
  clamped to dataset length. Start must be nonnegative and end must exceed start.
- `--dataset_ranges math500=0:8,gpqa=all` overrides the shared range for those
  datasets. Unlisted datasets use `--data_range`, or all rows if it is absent.
  Names must be requested; repeated names or malformed ranges fail.
- `--data_ids gsm8k_0,gsm8k_5` overrides both range selections. IDs are
  `<dataset>_<original_id>`; if a source has no ID, its loaded row index is used.
  Duplicate IDs are removed and unmatched IDs silently skipped; no matches is
  an error. Range syntax is still parsed/validated before ID selection.
  HumanEval source IDs contain a slash, e.g. `humaneval_HumanEval/0`.

YAML `repeat_size` creates that many samples per problem, indexed
`0..repeat_size-1`. Repeats are adjacent prompts, so `batch_size` counts
**samples**, not problems. Use positive integers for both values; there is no
comprehensive YAML schema validation. Greedy repeats can be identical.

Selected problems are distributed round-robin across jobs; all repeats of a
problem stay in its job. The current list-returning backends batch by dataset
within that job and flush completed results after each returned batch. The
harness also supports callback backends: they receive all job prompts once and
persist each indexed result as callbacks arrive (possibly out of order), with
duplicate/out-of-range/missing callbacks treated as errors. `tah_engine` uses this callback mode and keeps one engine for all datasets
in its job, including startup captures. Math grading can run in processes while
generation continues; raw outputs are persisted before grading, and CSV rows
are flushed as grades finish.

## Every evaluation CLI option

| Flag | Default / required | What happens |
| --- | --- | --- |
| `-h`, `--help` | — | Print help and exit before running evaluation. |
| `--eval_config PATH` | Required | Load a YAML mapping; copy the original YAML into every dataset run directory. |
| `--model_path PATH_OR_ID` | Required | Model and tokenizer checkpoint path or Hub ID. |
| `--backend {sglang,hf,tah,tah_engine}` | `tah_engine` | Select one of the implementations above. |
| `--engine-options JSON` | None | JSON object merged over YAML `engine`; keys are all constructor options in [engine.md](engine.md). |
| `--engine-max-batch-size INT` | None | Override `engine.max_batch_size`, the maximum resident requests (engine default 64). |
| `--engine-host-swap-gib FLOAT` | None | Override `engine.host_swap_bytes` in GiB (engine default 8). |
| `--engine-memory-fraction FLOAT` | None | Override `engine.memory_fraction` (engine default .9). |
| `--max-new-tokens INT` | None | Override top-level YAML `max_new_tokens` for all backends and requests. |
| `--grading-workers INT` | None | Override YAML `grading_workers`; 0 grades inline, positive counts grade in CPU processes. Engine default 4, other backends 0. |
| `--dataset_name NAMES` | Required | One name or comma-separated names; whitespace is trimmed. |
| `--job_nums INT` | `1` | Parallel worker/model instances; jobs with no assigned problems are skipped. Use a positive count. |
| `--tp_size_per_job INT` | `1` | GPUs assigned to each worker. SGLang uses actual tensor parallelism; HF/TaH ignore this as a TP parameter and use auto device placement within their visible GPUs. Use a positive count. |
| `--output_dir PATH` | `<model_path>/eval_results` | Parent for dataset/backend/timestamp folders; explicit paths also work for Hub IDs. |
| `--data_range END` or `--data_range START END` | All rows | Shared per-dataset slice, end exclusive. Other argument counts fail. |
| `--dataset_ranges SPECS` | None | Per-dataset `name=start:end` or `name=all` overrides, comma-separated. |
| `--data_ids IDS` | None | Specific standardized IDs, comma-separated; overrides row selection as described above. |
| `--del_job_dir VALUE` | `True` | Delete `job_<id>/` after all workers and coverage checks succeed. Uses argparse `type=bool`: **any nonempty string, including `False` or `0`, is true**. To retain successful worker directories from this CLI, use `--del_job_dir ''` (empty argument). Failed runs preserve them regardless. |
| `--logger_level LEVEL` | `WARNING` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`, case-insensitive. Unknown values fall back to WARNING; progress/print output still appears. |
| `--random_seed INT` | `42` | Same seed in every worker for Python, NumPy, torch and CUDA; engine request seeds are the base seed plus the job prompt index; job IDs do not offset it. Batch composition, backend and sharding can change outputs despite the same seed. |

## Evaluation YAML keys

The driver reads a flat mapping. It does not read dataset lists, model paths,
job counts, output roots, or seeds from YAML: set those via CLI. Unknown keys
are generally unused; misspelled required keys can produce a worker `KeyError`.
TaH additionally forwards all recognized `TaHConfig` fields below.

| Key | Used by | Default / effect |
| --- | --- | --- |
| `temperature` | All | Required. Sampling temperature; `tah`/`hf` sample iff >0, otherwise greedy. SGLang receives the value directly. Use 0 for greedy, positive for sampling. |
| `top_p` | All | Required. Nucleus mass threshold; reference range [0,1], where 0 keeps one maximal token and 1 disables nucleus filtering. HF/SGLang enforce their own accepted range. |
| `top_k` | All | TaH defaults to 0 (disabled); HF/SGLang omit it if absent/null and use their own defaults. For TaH supply a nonnegative integer; positive k clamps to vocabulary size and retains cutoff ties. Do not pass null to TaH. |
| `min_p` | All | TaH defaults to 0 (disabled); HF/SGLang omit it if absent/null. Reference keeps probabilities >= min_p × maximum probability. Use [0,1]; do not pass null to TaH. |
| `max_new_tokens` | All | Required. Generation cap per sample; EOS can stop earlier. Use a positive integer. |
| `batch_size` | Reference/HF/SGLang | Required by the harness. Samples per fixed batch; engine resident cap is `engine.max_batch_size` and defaults to 64 independently of this key. |
| `repeat_size` | Harness | Required. Samples per selected problem; positive integer. |
| `dtype` | All | `bfloat16`. HF/TaH look up `torch.<name>`; SGLang receives the string. The engine requires `bfloat16`. Other backends use a dtype the model/GPU supports (e.g. `bfloat16`, `float16`, `float32` for torch). |
| `mem_fraction_static` | SGLang | `0.90` (recipes: 0.8). Static-memory fraction passed to SGLang; **ignored by HF/TaH**, so lowering it will not fix reference OOM. |
| `attention_backend` | SGLang | `triton`. Forwarded to SGLang; availability is determined by the installed SGLang version. |
| `engine` | `tah_engine` | Optional nested mapping of all `TaHEngine` constructor options; see below. Unknown options fail. |
| `grading_workers` | Harness | Nonnegative process count; default 4 for engine, 0 for others. Code test execution already uses the EvalPlus process pool. |
| `use_flash_attention` | HF | False if absent. Truthy value selects `flash_attention_2` and requires flash-attn. Ignored by TaH (always SDPA) and SGLang. Use YAML booleans, not quoted strings. |

Reference sampling order is temperature → softmax → min-p/renormalize →
top-k/renormalize → top-p/renormalize → multinomial. Top-p keeps the first token
that crosses the mass threshold, and at least one token. Greedy uses argmax
and ignores these filters. HF and SGLang use their own sampling implementations;
these settings do not guarantee identical tokens across backends.
Sampling is shared across requests; the driver derives `do_sample` from
temperature. Stop strings and individual request seeds are not CLI options.
Engine memory, graph and compile options are accepted through `engine`.

### TaH model override keys

Both `tah` and `tah_engine` consume these keys, defined in
[`TaHConfig`](../tah/model/tah_config.py). The saved `tah_config.json` is the
starting point. The adapter constructs a `TaHConfig` from recognized YAML
fields; non-None and nonempty-dict values replace saved fields, **without
recursive dict merging**. Scalar dataclass defaults (`embedding_key` and
`adapter`) also participate even when omitted in YAML. None/empty dict keeps
the saved value. Avoid partial architectural overrides that differ from the
trained checkpoint; they can cause weight/shape errors. The recipe's complete
decider dictionary is deliberate.

| Key | Meaning / dataclass default when no saved value applies |
| --- | --- |
| `embedding_key` | Base model embedding attribute path; `model.embed_tokens`. Invalid paths fail early. |
| `max_iter` | Maximum total passes per token, including the first; default None, so inherit/set a checkpoint-compatible integer >=1. |
| `iter_decider` | Trained decider class: `MLPIterDecider` or `IterLabelDecider`; None means use the saved value. |
| `iter_decider_kwargs` | Constructor/load overrides for that decider; default `{}`. Nested keys below. |
| `eval_iter_decider` | None reuses trained decider; `iter_decider` aliases it (recipes); an `iter_decider...` attribute path resolves on the wrapper. A class name builds a separate fresh decider, without loading trained weights for it. |
| `eval_iter_decider_kwargs` | Constructor settings only for a fresh eval-time class; ignored for None/attribute aliases; default `{}`. Same class-specific keys below, except the special trained-decider `load_path`. |
| `input_updater_kwargs` | Default `{}`; only `topk` is read (default 100). Mixes embedding rows using softmax of the top-k accumulated logits before another pass. Other entries are retained but inert. This is independent of generation `top_k`. |
| `adapter` | Only `lora` supported; default `lora`, other names fail. |
| `adapter_kwargs` | Default `{}`. LoRA constructor options forwarded to installed PEFT's `LoraConfig`, except `base_grad`/`adapter_grad` (both default true) control parameter trainability. Typical options include `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `bias`, `use_rslora`. Saved `lora/` weights/config are reattached on load; this is not a sampling control. |
| `train_loss` | `NextTokenPredLoss` or `IterDeciderLoss`; default None inherits the checkpoint. Constructed during loading; generation supplies no labels and computes no training loss. |
| `train_loss_kwargs` | Default `{}`; loss constructor kwargs. `NextTokenPredLoss` has no active knobs; `IterDeciderLoss` accepts `pos_weight` (None), `skip_last_iter` (true), `max_iter` (wrapper cap if omitted). No benchmark generation effect. |
| `eval_loss` | Optional loss class, default None reuses train loss; no benchmark generation effect. |
| `eval_loss_kwargs` | Default `{}`, same loss kwargs as above; no benchmark generation effect. |

Removed legacy selectors (`input_updater`, `output_updater`,
`iter_label_generator`, `iter_attention_mode`) are not active overrides;
unknown YAML/checkpoint fields are filtered. The wrapper has fixed top-k soft
input updates, additive logits, and duo attention.

### Decider dictionary keys

The following are **constructor defaults**, not necessarily the effective
checkpoint values. Explicit nested `max_iter` wins over the wrapper's fill-in
default. Keep both caps consistent. `IterLabelDecider` accepts only `max_iter`
(constructor default 3); no labels are supplied by benchmark generation, so it
stops after one pass. `MLPIterDecider` accepts every row below except `load_path`,
which is a loader option only in `iter_decider_kwargs`:

| Nested key | MLP constructor default | Effect |
| --- | --- | --- |
| `topk` | 100 | Top-k logit features projected into hidden width; separate from input updater and sampling k. |
| `hidden_states_size` | 1024 | Base hidden width; must match checkpoint/model. Recipes use 1024 (0.6B) or 2048 (1.7B). |
| `hidden_states_layer_nums` | `[16,20,24,28]` | HF hidden-state indices to concatenate; 0 is embedding output, interior k is residual after layer k−1, final index is normalized final output. Empty list selects last state. Out-of-range indices fail. |
| `hidden_dims` | `[256,512,256]` | MLP widths; must match saved classifier weights. |
| `expansion_factor` | 4 | Classifier block expansion multiplier; architectural. |
| `dropout_rate` | 0.3 | Classifier dropout, disabled in eval mode. |
| `normalize_input` | false | Layer-normalize selected hidden features and softmax logit features before projection; alters architecture/features. |
| `threshold` | 0.5 | Continue when sigmoid(decision logit) **>** threshold, unless cap reached. In BF16, recipe 0.9 is effectively 0.8984375. |
| `max_iter` | 3 | Decider pass cap; wrapper supplies its own cap only when this key is absent. |
| `dtype` | `torch.bfloat16` | Python constructor dtype; ordinary YAML strings are not converted to torch dtypes here. Leave this nested option inherited/omitted; use top-level `dtype` for the backend cast. |
| `load_path` | Absent | Directory containing an alternative trained `iter_decider.bin`. Popped before construction; otherwise load the checkpoint's decider file if present. Not an MLP constructor argument. |

When loading a decider, constructor args take precedence over colliding state
entries (e.g. threshold). The dict override replaces the saved dict, so replacing
only threshold can inadvertently reset architectural values to constructor
defaults. Preserve the rest of the checkpoint-compatible dict when changing it.
The reference backend casts the complete TaH model/decider to top-level dtype
after load. The engine loads BF16 weights and uses the same override precedence.

The driver adds private `_logger_level` and `_random_seed` entries from CLI;
YAML values with those names are overwritten. They are internal tags, not
additional user options.

### Engine and grading configuration

The same eval YAML works with `--backend tah` and `--backend tah_engine`.
Add this optional mapping for engine controls (reference ignores it):

```yaml
engine:
  max_batch_size: 64
  memory_fraction: 0.9
  host_swap_bytes: 8589934592  # 8 GiB, live pinned swap budget
  prefill_chunk_size: 128
  cuda_graphs: true
  compile: false
grading_workers: 4
```

Every constructor option in [engine.md](engine.md#generate-with-the-engine)
is accepted here. The default resident cap is 64, limited to the number of
selected prompts in the job before startup capture. Explicit `graph_batch_sizes`
are also capped to this range. `batch_size` remains the fixed batch size of the
reference, HF and SGLang backends. Top-level `max_new_tokens` governs every
evaluation request; `engine.max_new_tokens` is only the engine's plain-request
fallback. `engine.seed` defaults to `--random_seed` and sets the base of the
engine's per-request seed sequence. Use one GPU and `--job_nums 1` on a 3090.

CLI `--engine-options` merges over YAML, then the three explicit engine flags
win over both. For example, a 16k generation run with the usual resident cap
on a machine with about 30 GB RAM:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python script/tah/evaluation/eval.py \
  --eval_config script/tah/recipes/qwen3_1.7/eval_tah.yaml \
  --model_path models/TaH-plus-1.7B --dataset_name math500,gpqa \
  --backend tah_engine --job_nums 1 --tp_size_per_job 1 \
  --max-new-tokens 16384 \
  --engine-max-batch-size 64 --engine-host-swap-gib 8 \
  --engine-memory-fraction .9 --output_dir data/eval_long
```

See [long-context sizing](engine.md#long-generations-and-host-memory)
for prompt bounds and settings. Defaults are CUDA graphs enabled and compile off.
When a host snapshot would exceed the swap budget, the newest active request
restarts from its original prompt and seed instead of aborting the run. It keeps
its sample index and emits one final result; its discarded attempt is counted
in `engine_run.restarts`/`wasted_tokens` and per-sample raw JSON `timing`. FIFO
swap restore still comes first, retries precede fresh requests, and fresh
admission stays blocked while a retry waits. Long generation can keep cap 64;
smaller caps are an optional way to reduce wasted work. Individual requests
must still fit model context and the GPU pool.

For small VMs, graph generation needs one CPU core; eager/debug needs at least
two. Allocate separate cores for grading processes or use `--grading-workers 0`.
See [CPU affinity measurements](engine.md#cpu-cores-and-affinity).

With `grading_workers > 0`, a generated raw JSON appears immediately with
`grading_status: pending`; the same file is atomically replaced when its grade
is complete, then the CSV row is flushed. The job waits for every grade before
reporting success. `grading_workers: 0` grades within the callback. Metrics record
grading work, inline time or the final wait for parallel graders. Math-verify
runs in each process's main thread so its signal timeouts remain effective.
EvalPlus uses its existing process pool and reports separate grading wall time.

## GPU and job allocation

Set `CUDA_VISIBLE_DEVICES` to an explicit **comma-separated list of integer
physical GPU indices**. For example, `CUDA_VISIBLE_DEVICES=0,1` with job_nums 2
and tp_size_per_job 1 assigns one model instance to each GPU; job_nums 1 and
tp_size_per_job 2 gives one job both GPUs. Each worker sees its assigned GPUs
renumbered from zero. Do not use UUID/MIG identifiers with this integer parser.

If visibility is unset **or empty**, this driver assumes GPU indices 0..7;
it does not discover the actual count. It also does not validate that
`job_nums × tp_size_per_job` fits the supplied list. Use no more jobs/GPUs than
the explicit pool provides. Empty visibility is safe for `--help` and the
collector, but it is **not a GPU exclusion mechanism for full evaluation**:
worker startup rewrites the variable from the allocator's pool.

`tah_engine` requires `--tp_size_per_job 1`; it runs on one GPU.

Workers use spawn and allocate a free localhost NCCL port (starting at
29555 + job_id ×100), set `MASTER_ADDR`, `MASTER_PORT`, and
`SGLANG_NCCL_PORT`. SGLang binds localhost using that assigned port. On Linux,
workers attempt to terminate when the parent dies. There is no built-in job
queue, distributed multi-node mode, CPU-device flag, or automatic retry.

## Crashes and partial results

Each completed sample writes an atomic `sample_<idx>.json` and flushes its
worker CSV row. The reference, HF and SGLang adapters finish a whole generation batch before
returning; samples in an interrupted batch are not saved by the harness.
Callback backends can persist results individually as they arrive.

Worker exceptions caught during evaluation write `job_<id>.error.txt` beside
each dataset's run artifacts. The parent combines available worker data even
after failures, verifies exact `(problem_id, sample_idx)` coverage (including
repeats and duplicates), and marks the run `complete` or `failed` in
`run_status.json`. A failed worker, missing job report, or coverage mismatch
causes a nonzero exit and preserves partial results and worker directories.
A combination/code-grading exception records a traceback in status and is
re-raised. Failure in one job/dataset marks the whole session failed.

Status starts as `running` with completed_samples 0 and is updated at
combination/finalization, **not after each sample**. For live progress inspect
worker CSVs and console output. A hard kill, initialization failure outside the
worker's exception handler, or parent interruption may leave `running` status
and no error text/combined CSV. Older unmarked runs are not proven complete.

There is no general resume CLI or automatic OOM rebatching. A rerun gets a new
timestamp and starts the selected workload again. Preserve the old outputs,
reduce batch_size or select explicit IDs as needed, and verify repeats/coverage
before using partial output. The collector requires `--allow_incomplete` for
known failed/running combined results. See [results](results.md) for schemas,
code grading, and diagnostic collection.

## Long runs and terminal reconnections

Start long jobs in a **detached** tmux session, and monitor it through a read-only
client. Detaching protects a job from a disconnected terminal; a normal attached
client can still send Ctrl-C or shell-activation commands into the running job.

For example, with the smoke config prepared above, these commands start a
separate evaluation and attach without allowing keyboard input to the job:

```bash
mkdir -p data/eval_smoke_detached
tmux new-session -d -s tah-eval -c "$PWD" \
  "bash -c 'set -o pipefail; CUDA_VISIBLE_DEVICES=0 .venv-tah/bin/python script/tah/evaluation/eval.py --eval_config /tmp/tah_eval_smoke.yaml --model_path nics-efc/TaH-plus-1.7B --dataset_name gsm8k --backend tah_engine --job_nums 1 --tp_size_per_job 1 --data_range 5 --output_dir data/eval_smoke_detached 2>&1 | tee data/eval_smoke_detached/console.log'"
tmux attach-session -r -t tah-eval
```

Use `tmux capture-pane -p -S -30 -t tah-eval` for a noninteractive progress
snapshot. The session can disappear after the command finishes; inspect the
saved log and `run_status.json` to distinguish completion from interruption.
Before restarting an interrupted run, count persisted CSV/raw-output pairs,
preserve them, and select only missing IDs into a new output root. The original
command starts all its selected problems again; it does not resume a worker CSV.

Source of truth: [CLI](../script/tah/evaluation/eval.py),
[job runner](../tah/evaluate/jobs.py), [backends](../tah/evaluate/backends.py),
[config loading](../tah/model/tah_model.py),
[deciders](../tah/model/iter_decider.py), [sampling](../tah/model/utils.py).

## Evaluations for the research report

Evaluations whose results go into the report follow the fixed protocol in
[report/PROTOCOL.md](../report/PROTOCOL.md): which benchmarks, seeds, token
limits and statistical analysis are used, and how new checkpoints are compared
with the baseline.
