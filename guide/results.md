# Reading and comparing TaH results

The evaluation driver writes one directory per dataset/backend/run under
`--output_dir` (default `<model_path>/eval_results`). Check completion and sample
coverage before interpreting accuracy. A CSV can exist after a failed run.

## Directory layout and files

```text
<output_root>/
  <dataset>_<backend>/
    YYYYMMDD_HHMMSS/
      [TASK_<start>_<end>/]       # present for a selected row range
        <eval_config_basename>   # copy of the original YAML
        run_status.json
        effective_config.yaml
        job_<id>.metrics.json              # successful job timing/engine telemetry
        detailed_results.csv
        evaluation_stats.csv
        samples.jsonl
        code_solutions.jsonl                # HumanEval/MBPP only
        code_solutions.eval_results.json    # after code grading
        job_<id>.error.txt                  # caught worker exception
        job_<id>/                           # retained on failure or by option
          detailed_results.csv
          evaluation_stats.csv             # written after this job succeeds
          details/
            <problem_id>/
              sample_<sample_idx>.json
  results_summary.csv                       # created by collector, not eval
```

All datasets in one invocation share the timestamp. Timestamps are produced
with the execution host's local clock; they are identifiers, not guaranteed UTC.
`TASK_0_8` means row indices `[0,8)` after dataset loading, not a problem-ID
range. Full-dataset runs and `--data_ids` selections omit `TASK_...`; end indices
are clamped to dataset length. Standardized IDs are
`<dataset>_<original_id>`; source IDs with `/` produce nested detail directories
(e.g. `humaneval_HumanEval/0/`).

| File | Contents / lifetime |
| --- | --- |
| Copied YAML | Original eval config. `effective_config.yaml` adds CLI overrides and seed/log tags; preserve both plus the invocation. |
| `effective_config.yaml` | Config actually sent to workers, including merged engine options and generation/grading overrides. |
| `job_<id>.metrics.json` | Successful job timing, grading work/wait, engine generation and callback wall scopes, allocated/reserved GPU peaks, startup memory budget, graph/compile and swap telemetry. Duplicated in each dataset directory for a shared job: do not sum the duplicates. |
| `job_<id>/detailed_results.csv` | Incremental completed-sample rows; flushed after each result. This is where to look during a run. |
| `job_<id>/details/.../sample_<idx>.json` | Raw output and grading detail written through a temporary file with fsync/atomic rename. Repeats use zero-based indices. |
| `job_<id>/evaluation_stats.csv` | Job-local problem/totals summaries, written only after its inference finishes successfully. |
| Run `detailed_results.csv` | Combined rows, sorted by problem ID and sample index; available after combination, including partial failure output. |
| Run `evaluation_stats.csv` | Combined problem/totals summaries. Code rows remain placeholders; consult code results for scores. |
| `samples.jsonl` | Combined sample JSON objects plus `id` and `sample` fields, one object per line. Built from worker detail files; may be partial on failure. Ordering follows jobs/detail-file traversal, not CSV sort order. |
| `run_status.json` | Completion/provenance metadata described below. Starts before workers run; not a live per-sample counter. |
| `job_<id>.error.txt` | Caught worker exception and traceback; kept at the dataset run level. An abrupt kill or early initialization failure may produce no such file. |
| `code_solutions.jsonl` | Code-only append log shared by workers, guarded with a file lock. Each row has original `task_id` and sanitized `solution`. It has no standardized problem ID or sample_idx. |
| `code_solutions.eval_results.json` | EvalPlus per-solution base/plus statuses and aggregate pass@k. Written after scoring; see below. |
| `results_summary.csv` | Collector output across runs; default at the root scanned, or explicit `--output_csv`. Overwritten on collection. |

Transient `sample_<idx>.json.tmp` or `run_status.json.tmp` may remain after a
crash; they are not finalized records. Successful runs delete worker directories
by default after coverage checks. Use `--del_job_dir ''` to keep them (the CLI's
boolean parsing makes `--del_job_dir False` true). Failed runs keep them
regardless. Combined `samples.jsonl` preserves raw outputs after successful
worker cleanup. Console logs need shell redirection. Engine token IDs and per-token counts are
saved in raw JSON; inference timing and GPU peaks are saved in job metrics.
Reference/HF/SGLang token IDs and full pass arrays are not exported.

## Every detailed CSV column

`detailed_results.csv` has the same columns at job and run levels:

| Column | Meaning |
| --- | --- |
| `problem_id` | Standardized dataset-prefixed problem ID. |
| `sample_idx` | Zero-based repeat index for this problem. Together with problem_id, the expected unique sample key. |
| `correct_answer` | Dataset ground truth as text. |
| `predicted_answer` | Parsed grader result as text; empty if no answer extracted. Code uses `pending_code_eval`. |
| `has_answer` | Python-style `True`/`False`; false for no extracted math answer and for deferred code rows. |
| `is_correct` | Python-style `True`/`False` from the math grader; false placeholder for code rows even after EvalPlus succeeds. |
| `input_tokens` | `len(tokenizer.encode(problem_text))`, **raw question only**, not the rendered/padded chat prompt. |
| `output_tokens` | `len(tokenizer.encode(output_text))`, retokenized decoded output; may differ from the exact generated token count due to decoding/special tokens. |
| `processing_time` | Backend-reported seconds: HF/TaH divide batch inference elapsed by batch outputs; SGLang attaches the whole batch elapsed to each output. CUDA events when CUDA is available, wall clock fallback otherwise. For the engine it is request latency including its queue time; overlapping latencies must not be summed for throughput. Excludes model load and helper warmup; does not measure end-to-end run wall time. |
| `mean_passes_per_token` | Mean generated-token pass count, six decimals; blank when unavailable/empty. |
| `fraction_tokens_at_least_2_passes` | Fraction of counted generated tokens using >=2 passes, six decimals; blank when unavailable/empty. |
| `pass_stat_tokens` | Number of generated tokens with pass counts; 0 when unavailable/empty. This is the weighting denominator, not output_tokens. |

Do not compare backend throughput by blindly summing `processing_time`: the
SGLang timing convention differs, concurrent jobs overlap, and grading/load/
other overhead are absent. Measure generation wall time separately for throughput.

## Job timing and engine telemetry

`job_<id>.metrics.json` records one shared job across every dataset it served.
`setup_seconds` covers model load and startup capture. `inference_call_seconds`
covers tokenization, one throwaway warmup, inference and callbacks.
`job_wall_seconds` measures from ready result writers through the final grading
wait. It excludes prompt construction, final statistics writing and cleanup.
`grading_work_seconds` sums measured math-grade durations (CPU process wall times
can overlap); `grading_inline_seconds` is blocking callback grading when workers
are zero; `grading_tail_seconds` is the final wait for all queued grades. Their
sum is not a total elapsed scope when grading runs in parallel.

Engine metrics also include `engine_options` (effective resident cap),
`memory_budget` (startup KV reservations and both graph capture rounds),
`engine_run` (generation wall/steps/swaps/host-swap peak), `callback_seconds`,
`generation_seconds`, and `peak_gpu_allocated_bytes`/`peak_gpu_reserved_bytes`.
`generation_seconds` is engine wall time minus synchronous callback time;
use exact `engine_run.generated_tokens / generation_seconds` for throughput
excluding load, warmup and persistence/inline grading. The engine wall scope
itself includes callbacks. Parallel grading still shares CPU resources with
generation. The GPU peaks are PyTorch allocated/reserved bytes after warmup,
excluding CUDA driver/module memory; startup capture device deltas are recorded
separately in `memory_budget`. Code grading follows generation and has its own
`run_status.code_grading_seconds`.

## Every statistics CSV column

`evaluation_stats.csv` contains one row per represented problem, a blank
separator, then a row with `problem_id = Total Accuracy`:

| Column | Meaning |
| --- | --- |
| `problem_id` | Problem ID, or the literal `Total Accuracy` for all completed samples. |
| `accuracy` | correct_count / total_samples, rounded to three decimals; sample-level totals, not pass@k. Code placeholder is zero. |
| `correct_count` | Number of rows with is_correct true. |
| `total_samples` | Number of completed sample rows for this problem/all problems. |
| `avg_output_length` | Arithmetic mean of output_tokens over those samples, rounded to two decimals. |
| `mean_passes_per_token` | Pass-token-weighted mean, six decimals or blank. |
| `fraction_tokens_at_least_2_passes` | Pass-token-weighted fraction, six decimals or blank. |
| `pass_stat_tokens` | Sum of counted generated tokens in those sample rows. |

For exact math accuracy use the integer counts; the stored `accuracy` and
collector's math score are rounded. Zero samples means no measurement, even
if a totals row displays zero. Missing repeats can inflate/deflate scores;
status/coverage matters.

For TaH, a generated token's reported pass count is the number of passes at
the **context position whose final logits sampled it**. This includes the last
prompt position's work for the first generated token; EOS is excluded. Counts
for all prompt positions are not exported by this harness. Both TaH backends supply these counts; HF/SGLang do not. Pass counts summarize compute,
not grading correctness.

For a set of rows with nonempty pass metrics and `pass_stat_tokens > 0`:

```text
aggregate metric = sum(row metric * row pass_stat_tokens) / sum(row pass_stat_tokens)
```

Rows without counts are excluded. This weights by generated tokens, not by
problems/samples or retokenized output length. Aggregation uses the already
rounded sample metrics, so tiny rounding differences from raw-count arithmetic
are possible. With max_iter 2, mean passes is approximately 1 + fraction >=2;
with higher caps that identity no longer holds.

## Raw sample JSON and JSONL

Each `sample_<idx>.json` has `problem_id`, `problem` (question text), `output`
(full decoded output), `correct_answer`, `predicted_answer`, and `is_correct`.
Code samples additionally have `extracted_code` and `entry_point`.
If the backend supplies pass counts, it adds `mean_passes_per_token`,
`fraction_tokens_at_least_2_passes`, and `pass_stat_tokens`; an empty count list
produces null metrics and 0 tokens. Engine results additionally include `token_ids`, `pass_counts` (generated),
`prompt_pass_counts`, `finish_reason`, and request `timing`. The reference
continues to export summaries without full arrays. During parallel grading,
an engine raw sample may have `grading_status: pending` and no grade yet;
that file is replaced when graded, then its CSV row appears. Pending samples
in a failed/interrupted run are ungraded evidence and do not count as completed.
The JSON contains no input/output token counts or processing_time; those are
in the CSV. Combined `samples.jsonl` keeps these fields and adds `id` (same as
problem_id) and `sample` (repeat index inferred from the filename).

## Run status and errors

| `run_status.json` field | Meaning |
| --- | --- |
| `status` | `running`, `complete`, or `failed`. Complete requires worker success and exact sample-key coverage; code grading also finishes before final status. |
| `model_path` | `Path(model_path).resolve()` recorded by the driver. For a Hub ID this is a resolved local representation of that string, not the actual snapshot download path. |
| `backend`, `dataset`, `timestamp` | Selected backend, this dataset, common invocation timestamp. |
| `expected_samples` | Selected problem count for this dataset × repeat_size. |
| `completed_samples` | Combined CSV row count, updated during combination, not while generation runs. |
| `batch_size`, `repeat_size`, `random_seed` | YAML batch/repeat settings and worker seed from CLI. |
| `completed_jobs`, `failed_jobs` | Final session-wide worker-report totals; failed_jobs also signals missing reports or coverage mismatch. Not necessarily specific to this dataset. Added on normal finalization. |
| `error` | Traceback on combination/code-evaluation failure; optional. Worker-specific traceback is in `job_<id>.error.txt`. |

`code_grading_seconds`, when present in status, is the parent EvalPlus wall time
for that dataset.

Status is atomically replaced. An interrupted process may leave `running`,
no combined summaries, and only worker artifacts. Known failed runs may still
have combined CSVs and even code grades. Keep them for diagnosis; they are not
complete benchmarks. Absence of status in older output is unverified, not
evidence of success. The collector cannot reconstruct exact coverage or detect
all corruptions; it checks completed totals but does not validate every raw
sample, ID, repeat, or config.

## Code-evaluation results

For HumanEval/MBPP, the driver sanitizes every output into
`code_solutions.jsonl`, then calls the repository's EvalPlus wrapper with
`allow_subset=True`. Subset runs are scored against the selected task IDs;
they are not full-benchmark scores. It uses base and plus tests, by default
half the CPU cores (at least one), default EvalPlus time limits, and fast checks
(one failure surfaced per failed sample). Ground-truth computations/data are
cached in EvalPlus's own `CACHE_DIR`, outside the evaluation output root.

`code_solutions.eval_results.json` contains:

| Field | Contents |
| --- | --- |
| `date` | Grading timestamp string from the execution host's clock. |
| `hash` | EvalPlus dataset hash. |
| `eval` | Mapping original task ID → ordered list of graded completions. |
| `eval[task_id][].task_id` | Original code task ID. |
| `eval[task_id][].solution` | Sanitized solution that was checked. |
| `eval[task_id][].base_status`, `.plus_status` | Evaluator status for each test suite (`pass` indicates success); plus can be null in base-only API evaluation. |
| `eval[task_id][].base_fail_tests`, `.plus_fail_tests` | Failed-input details (normally last failing input; empty on pass or when unavailable). MBPP inputs are serialized into JSON-compatible form. |
| `pass_at_k.base` | Mean per-task estimated pass@k on base tests. |
| `pass_at_k.plus` | Mean per-task estimated pass@k requiring both base and plus success. |

Keys `pass@1`, `pass@10`, `pass@100` appear only when **every evaluated task**
has at least k samples. The plus block is absent for base-only API calls.
For unequal numbers of samples per task, pass@1 is a mean of per-task rates,
which can differ from pooled correct_count / total_samples. CSV math/code
placeholders are never rewritten with these results. Use this JSON or the
collector's `pass@1 (base)` metric for code accuracy; inspect the JSON separately
for plus scores.

The wrapper also recognizes the legacy filename
`code_solutions_eval_results.json` if it already exists and can reuse previous
grading results. The collector looks only for
`code_solutions.eval_results.json`; legacy filenames/missing grades cause its
fallback to CSV `accuracy` (zero placeholders for code). Treat that fallback as
ungraded code, not zero measured pass@1.

## Results collector

Point `--results_dir` at the **output root containing dataset/backend folders**,
not at a dataset or timestamp directory. The collector recursively finds
combined `evaluation_stats.csv`, ignores any path component starting `job_`,
and writes one row per dataset/run directory.

After the [smoke evaluation](evaluation.md#run-a-small-tah-evaluation):

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python script/tah/evaluation/collect_results.py \
  --results_dir data/eval_smoke \
  --model_name TaH-plus-1.7B
```

For existing saved experiments under `data/engine/wperf_bs16` (when present),
this CPU-only example writes a separate summary without changing run files:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python script/tah/evaluation/collect_results.py \
  --results_dir data/engine/wperf_bs16 --model_name TaH-plus-1.7B \
  --output_csv /tmp/tah_wperf_summary.csv
```

For partial diagnostic output from the smoke root, use:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python script/tah/evaluation/collect_results.py \
  --results_dir data/eval_smoke --model_name TaH-plus-1.7B \
  --allow_incomplete --output_csv /tmp/tah_results_partial.csv
```

| Collector flag | Default / behavior |
| --- | --- |
| `-h`, `--help` | Print help and exit. |
| `--results_dir PATH` | Required. Existing output root; missing root raises FileNotFoundError. |
| `--output_csv PATH` | `<results_dir>/results_summary.csv`. Creates parent directories and overwrites output. |
| `--model_name TEXT` | Parent directory name of results root. Set explicitly for custom roots, otherwise `data/eval_smoke` would be labeled `data`. |
| `--allow_incomplete` | Absent by default. Include known failed/running runs for diagnosis. Without it, encountering any marked non-complete run aborts collection before writing a summary. |

Unmarked old runs are included even without `--allow_incomplete`, labeled
`legacy_unverified`. Explicitly complete runs whose chosen metric sample total
disagrees with expected_samples raise even with `--allow_incomplete`.
Directories with no combined stats are not reported; a root with no matching
files produces a header-only summary, not a completion check.

The Python API is also exported by `tah.evaluate`:
`collect_results(results_dir, output_file=None, model_name=None, allow_incomplete=False)`;
it writes the same CSV and returns the list of row dictionaries.

## Every collector CSV column

| Column | Meaning |
| --- | --- |
| `model` | Explicit label or root parent name. |
| `run` | Path beginning at timestamp, including a TASK suffix when present. |
| `dataset` | Parsed dataset folder prefix. |
| `backend` | Parsed suffix: `sglang`, `hf`, `tah`, or `tah_engine`. Unknown suffix becomes blank. |
| `metric` | `accuracy`, or `pass@1 (base)` when recognized EvalPlus JSON has that score. |
| `accuracy_or_pass_at_1` | Three-decimal stored accuracy for math; EvalPlus's base pass@1 value for code. |
| `correct_count` | CSV correct count for math; count of base-passing code completions for code. |
| `total_samples` | CSV completed row count for math; number of code completions in the grade JSON for code. |
| `problem_count` | Nonempty problem rows excluding totals for math; task count in grade JSON for code. |
| `avg_output_length` | Two-decimal CSV total average of retokenized outputs. |
| `mean_passes_per_token` | Six-decimal CSV weighted total, or blank. |
| `fraction_tokens_at_least_2_passes` | Six-decimal CSV weighted total, or blank. |
| `run_status` | Status metadata's value, or `legacy_unverified` when absent. |
| `expected_samples` | Status metadata count, blank for legacy runs. |

The collector does not add plus pass@1, pass_stat_tokens, CLI provenance,
configuration details, throughput or GPU memory columns. Keep source CSVs,
code JSON, copied YAML, status and invocation records alongside comparisons.

## Compare runs

1. Check status and expected/completed sample counts; verify identical
   `(problem_id, sample_idx)` coverage using detailed CSVs. For legacy runs,
   verify selection/repeats manually. Exclude partial runs from final comparisons.
2. Match checkpoint/tokenizer, dataset version/IDs, slice, repeats, sampling,
   max_new_tokens, dtype, decider cap/threshold, backend, batch/job configuration
   and seed. Copied YAML plus status covers some of this; retain the exact CLI
   and checkpoint revision for the rest.
3. Collect each output root, set a meaningful model label, and compare like
   datasets and metrics. Use code base pass@1 against base pass@1, and inspect
   plus scores separately. Do not mix subsets and full datasets.
4. Compute precise math accuracy from correct_count / total_samples; use
   output lengths and pass metrics to explain compute differences. Combine pass
   summaries with pass_stat_tokens weights from statistics CSVs. Measure wall
   time/GPU memory separately if comparing speed or memory.

To inspect the archived summary from the command above:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python - <<'PY'
import csv
with open('/tmp/tah_wperf_summary.csv', newline='') as f:
    for row in csv.DictReader(f):
        n = int(row['total_samples'])
        exact = int(row['correct_count']) / n if n else None
        print(row['dataset'], row['run'], row['run_status'], row['metric'],
              row['accuracy_or_pass_at_1'], 'count_ratio=', exact)
PY
```

Source of truth: [result writers](../tah/evaluate/jobs.py),
[code grader](../tah/evaluate/codeeval.py),
[collector](../tah/evaluate/results.py).
