# Evaluation internals

## Scope

`tah/evaluate/` loads benchmark rows, renders prompts, invokes backends and persists
graded samples. `script/tah/evaluation/` provides the CLI, protocol orchestration
and analysis. User-facing options live in [the evaluation guide](../guide/evaluation.md).

## Module map

- `datasets.py`: registry loading and standardized rows, including `_original_*` source columns.
- `jobs.py`: per-dataset prompts, indexed callbacks, durable samples, grading and combination.
- `backends.py`: reference, engine, HF and SGLang adapters; engine receives rendered prompt token IDs.
- `codeeval.py`: original harness code prompts, EvalPlus sanitizer export and unchanged base/plus graders.
- `code_prompts.py`: independent TaH2 reasoning templates and offline block extraction; no TaH2 runtime imports.
- `matheval.py`, `results.py`: math grading and result collection.
- `run_protocol.py`, `compare_runs.py`: protocol run/resume/merge, paired analysis and seed aggregation.

## Validating a change

Run CPU evaluation tests, including existing backend and durable-callback checks:

```bash
mkdir -p data/tmp/evaluation-tests
CUDA_VISIBLE_DEVICES='' TMPDIR="$PWD/data/tmp/evaluation-tests" .venv-tah/bin/python -m pytest -q \
  script/tah/evaluation/tests tah/engine/tests/test_eval_integration.py \
  --basetemp=data/tmp/evaluation-tests/pytest
```

`test_code_prompts.py` loads only the local checkpoint tokenizer (override its path
with `TAH_CODE_TOKENIZER`). It compares all 164 HumanEval and 378 MBPP prompts to
the actual imported TaH2 prompt function, through both dataset loaders, with Hub
fetches redirected to verified local files. It also checks pre-change harness
prompt digests, chat-template kwargs, reasoning extraction and worker-persisted
solutions for both variants. No model is loaded and no dataset download is needed.
These parity tests intentionally require TaH2's evaluation modules to import in
the test environment; production code never imports them.

Preflight a protocol without loading a model or launching evaluation:

```bash
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python script/tah/evaluation/run_protocol.py run \
  --config script/tah/recipes/qwen3_1.7/protocol_paper_code_tah2.yaml \
  --model_path models/TaH-plus-1.7B --backend tah_engine --seed 42 \
  --output_root data/tmp/evaluation-tests/plan-seed42 --plan-only
```

## Internals reference

- `code_prompt` defaults to `harness`; that branch still calls the original prompt
  builder with `reasoning=False` and sanitizes the whole output by entry point.
  Validate `harness | tah2` before backend setup. Math prompts are unaffected.
- `tah2` matches `tah2/evaluate/local_eval.py::process_batch_items`: strip the
  standardized question, leave assertion whitespace intact, render one user turn
  with `tokenize=False, add_generation_prompt=True`, and omit `enable_thinking`.
  This checkpoint's saved template ends at `<|im_start|>assistant\n`; it does not
  insert an open `<think>` prefix. Do not add one independently of the template.
- TaH2's registry points to `TaH-plus/eval_science_code/{humaneval,mbpp}.jsonl`.
  At revision `8737a89ea8d141dae003d33ca1f818de58b237b1`, those files are byte-identical
  to `data/initial_data/eval/` (whole-file SHA256 pinned in the parity test).
  Questions remain the entire `prompt`, including MBPP's triple quotes and embedded
  example. MBPP has a separate `assertion` field on all 378 rows; the loader carries
  it as `_original_assertion`, which `_build_problem` restores. Do not split the
  prompt to synthesize tests: the separate field contains additional test cases.
  HumanEval uses `test` → `_original_test` → assertion, unused by its template.
- Extraction matches the authors' actual offline call `extract_code(output_text)`
  **without** the optional entry point: first complete lowercase `python` fence
  containing a `def`, else last such fence, else empty. Thinking is not removed
  before selection, and no EvalPlus sanitization follows. Preserve this behavior
  even for draft code inside thinking, demos, missing fences or wrong names.
- Code grading remains deferred to EvalPlus, with original task IDs. Callback
  indexes and repeat indexes must survive out-of-order completion. Raw JSON text
  stays intact; only the submitted `solution` uses variant-specific extraction.
- Protocol group `overrides` merge onto the base eval recipe. Put `code_prompt`
  there, so derived configs and their resume hashes record the selection.
