# User guide

Start with [installation](installation.md), then [evaluation](evaluation.md) to run
TaH benchmarks and [results](results.md) to inspect or compare their outputs.
[Engine](engine.md) covers generation, memory controls and validation of the TaH inference engine.
These evaluation and results guides cover **TaH v1** (`tah/`); TaH2 has separate
entrypoints and dependencies. See the main [README](../README.md#tah2-usage) for
TaH2 training, offline/online evaluation, and serving examples.

Run commands from the repository root with the appropriate environment active.
GPU commands in this guide require an available GPU; CLI help and collecting
saved results can run with `export CUDA_VISIBLE_DEVICES=""`.

## Repository map

| Location | What lives here / when to use it |
| --- | --- |
| `pyproject.toml` | Editable package install and dependency extras; use separate TaH and TaH2 environments. |
| `tah/model/` | Reference TaH causal LM, selective iterations, LoRA, deciders, configuration, generation helper. |
| `tah/train/`, `tah/utils/` | TaH trainer, collator, and data preparation utilities. |
| `tah/evaluate/` | TaH dataset loading, four inference backends, job driver, math/code graders, results collector. |
| `tah/evaluate/eval_configs/dataset_configs.json` | Benchmark names, sources, splits, fields, and prompt templates. |
| `tah/engine/` | TaH v1 generation engine, continuous batching, graphs and swap; default eval backend. |
| `script/tah/playground/inference_example.py` | Engine generation demo (reference flag available) with per-token iteration colors; prints to the console. |
| `script/tah/evaluation/eval.py` | Benchmark CLI; see [evaluation](evaluation.md). |
| `script/tah/evaluation/collect_results.py` | Saved-results CSV collector; see [results](results.md). |
| `script/tah/preparation/` | Download training data, filter/split, label hard tokens, optionally prune a model. |
| `script/tah/train/SFT_TaH.py` | Two-stage TaH training entrypoint (`--config`); see the main README's training instructions. |
| `script/tah/recipes/` | Qwen3 0.6B/1.7B train/eval YAMLs, 1.7B single-GPU training variants, Accelerate configs; see [recipe index](../script/tah/recipes/README.md). |
| `tah2/` | TaH2 model, kernels, training, evaluation, utilities, and bundled `minisgl` engine. |
| `script/tah2/`, `bash/` | TaH2 data/train/eval recipes and launchers; use `.venv-tah2`. |
| `models/` | Local downloaded checkpoints; ignored by Git. TaH checkpoints include base weights/config/tokenizer, `tah_config.json`, `lora/`, and `iter_decider.bin`. |
| `data/` | Local training/benchmark data and generated artifacts; ignored by Git. |
| `data/initial_data/eval/` | JSONL benchmarks needed by the local dataset registry entries. |
| `data/raw_data/`, `data/initial_data/`, `data/processed_data/` | Training-data download, filtered splits, and labeled data respectively. |
| `data/engine/` | Saved evaluation experiments, engine fixtures and development reports; local artifacts, not installed package data. |
| `output/`, `wandb/` | Local generated outputs and training logs when selected by recipes/tools; ignored by Git. |
| `resource/` | README figures and logos. |

Evaluation defaults to `<model_path>/eval_results/`; `--output_dir` selects a
different root. For a Hub model ID this default is a **relative local directory**
(e.g. `nics-efc/TaH-plus-1.7B/eval_results`), not the Hub cache. Prefer an explicit
output root. Training uses the recipe's `data.output_dir`, adds a model/decider
and timestamp path (or `continue_training/<model>/<timestamp>`), and saves the
final checkpoint under `final_model/` with `training_config.yaml` at the run root.
TaH2 output paths are controlled by its separate recipes and launchers.

## Find the right instructions

| Task | Read |
| --- | --- |
| Install extras, resolve import/JIT errors, check kernels | [installation.md](installation.md) |
| Pick a backend/dataset, select rows, configure sampling/repeats/jobs | [evaluation.md](evaluation.md) |
| Read CSV/JSON files, check completion, collect summaries, compare runs | [results.md](results.md) |
| Generate with the engine, size memory, choose graphs, use validation tools | [engine.md](engine.md) |
| Train TaH, prepare labels, run the generation demo | Main [README](../README.md#tah-usage) and [recipes](../script/tah/recipes/README.md) |
| Train/evaluate/serve TaH2 | Main [README](../README.md#tah2-usage) and [mini-SGLang README](../tah2/minisgl/README.md) |
