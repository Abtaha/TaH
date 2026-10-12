# Installation and environment checks

Run these commands from the repository root. Keep TaH and TaH2 in separate
environments: their torch, Transformers, Triton, and FlashInfer pins conflict.
The package declares Python >=3.10; the reference TaH environment uses Python
3.10, while the TaH2 setup uses Python 3.12 on Linux with CUDA GPUs.

## TaH environment

On a fresh checkout:

```bash
python3.10 -m venv .venv-tah
source .venv-tah/bin/activate
python -m pip install -e '.[tah,evaluation,engine]'
python -m pip check
```

For an existing environment, activate it before running scripts. Preserve the
TaH pins: torch 2.6.0, Transformers 4.52.4, PEFT 0.15.2. The CUDA suffix on torch
can vary with the installed build. No separate engine wheel index is configured
by this project; the evaluation extra does require Git/network access for its
EvalPlus VCS dependency.

| Extra | Purpose / dependencies |
| --- | --- |
| `tah` | Reference model/scripts: torch 2.6.0, Transformers 4.52.4, PEFT 0.15.2, Accelerate, datasets, PyYAML. |
| `evaluation` | Grading and backends: pylatexenc 2.10, sympy 1.13.1, math_verify, termcolor, SGLang >=0.4.6, EvalPlus pinned to commit `26d6d00bb1fd0fa37f39c99d5290da67891d1c5e` (the harness target, 0.4.0.dev44). Install alongside `tah`. |
| `engine` | torch 2.6.0, Triton 3.2.0, flashinfer-python 0.2.5, Ninja 1.11.1.3, safetensors 0.8.0, tokenizers 0.21.4. Current checks/components; see [engine status](engine.md). |
| `training` | W&B, DeepSpeed 0.17.1, flash-attn 2.7.3; install alongside `tah` for training. Also supplies flash-attn for HF's optional `use_flash_attention`. |
| `tah2` | TaH2 stack including torch 2.9.1, Transformers 4.57.3, Triton 3.5.1, flashinfer-python 0.6.4 and its train/eval/server dependencies; see `pyproject.toml` for the full pins. |
| `dev` | pytest >=6 and Ruff >=0.11 for development checks. |

Only select the extras needed for your workflow; installing the project without
extras installs its packages but no runtime dependencies. Reference evaluation
uses `.[tah,evaluation]`; the engine check additionally needs `engine`.
For training:

```bash
source .venv-tah/bin/activate
python -m pip install -e '.[tah,training,evaluation]'
```

## TaH2 environment

In a separate shell/environment:

```bash
python3.12 -m venv .venv-tah2
source .venv-tah2/bin/activate
python -m pip install -e '.[tah2]'
python -m pip check
```

Use it for `tah2/`, `script/tah2/`, and `bash/`, following the main README.
Do not add `tah` or `engine` to this environment.

## System prerequisites and the engine import check

The TaH engine targets Linux, a CUDA-enabled torch build, an NVIDIA GPU with
compute capability >=8.0 (BF16), and a driver supporting that torch CUDA build.
The RTX 3090 has been checked locally; L40S vGPU portability remains to be
verified. FlashInfer's generic PyPI build compiles kernels on first use. JIT
needs a CUDA Toolkit of the **same major version** as `torch.version.cuda`,
`nvcc`, a working C++ build toolchain, and Python development headers matching
the active interpreter. The NVIDIA driver alone does not supply this toolkit.
Install matching development headers through your system package manager
(e.g. `python3.10-devel` on Fedora or `python3.10-dev` on Debian/Ubuntu, where
available).

`nvcc` is searched on `PATH` first, then at `$CUDA_HOME/bin/nvcc` or
`$CUDA_PATH/bin/nvcc` (`CUDA_HOME` takes precedence). Header search checks the
interpreter's include directory, then `CPATH`, `C_INCLUDE_PATH`, and
`CPLUS_INCLUDE_PATH`. These variables can point at an already-installed toolkit
or matching headers; the check only detects `Python.h`, so ensure its version
really matches the interpreter.

When a GPU is available and idle, activate TaH and run:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python -c 'import tah.engine'
```

Importing `tah.engine` calls `check_environment()` unless
`CUDA_VISIBLE_DEVICES` is **exactly the empty string**. The check imports torch,
FlashInfer and `flashinfer.jit.has_prebuilt_ops`, then checks CUDA/GPU/toolchain
in this order:

| Error / leading message | Meaning and action |
| --- | --- |
| `TaH engine dependencies are missing` | An import failed; install `.[tah,evaluation,engine]` in the active environment and inspect the chained import error. |
| `TaH engine requires a CUDA-enabled PyTorch build` | `torch.version.cuda` is absent; use the project's CUDA torch stack in a Linux environment. |
| `... no usable NVIDIA GPU` | CUDA build exists but `torch.cuda.is_available()` is false; check driver compatibility and GPU visibility. |
| `... compute capability >= 8.0 (sm80)` | The current CUDA device is too old; select a supported GPU. |
| `... nvcc was not found` | JIT build needs the CUDA Toolkit; expose `nvcc` on `PATH` or set `CUDA_HOME`. |
| `... Python development headers were not found` | Install matching headers or expose their include directory with the header variables above. |
| `Could not run CUDA compiler ...` | The detected compiler failed to run; inspect the reported OS/compiler error. |
| `... nvcc ..., but PyTorch was built with CUDA ...` | Parsed CUDA major versions differ; select a matching toolkit through `PATH`/`CUDA_HOME`. |

The nvcc/header/version checks run only when FlashInfer lacks prebuilt ops.
Minor CUDA version equality is not required by this check. A successful import
checks these prerequisites, not kernel correctness, free-memory sufficiency,
or every possible compilation error. It does not implement a separate OS or
driver-version comparison.

For CPU-only help, saved-result processing, or engine development tooling:

```bash
export CUDA_VISIBLE_DEVICES=""
source .venv-tah/bin/activate
python -c 'import tah.engine, torch; assert not torch.cuda.is_initialized()'
python script/tah/evaluation/eval.py --help
python script/tah/evaluation/collect_results.py --help
```

This bypasses the automatic engine check, not its requirements for inference.
Explicitly calling `tah.engine.check_environment()` still performs the check
and will report no usable GPU. **Do not start the full evaluation driver to
test CPU-only operation:** its GPU allocator treats empty visibility as GPUs
0..7 and assigns them to workers (see [evaluation](evaluation.md)).

## Kernel check (GPU only)

After prerequisites pass and the GPU is free:

```bash
source .venv-tah/bin/activate
CUDA_VISIBLE_DEVICES=0 python -m tah.engine.tests.kernel_check
```

This module has no CLI options, including no `--help` parser. It JIT-compiles
as needed, checks BF16 FlashInfer against dense PyTorch SDPA with `rtol=0.02`,
`atol=0.015`, and prints torch/CUDA/FlashInfer/GPU information plus
`case`, `mode`, `mean_us`, `max_abs_error`. It exercises page-size-1 arbitrary
KV row indices, GQA 16/8 heads with head dimension 128, eager and CUDA graph
decode for both Qwen3 model labels, and arbitrary-mask prefill (q=4, kv=7).
It benchmarks with 20 warmups and 100 measured iterations, writes no result
files, and raises on a numerical mismatch or absent CUDA.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Wrong package versions/imports | Check the active interpreter with `python -m pip --version`, then `python -m pip check`; select `.venv-tah` or `.venv-tah2` as appropriate. |
| Missing re-exports after package layout changes | Re-run `python -m pip install -e '.[tah]'` in the TaH environment; editable install finders can cache the old layout. |
| FlashInfer compile fails after import passed | Inspect the compiler output, matching headers, C++ toolchain, toolkit paths, and writable JIT cache; first-use compilation is separate from the import check. |
| HF `flash_attention_2` missing | Leave `use_flash_attention` false (the default), or install the pinned training extra with its build prerequisites. This option affects only `hf`. |
| EvalPlus import mismatch | Install the `evaluation` extra's pinned commit, rather than replacing it with a different release. |
| Local benchmark file missing | Check the paths in the [dataset table](evaluation.md#datasets); the editable install does not fetch `data/`. |
| CUDA OOM while evaluating | Reduce YAML `batch_size`, especially for long prompts/repeats; see [crash behavior](evaluation.md#crashes-and-partial-results) before interpreting saved output. |

Source of truth: [dependency extras](../pyproject.toml),
[environment check](../tah/engine/__init__.py),
[kernel check](../tah/engine/tests/kernel_check.py).
