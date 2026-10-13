# Evaluation protocol for the report

This protocol was fixed before the confirmatory runs it describes; the git history of this file is the record of that. Deviations are reported as deviations. Results live in the numbered chapters under `report/` (format: [`AGENTS.md`](AGENTS.md)).

## Backends

- **Reference**: `TaHForCausalLM` with `TaHForCasualLM_generate` (`--backend tah`). The correctness oracle; its code is not modified.
- **Engine**: `tah.engine` (`--backend tah_engine`), CUDA graphs on, `torch.compile` off. The engine version is frozen for all runs of one comparison (baseline and every checkpoint compared against it). If the engine changes, the baseline is re-run.

## R1 — engine equivalence to the reference

Evidence is reported at two levels.

1. **Mechanism level**: teacher-forced per-token decisions and logits of the engine vs the reference, judged against the reference's own numerical self-noise (bf16, different execution layouts) on the same token sequences: dataset fixtures, model-generated sequences up to ~4k positions, and long-context position buckets. Sampler distributions are compared on real logits.
2. **End-to-end accuracy (confirmatory)**:
   - Workload: MATH500 problems [0, 200) and GPQA-Diamond (all 198), 2 samples per problem, `max_new_tokens` 4096, temperature 0.6, top_p 0.95, top_k 20, decider threshold 0.9 (config `script/tah/recipes/qwen3_1.7/eval_tah.yaml` with `repeat_size: 2`).
   - Runs: reference with seed 1234 (confirmatory run); engine with seeds 42, 1234, 2024. An earlier reference run (seed 42) and earlier engine comparisons are reported as exploratory; they motivated the confirmatory run.
   - Analysis unit is the problem. Per problem, accuracy is the mean over all samples of a backend (runs × repeats); d_p = engine − reference. The estimate is the mean of d_p over problems with a paired standard error from the problem-level spread. Primary analysis: both datasets pooled (398 problems). Secondary: per dataset.
   - Equivalence test (TOST): the 90% confidence interval of the pooled difference must lie inside **±3 percentage points**. Also reported: reference vs reference (the two reference runs, same analysis) as the scale of run-to-run noise; mean iterations per generated token (engine/reference ratio within ±10%).
   - If the interval is not inside the margin, the result is reported as "equivalence not shown", with the estimate. Runs are not repeated until a test passes.

## R2 — reproduction of the published TaH+ 1.7B results

Reference: Fu et al., *Think-at-Hard*, arXiv:2511.08577 (v4), §6.1 and Table 2 (row TaH+, 1.7B).

| Benchmark | `max_new_tokens` | Samples per problem | Paper (TaH+ 1.7B) |
|---|---:|---:|---:|
| AIME25 | 8192 | 8 | 15.4 |
| AMC23 | 8192 | 8 | 48.4 |
| OlympiadBench | 8192 | 1 | 37.6 |
| MATH500 | 8192 | 1 | 72.6 |
| GPQA-Diamond | 8192 | 1 | 39.4 |
| HumanEval+ | 8192 | 1 | 51.5 |
| MBPP+ | 8192 | 1 | 57.5 |
| GSM8K | 4096 | 1 | 84.5 |
| MMLU-STEM | 4096 | 1 | 76.6 |
| Average | | | 53.7 |

- Checkpoint `nics-efc/TaH-plus-1.7B`; engine backend; zero-shot CoT prompts of the harness; pass@1. Sampling: temperature 0.6 (paper); top_p 0.95, top_k 20 (repository eval config; the paper states only the temperature).
- Seeds 42, 1234, 2024: three complete protocol runs.
- Reported per benchmark: mean over seeds ± standard error (problem level, samples pooled over seeds), the 9-benchmark average, and the paper value. A benchmark is "reproduced" if the paper value lies inside our 95% confidence interval; otherwise the gap is reported. The paper's single-sample numbers carry roughly 2–3.5 points of standard error themselves.
- The runnable form of this protocol is a protocol config under `script/tah/recipes/qwen3_1.7/`, executed by `script/tah/evaluation/run_protocol.py`.

## Later checkpoints (e.g. RL-trained deciders)

Evaluated with exactly the R2 protocol, the same seeds and the same frozen engine, and compared to the R2 baseline with the paired per-problem analysis of R1 (difference with confidence interval). They are not compared against reference-backend or paper numbers alone.
