# Analysis and reproduction

`analyze.py` reads the curated data and the editable `README.template.md`, checks
manifest hashes, recomputes numerical gates and paired statistics, and produces
README, the short per-layout appendix, booktabs tables and a deterministic PDF.
It needs only Python's standard library for the historical results. The optional
R1 adapter imports the repository comparator and PyYAML from `.venv-tah`.

```bash
python report/1-engine/analyze.py
CUDA_VISIBLE_DEVICES='' .venv-tah/bin/python -B report/1-engine/scripts/check.py
bash report/1-engine/reproduce.sh
bash report/1-engine/reproduce.sh equivalence --dry-run
```

The final two commands start no experiments. Named steps without `--dry-run`
execute their printed commands using `.venv-tah/bin/python`, write complete
outputs under `reproduction/`, extract selected metrics into `data/`, update the
manifest and rerun analysis. Use a free GPU and detached tmux for long steps.
`fixtures` must finish before `equivalence` or `negative`; use the same
`--output-root` for these steps. A fresh output root avoids overwriting a
previous corpus. Each subprocess keeps a log. Source inventories detect engine
changes during a step; R1 also freezes the inventory across resumed invocations.
No GPU reproduction was run while curating this chapter.

The `exact` step is CPU only (tens of seconds). It runs the CPU model tests
(`test_model.py`: the three tiny sets at atol = rtol = 0, the constructor
variants) and the opt-in probes (`test_cpu_probes.py`: 8-token probes of the
released checkpoint and of three full-size synthetic models, one 4096-token
tiny probe). The tests write `data/engine/wp3/cpu-pytest/*.json`; the step
refuses to run when its prerequisites are missing (tests would skip silently):
the tiny fixtures `data/engine/fixtures/cpu-tiny` and models
`data/engine/fixture-models/{tiny,full}` (from `tah.engine.generate_fixtures`,
profiles `tiny` and `full --prepare-only`) and `models/TaH-plus-1.7B`. It also
fails if a report was not rewritten by the run. The recorded `exact.json` was
curated from a run of exactly these two test files on the current engine
sources (35 passed, 27 s on CPU).

The speed step refreshes eager, graphs, overlap and the graph profile. It keeps
the explicitly labelled historical reference observation. The separate `reference`
step reruns the reference harness and memory monitor; its timing is the sum of
CSV batch-inference contributions rather than the old coarse stopwatch. It
updates the timing-scope label as well as the numbers. The reference uses a
coarse inference timer, retokenized output and sampled device memory, whereas
engine metrics use generation time, exact IDs and allocator peaks. Do not
substitute harness request latency sums for either throughput denominator.

`curate.py STEP --sources PATH [--root DIR]` can import existing experiment
outputs without running a model (`--root` resolves relative source paths). `sources.json` gives the current original output paths; an
alternative mapping can point at fresh experiments. It preserves aggregate
numerators/denominators, reference limits, source hashes and runtime inventories;
it does not recalibrate gates. `equivalence` verifies generated registration
hashes and time order against the original corpus manifest before discarding
large per-sequence detail. The analyzer audits every retained binding metric.
Ratios in the summary maximize rates over layouts and depths independently;
they are not ratios of pooled event totals. Zero over zero is displayed as exact.

The historical single-token/full-layout assertion can fail while each matched
reference gate passes. Its nonzero exit is retained in the log; collection still
requires the separate reference reports and audits them. The compile negative
step similarly retains expected failing gate reports. Missing reports or other
failures stop collection. No gate limits are changed to make a run pass.

`validation_cli.py` redirects fixture paths and loads original W-perf prompt
records through `sources.json`; `cached_validation.py` adapts the cached test's
output paths; `export_fixture_limits.py` verifies regenerated production
fixtures. `longctx.py` reads the original exploratory generations named in
`sources.json`. These raw files, models and datasets are prerequisites for GPU
reproduction, but not CPU analysis. `origins.json` records the adapted scripts'
source hashes. Full tensors/traces/text remain in experiment directories.

`exploratory` only reimports the historical per-sample CSVs: recovered reference
batches changed RNG streams and cannot be replayed exactly. `accuracy` instead
runs the pre-registered protocol, merges canonical results and copies only CSVs,
completion/configuration/protocol records and a frozen source inventory into
`data/r1/`. Full generated text remains at the recorded experiment paths. See
[R1.md](R1.md); inspect protocol deviations before treating results as confirmatory.

## Data files

`data/` holds only what the tables are computed from. `MANIFEST.json` records
every file with its schema, source path/hash/date, checkpoint hashes,
environment and available code provenance; complete raw outputs remain at those
source paths.

- `speed.json`: counts, seconds, byte peaks and profile milliseconds.
- `equivalence.json`: aggregate errors and counts, reference bounds, registration hashes and times.
- `diagnostics.json`: position-bucket errors and dimensionless sampler statistics.
- `robustness.json`: completion, cleanup and identity audits.
- `negative.json`: compile comparisons.
- `method.json`: declared policies and runtime estimates.
- `exact.json`: level-0 counts (records, positions, logit coordinates, decision,
  pass-count and top-1 mismatches, largest logit difference) of the CPU
  comparisons asserted at atol = rtol = 0, with the constructor-case summary.
- `port.json`: second-GPU validation (L40S) parsed from the operator's summary
  report with `curate.py port`; raw reports stay on that machine. Procedure:
  `dev-guide/engine.md`, "Validating on a new GPU or machine".
- `evaluations/{reference-exploratory,engine-seed42,engine-seed1234,engine-seed2024}/{math500,gpqa}/detailed_results.csv`:
  per-sample problem/repeat key, correctness, output length, answer extraction
  and pass summaries (times in seconds). The accuracy table and the exploratory
  estimate are computed from these.

## Refreshing performance

Rerun `speed`, or replace the experiment paths in `sources.json` and run
`python report/1-engine/scripts/curate.py speed` followed by `analyze.py`; this
replaces `speed.json` and its manifest entry. The historical reference
observation is retained explicitly; `reference` refreshes it using CSV
batch-inference timing and sampled device memory, with that scope labelled. New
runtime hashes must accompany new engine measurements.
