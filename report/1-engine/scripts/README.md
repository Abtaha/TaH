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

The speed step refreshes eager, graphs, overlap and the graph profile. It keeps
the explicitly labelled historical reference observation. The separate `reference`
step reruns the reference harness and memory monitor; its timing is the sum of
CSV batch-inference contributions rather than the old coarse stopwatch. It
updates the timing-scope label as well as the numbers. The reference uses a
coarse inference timer, retokenized output and sampled device memory, whereas
engine metrics use generation time, exact IDs and allocator peaks. Do not
substitute harness request latency sums for either throughput denominator.

`curate.py STEP --sources PATH` can import existing experiment outputs without
running a model. `sources.json` gives the current original output paths; an
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
