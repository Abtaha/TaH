# Report chapters — format

`report/` holds the evidence for the research report (LaTeX on Overleaf). One numbered directory per major change or experiment: `1-engine/`, `2-baseline/`, `3-…`. `report/README.md` indexes the chapters. Everything is in English and committed.

Each chapter contains:
- `README.md` — written for the report reader: what was done and why, method, results as tables/figures, limitations. Ends with a **Data guide**: every file in `data/`, its schema/columns, units, how it was produced and how to interpret it.
- `data/` — the raw data the results are computed from: metric JSON, per-sample CSV, large text files compressed as `.jsonl.xz`. Target ≤ ~20 MB per chapter; never multi-GB tensors (regenerate them instead). `data/MANIFEST.json` records checkpoint SHA256, environment versions, git commit of the code, source path and SHA256 of every data file, seeds and dates.
- `analyze.py` — CPU only, seconds: reads only `data/` and regenerates every number, table and figure used in `README.md`, writing `tables/*.tex` (booktabs, `\input`-able in Overleaf) and `figures/*.pdf`. Deterministic: a second run is byte-identical. No number is typed by hand into `README.md` or the LaTeX source.
- `reproduce.sh` (or `.py`) — GPU, may take hours: regenerates `data/` from the checkpoint/fixtures with code in this repository (exact commands, seeds, runtime estimates). Scripts it needs live in the repository, not in ignored directories.

Describe results and methods, not the development process (no task/agent bookkeeping).
