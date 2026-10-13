# Research report evidence

The [evaluation protocol](PROTOCOL.md) defines confirmatory workloads, seeds and
analysis. Chapters curate the results used in the research report; the
[chapter format](AGENTS.md) defines provenance and reproducibility requirements.

| Chapter | Contents | Status |
| --- | --- | --- |
| [Engine](1-engine/README.md) | Design, reference self-noise, mechanism checks, performance, robustness, exploratory accuracy and negative results | Curated summary; confirmatory R1 pending |

Regenerate the engine chapter on CPU:

```bash
python report/1-engine/analyze.py
```

GPU reproduction commands and runtime estimates are listed by
`bash report/1-engine/reproduce.sh`; printing the plan runs no experiments.
