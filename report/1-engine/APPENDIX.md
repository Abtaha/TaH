# Per-layout numerical equivalence

Each cell is the engine-versus-reference rate divided by the reference-versus-reference rate on the same sequences (1.0 = the engine differs from the reference as much as the reference differs from itself under a different execution layout; limit 1.5). Decision flips compare continue/stop decisions (maximum over depths); pass-count mismatch compares final passes per token; logit RMSE is the root mean squared logit difference; top-1 mismatch compares the highest-logit token. Exact means both counts are zero. Layouts: full sequence, cached with 1 or 256 tokens per step, and generated sequences.

| Checkpoint | Layout | Decision flips | Pass-count mismatch | Logit RMSE | Top-1 mismatch | Result |
| --- | --- | --- | --- | --- | --- | --- |
| real | full | 1.15 | 1.15 | 1.08 | 1.19 | PASS |
| real | cached-1 | 1.03 | 1.03 | 1.02 | 1.17 | PASS |
| real | cached-256 | 1.06 | 1.06 | 1.03 | 1.14 | PASS |
| real | generated | 1.01 | 1.01 | 1.03 | 1.06 | PASS |
| mlp3 | full | 0.97 | 0.95 | 0.99 | 0.96 | PASS |
| mlp3 | cached-1 | 1.01 | 0.99 | 0.99 | 0.92 | PASS |
| mlp3 | cached-256 | 1.01 | 1.00 | 1.01 | 1.01 | PASS |
| mlp3 | generated | 1.23 | 0.87 | 1.02 | 1.05 | PASS |
| mlp1 | full | exact | exact | 1.04 | 1.11 | PASS |
| mlp1 | cached-1 | exact | exact | 1.02 | 1.13 | PASS |
| mlp1 | cached-256 | exact | exact | 1.00 | 1.20 | PASS |
| mlp1 | generated | exact | exact | 0.97 | 0.92 | PASS |
| label3 | full | exact | exact | 1.04 | 1.06 | PASS |
| label3 | cached-1 | exact | exact | 1.02 | 1.06 | PASS |
| label3 | cached-256 | exact | exact | 1.00 | 1.07 | PASS |
| label3 | generated | exact | exact | 0.97 | 0.92 | PASS |

See [the chapter](README.md#numerical-equivalence).
