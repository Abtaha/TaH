# Per-layout numerical equivalence

Each disagreement ratio divides the engine-versus-reference disagreement rate by the reference-versus-reference disagreement rate. The denominator compares two reference executions of the same sequences under different execution layouts. The full-logit RMSE ratio divides engine-versus-reference RMSE by reference-versus-reference RMSE. Continue/stop compares decisions at jointly visited depths; final pass counts compare complete token positions. Full-logit RMSE is the root mean squared logit difference; highest-logit disagreement compares the token with the largest score. Each continue/stop cell maximizes over depths; exact means both counts are zero.

| Checkpoint | Layout | Continue/stop disagreement ratio | Final pass-count disagreement ratio | Full-logit RMSE ratio | Highest-logit token disagreement ratio | Gate result |
| --- | --- | --- | --- | --- | --- | --- |
| real | full | 1.147 | 1.147 | 1.076 | 1.192 | PASS |
| real | cached-1 | 1.026 | 1.026 | 1.023 | 1.170 | PASS |
| real | cached-256 | 1.057 | 1.057 | 1.028 | 1.142 | PASS |
| real | generated | 1.014 | 1.014 | 1.025 | 1.058 | PASS |
| mlp3 | full | 0.965 | 0.954 | 0.992 | 0.956 | PASS |
| mlp3 | cached-1 | 1.012 | 0.987 | 0.993 | 0.923 | PASS |
| mlp3 | cached-256 | 1.009 | 1.005 | 1.011 | 1.007 | PASS |
| mlp3 | generated | 1.232 | 0.872 | 1.024 | 1.046 | PASS |
| mlp1 | full | exact | exact | 1.036 | 1.112 | PASS |
| mlp1 | cached-1 | exact | exact | 1.022 | 1.129 | PASS |
| mlp1 | cached-256 | exact | exact | 1.005 | 1.195 | PASS |
| mlp1 | generated | exact | exact | 0.967 | 0.925 | PASS |
| label3 | full | exact | exact | 1.039 | 1.061 | PASS |
| label3 | cached-1 | exact | exact | 1.025 | 1.064 | PASS |
| label3 | cached-256 | exact | exact | 0.997 | 1.073 | PASS |
| label3 | generated | exact | exact | 0.967 | 0.925 | PASS |

Ratios and gate scope are defined in [the chapter](README.md#numerical-equivalence).
