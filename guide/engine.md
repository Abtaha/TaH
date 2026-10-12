# TaH inference engine

The new TaH v1 inference engine in `tah/engine/` is **under development**.
Environment/kernel checks and model, checkpoint, fixture, and sampling
components exist, but a complete generation engine and evaluation backend
are not available yet. The evaluation CLI currently accepts only `tah`, `hf`,
and `sglang`.

It is intended to provide fast single-GPU inference for TaH v1 checkpoints,
with selective per-token passes, bounded chunked prefill, continuous batching,
managed KV memory, CUDA graph decode, and sampling with per-request seeds.
Planned results include generated tokens/text, prompt and generation pass
counts, finish reasons, and timing. API details and runnable generation
examples will be documented here when implemented; no generation API is
promised by this placeholder.

Use [installation](installation.md) for the existing `engine` extra and
environment/kernel checks, and [evaluation](evaluation.md) for the shipped
reference `tah` backend. TaH2's bundled `tah2/minisgl/` is a separate engine;
see the main [README](../README.md#tah2-usage) for its usage.
