"""G-TF assertions and trace conversion, used by pytest/tooling only."""
from __future__ import annotations

import torch

from .reference_capture import compare, summarize


def output_trace(output, template):
    trace = {key: getattr(output, key).detach().cpu() for key in (
        "visited", "decision", "decision_logits", "continue_probability", "decider_hidden")}
    trace["pass_count"] = output.iter_count.cpu()
    # Coordinate systems come from the oracle; top-k errors are measured at
    # reference indices, not by comparing independently sorted lists.
    for key in ("final_topk_values", "final_topk_indices", "logit_positions", "final_logits_sample"):
        trace[key] = template[key]
    return trace


def compare_output(template, output, reference_logits, threshold):
    return compare(template, output_trace(output, template), reference_logits,
                   output.logits.cpu(), threshold)


def require_production(suite):
    assert suite.manifest.get("production_gates_eligible") is True, "G-TF requires production fixtures; cpu-tiny is development only"
    assert suite.manifest.get("profile") == "full"
    assert set(suite.manifest["checkpoints"]) == {"real", "mlp3", "mlp1", "label3"}
    assert suite.manifest["self_noise"]["mlp3"]["full_fixture_depth_coverage_pass"] is True
    for name in suite.manifest["checkpoints"]:
        records = list(suite.records(kind="teacher_forced", checkpoint=name))
        assert len(records) == 45, f"missing WP2 TF layouts for {name}"
        assert len({r["sequence_id"] for r in records}) == 9


def assert_gtf(rows, noise):
    stats = summarize(rows)
    for depth in stats["per_depth"]:
        assert depth["decision_agreement"] >= .995, depth
        assert depth["decision_mismatches"] == depth["mismatches_within_0.01"], depth
    for metric in ("logit_mae", "logit_rmse", "logit_max_abs"):
        assert stats[metric] <= noise[f"g_tf_{metric}_limit"], (metric, stats[metric], noise)
    disagreement = stats["top1_mismatches"] / max(stats["top1_compared"], 1)
    assert disagreement <= noise["g_tf_top1_disagreement_limit"], (disagreement, noise)
    for coordinates in ("retained_topk_error", "retained_dense_error"):
        for metric in ("logit_mae", "logit_rmse", "logit_max_abs"):
            assert stats[coordinates][metric] <= noise[coordinates][f"g_tf_{metric}_limit"], (coordinates, metric, stats, noise)
    return stats
