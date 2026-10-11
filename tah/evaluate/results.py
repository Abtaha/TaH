"""Collect evaluation runs into a compact model-level CSV table."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


_BACKENDS = ("tah_engine", "sglang", "tah", "hf")
_TIMESTAMP = re.compile(r"^\d{8}_\d{6}$")
_FIELDS = (
    "model", "run", "dataset", "backend", "metric", "accuracy_or_pass_at_1",
    "correct_count", "total_samples", "problem_count", "avg_output_length",
    "mean_passes_per_token", "fraction_tokens_at_least_2_passes",
)


def _parse_dataset_backend(folder: str) -> tuple[str, str]:
    for backend in _BACKENDS:
        suffix = "_" + backend
        if folder.endswith(suffix):
            return folder[:-len(suffix)], backend
    return folder, ""


def _stats_total(path: Path) -> Dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    total = next((r for r in rows if r.get("problem_id") == "Total Accuracy"), {})
    return {
        "accuracy": total.get("accuracy", ""),
        "correct_count": total.get("correct_count", ""),
        "total_samples": total.get("total_samples", ""),
        "avg_output_length": total.get("avg_output_length", ""),
        "mean_passes_per_token": total.get("mean_passes_per_token", ""),
        "fraction_tokens_at_least_2_passes": total.get("fraction_tokens_at_least_2_passes", ""),
    }


def _code_metrics(run_dir: Path) -> Optional[Dict[str, object]]:
    result_file = run_dir / "code_solutions.eval_results.json"
    if not result_file.is_file():
        return None
    with open(result_file, encoding="utf-8") as f:
        result = json.load(f)
    eval_rows = result.get("eval", {})
    base_pass = result.get("pass_at_k", {}).get("base", {}).get("pass@1")
    if base_pass is None:
        return None
    try:
        from evalplus.eval import PASS
    except ImportError:
        PASS = "pass"
    total_samples = sum(len(task_rows) for task_rows in eval_rows.values())
    correct_count = sum(
        1 for task_rows in eval_rows.values() for sample in task_rows
        if sample.get("base_status") == PASS
    )
    return {
        "metric": "pass@1 (base)",
        "accuracy_or_pass_at_1": base_pass,
        "correct_count": correct_count,
        "total_samples": total_samples,
        "problem_count": len(eval_rows),
    }


def collect_results(results_dir: str | Path, output_file: Optional[str | Path] = None) -> List[dict]:
    """Find all evaluation stats below one model's results root and write a CSV."""
    root = Path(results_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"results directory does not exist: {root}")
    output = Path(output_file).expanduser().resolve() if output_file else root / "results_summary.csv"
    model = root.parent.name or root.name
    rows: List[dict] = []
    for stats_file in sorted(root.rglob("evaluation_stats.csv")):
        relative = stats_file.relative_to(root)
        parts = relative.parts
        if any(part.startswith("job_") for part in parts):
            continue
        if len(parts) < 3:
            continue
        dataset, backend = _parse_dataset_backend(parts[0])
        run_dir = stats_file.parent
        run_parts = list(parts[1:-1])
        timestamp_idx = next((i for i, part in enumerate(run_parts) if _TIMESTAMP.match(part)), None)
        run_id = "/".join(run_parts[timestamp_idx:]) if timestamp_idx is not None else "/".join(run_parts)
        stats = _stats_total(stats_file)
        code = _code_metrics(run_dir)
        if code is not None:
            metric = code["metric"]
            accuracy = code["accuracy_or_pass_at_1"]
            correct_count = code["correct_count"]
            total_samples = code["total_samples"]
            problem_count = code["problem_count"]
        else:
            metric = "accuracy"
            accuracy = stats["accuracy"]
            correct_count = stats["correct_count"]
            total_samples = stats["total_samples"]
            # Per-problem rows are the non-empty rows before the totals section.
            problem_count = sum(1 for row in _read_stats_rows(stats_file) if row.get("problem_id") not in (None, "", "Total Accuracy"))
        rows.append({
            "model": model,
            "run": run_id,
            "dataset": dataset,
            "backend": backend,
            "metric": metric,
            "accuracy_or_pass_at_1": accuracy,
            "correct_count": correct_count,
            "total_samples": total_samples,
            "problem_count": problem_count,
            "avg_output_length": stats["avg_output_length"],
            "mean_passes_per_token": stats["mean_passes_per_token"],
            "fraction_tokens_at_least_2_passes": stats["fraction_tokens_at_least_2_passes"],
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _read_stats_rows(path: Path) -> List[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect TaH evaluation results across datasets and runs")
    parser.add_argument("--results_dir", required=True, help="Model eval_results directory")
    parser.add_argument("--output_csv", default=None, help="Output CSV (default: <results_dir>/results_summary.csv)")
    args = parser.parse_args()
    rows = collect_results(args.results_dir, args.output_csv)
    print(f"Wrote {len(rows)} run/dataset rows to {args.output_csv or Path(args.results_dir) / 'results_summary.csv'}")


if __name__ == "__main__":
    main()
