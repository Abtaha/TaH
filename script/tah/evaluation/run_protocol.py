#!/usr/bin/env python3
"""Run, resume, and merge evaluation protocols without changing eval.py.

Protocol YAML files describe an eval recipe plus non-overlapping groups. This
tool writes one derived recipe and exact command record per sub-run, invokes
the existing eval.py CLI, and assembles completed samples into ordinary
per-dataset result directories.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "script/tah/evaluation/eval.py"
CSV_FIELDS = [
    "problem_id", "sample_idx", "correct_answer", "predicted_answer", "has_answer",
    "is_correct", "input_tokens", "output_tokens", "processing_time",
    "mean_passes_per_token", "fraction_tokens_at_least_2_passes", "pass_stat_tokens",
]
STATS_FIELDS = [
    "problem_id", "accuracy", "correct_count", "total_samples", "avg_output_length",
    "mean_passes_per_token", "fraction_tokens_at_least_2_passes", "pass_stat_tokens",
]


def now_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML must contain a mapping: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def yaml_data(value: Any) -> str:
    return yaml.safe_dump(value, sort_keys=False, allow_unicode=True)


def parse_range(value: Any, length: int) -> list[int]:
    if value is None or value == "all":
        return list(range(length))
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) != 2:
            raise ValueError(f"range must be 'start:end' or 'all', got {value!r}")
        start, end = (int(x) for x in parts)
    elif isinstance(value, int):
        start, end = 0, value
    elif isinstance(value, (list, tuple)) and len(value) in (1, 2):
        start, end = (0, int(value[0])) if len(value) == 1 else (int(value[0]), int(value[1]))
    else:
        raise ValueError(f"range must be 'start:end', [start,end], [end], or all; got {value!r}")
    if start < 0 or end <= start:
        raise ValueError(f"invalid range {start}:{end}")
    return list(range(min(start, length), min(end, length)))


def load_dataset_items(datasets: list[str]) -> dict[str, list[dict]]:
    # Use the same existing loader as eval.py so expected IDs follow its source
    # filters, fallback split, and ID normalization exactly.
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tah.evaluate.datasets import load_combined_dataset

    items, _ = load_combined_dataset(datasets)
    result = {name: [] for name in datasets}
    for item in items:
        result[item["_source_dataset"]].append(item)
    return result


def normalize_group(group: dict, all_items: dict[str, list[dict]]) -> dict:
    name = str(group.get("name", "")).strip()
    if not name or any(ch in name for ch in "/\\") or name in {".", ".."}:
        raise ValueError(f"group needs a simple path-safe name: {name!r}")
    datasets = group.get("datasets")
    if isinstance(datasets, str):
        datasets = [x.strip() for x in datasets.split(",") if x.strip()]
    if not isinstance(datasets, list) or not datasets or len(set(datasets)) != len(datasets):
        raise ValueError(f"group {name}: datasets must be a nonempty list of unique names")
    unknown = sorted(set(datasets) - set(all_items))
    if unknown:
        raise ValueError(f"group {name}: unknown datasets {unknown}")

    range_map = group.get("data_ranges", {})
    if not isinstance(range_map, dict):
        raise ValueError(f"group {name}: data_ranges must be a mapping")
    common_range = group.get("data_range")
    ids = group.get("data_ids")
    if ids is not None:
        if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
            raise ValueError(f"group {name}: data_ids must be a nonempty list of strings")
        if len(set(ids)) != len(ids):
            raise ValueError(f"group {name}: duplicate data_ids")

    selected: dict[str, list[str]] = {}
    selected_indices: dict[str, list[int]] = {}
    for dataset in datasets:
        rows = all_items[dataset]
        if ids is not None:
            wanted = set(ids)
            selected_indices[dataset] = [i for i, row in enumerate(rows) if str(row["id"]) in wanted]
        else:
            if dataset in range_map:
                selection = parse_range(range_map[dataset], len(rows))
            elif common_range is not None:
                selection = parse_range(common_range, len(rows))
            else:
                selection = list(range(len(rows)))
            selected_indices[dataset] = selection
        selected[dataset] = [str(rows[i]["id"]) for i in selected_indices[dataset]]
        if not selected[dataset]:
            raise ValueError(f"group {name}: selection for {dataset} is empty")
    if ids is not None:
        got = {pid for ids_for_ds in selected.values() for pid in ids_for_ds}
        missing = sorted(set(ids) - got)
        if missing:
            raise ValueError(f"group {name}: data_ids did not match loaded rows: {missing}")

    overrides = group.get("overrides", group.get("eval_config_overrides", {}))
    if not isinstance(overrides, dict):
        raise ValueError(f"group {name}: overrides must be a mapping")
    for key in ("repeat_size", "max_new_tokens", "batch_size"):
        if key in group:
            overrides = {**overrides, key: group[key]}
    repeat = overrides.get("repeat_size")
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 1:
        raise ValueError(f"group {name}: repeat_size must be a positive integer override")
    return {
        "name": name,
        "datasets": datasets,
        "data_ranges": range_map,
        "data_range": common_range,
        "data_ids": ids,
        "selected_ids": selected,
        "selected_indices": selected_indices,
        "overrides": overrides,
        "repeat_size": repeat,
    }


def load_protocol(protocol_path: Path) -> tuple[dict, dict, list[dict]]:
    protocol = load_yaml(protocol_path)
    eval_config_value = protocol.get("eval_config")
    if not isinstance(eval_config_value, str) or not eval_config_value.strip():
        raise ValueError("protocol YAML requires eval_config")
    base_path = Path(eval_config_value)
    if not base_path.is_absolute():
        base_path = (protocol_path.parent / base_path).resolve()
    base = load_yaml(base_path)
    groups = protocol.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ValueError("protocol YAML requires a nonempty groups list")
    all_names = []
    for group in groups:
        names = group.get("datasets", [])
        if isinstance(names, str):
            names = [x.strip() for x in names.split(",") if x.strip()]
        all_names.extend(names)
    if not all_names:
        raise ValueError("protocol has no datasets")
    all_items = load_dataset_items(list(dict.fromkeys(all_names)))
    normalized = [normalize_group(group, all_items) for group in groups]
    group_names = [group["name"] for group in normalized]
    if len(group_names) != len(set(group_names)):
        raise ValueError("protocol group names must be unique")

    seen: dict[str, set[str]] = {}
    for group in normalized:
        repeat = group["repeat_size"]
        for dataset, ids in group["selected_ids"].items():
            overlap = seen.setdefault(dataset, set()).intersection(ids)
            if overlap:
                raise ValueError(f"protocol groups overlap in {dataset}: {sorted(overlap)[:10]}")
            seen[dataset].update(ids)
            group["expected_keys"] = group.get("expected_keys", {})
            group["expected_keys"][dataset] = [[pid, i] for pid in ids for i in range(repeat)]
    protocol["_resolved_eval_config"] = str(base_path)
    protocol["_resolved_groups"] = normalized
    protocol["_expected_problem_ids"] = {k: sorted(v) for k, v in seen.items()}
    protocol["_dataset_sizes"] = {name: len(items) for name, items in all_items.items()}
    return protocol, base, normalized


def load_run_rows(run_dir: Path, remap_sample_idx: int | None = None) -> tuple[list[dict], list[dict]]:
    """Read eval output rows, preferring combined files over worker files."""
    leaves: list[tuple[Path, list[Path]]] = []
    seen_dirs: set[Path] = set()
    for status_file in run_dir.rglob("run_status.json"):
        leaf = status_file.parent
        combined = leaf / "detailed_results.csv"
        csvs = [combined] if combined.exists() else sorted(leaf.glob("job_*/detailed_results.csv"))
        if csvs:
            leaves.append((leaf, csvs))
            seen_dirs.add(leaf)
    if not leaves:
        # Legacy canonical run directories often have no run_status.json.
        csvs = [p for p in run_dir.rglob("detailed_results.csv") if "job_" not in p.parts]
        leaves = [(p.parent, [p]) for p in csvs]

    rows: list[dict] = []
    samples: list[dict] = []
    for leaf, csvs in leaves:
        for csv_path in csvs:
            with csv_path.open(newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    if not row.get("problem_id"):
                        continue
                    row["sample_idx"] = int(row["sample_idx"])
                    if remap_sample_idx is not None:
                        row["sample_idx"] = remap_sample_idx
                    rows.append(row)

        sample_files = [leaf / "samples.jsonl"]
        sample_files = [p for p in sample_files if p.exists()]
        if not sample_files:
            sample_files = sorted(leaf.glob("job_*/details/**/sample_*.json"))
        for sample_file in sample_files:
            if sample_file.suffix == ".jsonl":
                with sample_file.open(encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            item = json.loads(line)
                            if remap_sample_idx is not None:
                                item["sample"] = remap_sample_idx
                            samples.append(item)
            else:
                item = json.loads(sample_file.read_text(encoding="utf-8"))
                item["id"] = item.get("problem_id", sample_file.parent.name)
                try:
                    item["sample"] = int(sample_file.stem.split("_")[-1])
                except ValueError:
                    continue
                if remap_sample_idx is not None:
                    item["sample"] = remap_sample_idx
                samples.append(item)
    return rows, samples


def key_of(row: dict) -> tuple[str, int]:
    return str(row["problem_id"]), int(row["sample_idx"])


def ensure_unique_rows(rows: list[dict], where: str) -> dict[tuple[str, int], dict]:
    by_key: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = key_of(row)
        if key in by_key:
            raise ValueError(f"duplicate (problem, sample) {key} in {where}")
        by_key[key] = row
    return by_key


def write_attempt_command(attempt_dir: Path, argv: list[str], *, config_path: Path,
                          remap_sample_idx: int | None, selection: dict) -> None:
    command = {
        "argv": argv,
        "shell_command": shlex.join(argv),
        "cwd": str(REPO_ROOT),
        "derived_eval_config": str(config_path),
        "sample_index_remap": remap_sample_idx,
        "selection": selection,
        "started_at": now_utc(),
        "status": "started",
    }
    write_json(attempt_dir / "command.json", command)


def shell_display(argv: list[str]) -> str:
    return shlex.join(argv)


def main_run(args: argparse.Namespace) -> int:
    protocol_path = args.config.resolve()
    protocol, base_config, groups = load_protocol(protocol_path)
    backends = protocol.get("backends")
    if backends and args.backend not in backends:
        raise ValueError(f"backend {args.backend!r} is not allowed by this protocol; choose {backends}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    state_path = output_root / "protocol_state.json"
    requested = {
        "protocol_config": str(protocol_path),
        "protocol_config_sha256": sha256_file(protocol_path),
        "eval_config_sha256": sha256_file(Path(protocol["_resolved_eval_config"])),
        "protocol_name": protocol.get("name", protocol_path.stem),
        "model_path": str(Path(args.model_path).resolve()) if Path(args.model_path).exists() else args.model_path,
        "backend": args.backend,
        "seed": args.seed,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        for key, value in requested.items():
            if state.get(key) != value:
                raise ValueError(f"output root already belongs to a different protocol run: {key} differs")
    else:
        state = {
            **requested,
            "created_at": now_utc(),
            "resolved_eval_config": protocol["_resolved_eval_config"],
            "expected_problem_ids": protocol["_expected_problem_ids"],
            "dataset_sizes": protocol["_dataset_sizes"],
            "groups": {},
        }
    for group in groups:
        gp = state["groups"].setdefault(group["name"], {
            "status": "pending", "datasets": group["datasets"],
            "selected_ids": group["selected_ids"], "selected_indices": group["selected_indices"],
            "data_ids": group["data_ids"], "data_ranges": group["data_ranges"],
            "data_range": group["data_range"], "overrides": group["overrides"],
            "repeat_size": group["repeat_size"],
            "expected_keys": group["expected_keys"], "runs": [],
        })
        for field in ("datasets", "selected_ids", "selected_indices", "data_ids", "data_ranges",
                      "data_range", "overrides", "repeat_size", "expected_keys"):
            if gp.get(field) != group.get(field):
                raise ValueError(f"protocol group field {field} changed since this output root was created: {group['name']}")
    write_json(state_path, state)

    for group in groups:
        group_state = state["groups"][group["name"]]
        group_dir = output_root / "groups" / group["name"]
        group_runs_dir = group_dir / "runs"
        group_runs_dir.mkdir(parents=True, exist_ok=True)
        expected_keys = {
            (str(pid), int(sample))
            for dataset_keys in group_state["expected_keys"].values()
            for pid, sample in dataset_keys
        }
        already: dict[tuple[str, int], dict] = {}
        for run_record in group_state["runs"]:
            attempt = output_root / run_record["attempt_dir"]
            rows, _ = load_run_rows(attempt, run_record.get("sample_index_remap"))
            for key, row in ensure_unique_rows(rows, str(attempt)).items():
                if key in already:
                    raise ValueError(f"duplicate completed pair {key} across group sub-runs: {group['name']}")
                already[key] = row
        unexpected = set(already) - expected_keys
        if unexpected:
            raise ValueError(f"group {group['name']} produced unexpected sample pairs: {sorted(unexpected)[:10]}")
        missing = expected_keys - set(already)
        if not missing:
            group_state["status"] = "complete"
            group_state["completed_samples"] = len(already)
            write_json(state_path, state)
            print(f"SKIP {group['name']}: complete ({len(already)} samples)")
            continue

        # The first attempt preserves the protocol's repeat grouping. After an
        # interruption, launch one-repeat sub-runs partitioned by missing
        # sample index and map local sample 0 back to that missing index.
        todo: list[tuple[list[str] | None, int | None]] = []
        if not group_state["runs"]:
            todo.append((None, None))
        else:
            for sample_idx in sorted({sample for _, sample in missing}):
                ids = sorted(pid for pid, sample in missing if sample == sample_idx)
                todo.append((ids, sample_idx))

        for ids, remap in todo:
            attempt_number = len(group_state["runs"]) + 1
            suffix = "initial" if remap is None else f"resume-sample-{remap}"
            attempt_name = f"{attempt_number:04d}-{suffix}"
            attempt_dir = group_runs_dir / attempt_name
            attempt_dir.mkdir(parents=True, exist_ok=False)
            config = dict(base_config)
            config.update(group["overrides"])
            if remap is not None:
                config["repeat_size"] = 1
            config_path = attempt_dir / "eval_config.yaml"
            config_path.write_text(yaml_data(config), encoding="utf-8")
            eval_output = attempt_dir / "eval_output"
            argv = [
                sys.executable, str(EVAL_SCRIPT),
                "--eval_config", str(config_path),
                "--model_path", str(requested["model_path"]),
                "--dataset_name", ",".join(group["datasets"]),
                "--backend", args.backend,
                "--job_nums", str(args.job_nums),
                "--tp_size_per_job", str(args.tp_size_per_job),
                "--random_seed", str(args.seed),
                "--output_dir", str(eval_output),
                "--logger_level", args.logger_level,
            ]
            if ids is not None:
                argv.extend(["--data_ids", ",".join(ids)])
            elif group["data_ids"] is not None:
                argv.extend(["--data_ids", ",".join(group["data_ids"])])
            elif group["data_ranges"]:
                range_specs = []
                for dataset in group["datasets"]:
                    if dataset in group["data_ranges"]:
                        value = group["data_ranges"][dataset]
                        if value == "all" or value is None:
                            range_specs.append(f"{dataset}=all")
                        elif isinstance(value, str):
                            range_specs.append(f"{dataset}={value}")
                        elif isinstance(value, int):
                            range_specs.append(f"{dataset}=0:{value}")
                        elif len(value) == 1:
                            range_specs.append(f"{dataset}=0:{value[0]}")
                        else:
                            range_specs.append(f"{dataset}={value[0]}:{value[1]}")
                    elif group["data_range"] is not None:
                        value = group["data_range"]
                        if value == "all":
                            range_specs.append(f"{dataset}=all")
                        elif isinstance(value, str):
                            range_specs.append(f"{dataset}={value}")
                        elif isinstance(value, int):
                            range_specs.append(f"{dataset}=0:{value}")
                        elif len(value) == 1:
                            range_specs.append(f"{dataset}=0:{value[0]}")
                        else:
                            range_specs.append(f"{dataset}={value[0]}:{value[1]}")
                if range_specs:
                    argv.extend(["--dataset_ranges", ",".join(range_specs)])
            elif group["data_range"] is not None:
                value = group["data_range"]
                if isinstance(value, str):
                    if value != "all":
                        argv.extend(["--data_range", *value.split(":")])
                elif isinstance(value, int):
                    argv.extend(["--data_range", str(value)])
                else:
                    argv.extend(["--data_range", *[str(x) for x in value]])

            selection = {"datasets": group["datasets"], "data_ids": ids,
                         "sample_index_remap": remap}
            if ids is None:
                selection.update(data_ranges=group["data_ranges"], data_range=group["data_range"])
            write_attempt_command(attempt_dir, argv, config_path=config_path,
                                  remap_sample_idx=remap, selection=selection)
            record = {
                "attempt_dir": str(attempt_dir.relative_to(output_root)),
                "sample_index_remap": remap,
                "selection": selection,
                "expected_pairs": len(missing) if remap is None else sum(1 for _, s in missing if s == remap),
            }
            group_state["runs"].append(record)
            group_state["status"] = "running"
            write_json(state_path, state)
            print(f"RUN {group['name']}: {shell_display(argv)}", flush=True)
            if args.plan_only:
                print(f"  config: {config_path}")
                continue

            completed = subprocess.run(argv, cwd=REPO_ROOT, check=False)
            command_path = attempt_dir / "command.json"
            command_record = json.loads(command_path.read_text(encoding="utf-8"))
            command_record.update(exit_code=completed.returncode,
                                  finished_at=now_utc(),
                                  status="complete" if completed.returncode == 0 else "failed")
            write_json(command_path, command_record)
            record["exit_code"] = completed.returncode
            record["finished_at"] = command_record["finished_at"]
            write_json(state_path, state)
            if completed.returncode != 0:
                group_state["status"] = "interrupted"
                write_json(state_path, state)
                raise RuntimeError(f"eval.py failed for protocol group {group['name']} (exit {completed.returncode}); rerun the same command to resume")

            # Refresh before planning the next missing slot.
            rows, _ = load_run_rows(attempt_dir, remap)
            attempt_keys = set(ensure_unique_rows(rows, str(attempt_dir)))
            if not attempt_keys.issubset(expected_keys):
                raise ValueError(f"eval.py produced unexpected keys in group {group['name']}: {sorted(attempt_keys - expected_keys)[:10]}")
            for key in attempt_keys:
                if key in already:
                    raise ValueError(f"resume sub-run duplicated completed pair {key} in group {group['name']}")
                already[key] = next(row for row in rows if key_of(row) == key)
            missing = expected_keys - set(already)

        if args.plan_only:
            group_state["status"] = "planned"
            write_json(state_path, state)
            continue
        if missing:
            group_state["status"] = "interrupted"
            group_state["completed_samples"] = len(already)
            group_state["missing_samples"] = len(missing)
            write_json(state_path, state)
            raise RuntimeError(f"group {group['name']} returned successfully but is missing {len(missing)} planned sample pairs")
        group_state["status"] = "complete"
        group_state["completed_samples"] = len(already)
        group_state.pop("missing_samples", None)
        write_json(state_path, state)
        print(f"DONE {group['name']}: {len(already)} samples", flush=True)

    if not args.plan_only and all(s["status"] == "complete" for s in state["groups"].values()):
        state["status"] = "complete"
    else:
        state["status"] = "planned" if args.plan_only else "incomplete"
    write_json(state_path, state)
    return 0


def rows_to_stats(rows: list[dict]) -> list[dict]:
    by_problem: dict[str, list[dict]] = {}
    for row in rows:
        by_problem.setdefault(str(row["problem_id"]), []).append(row)
    result = []
    for pid, problem_rows in sorted(by_problem.items()):
        n = len(problem_rows)
        correct = sum(1 for row in problem_rows if str(row["is_correct"]).lower() in ("true", "1"))
        pass_tokens = sum(int(float(row.get("pass_stat_tokens") or 0)) for row in problem_rows)
        pass_mean = _weighted_row_field(problem_rows, "mean_passes_per_token")
        pass_two = _weighted_row_field(problem_rows, "fraction_tokens_at_least_2_passes")
        result.append({
            "problem_id": pid,
            "accuracy": f"{correct / n:.3f}",
            "correct_count": correct,
            "total_samples": n,
            "avg_output_length": f"{sum(int(float(row.get('output_tokens') or 0)) for row in problem_rows) / n:.2f}",
            "mean_passes_per_token": "" if pass_mean is None else f"{pass_mean:.6f}",
            "fraction_tokens_at_least_2_passes": "" if pass_two is None else f"{pass_two:.6f}",
            "pass_stat_tokens": pass_tokens,
        })
    return result


def _weighted_row_field(rows: list[dict], field: str) -> float | None:
    weighted = 0.0
    tokens = 0
    for row in rows:
        n = int(float(row.get("pass_stat_tokens") or 0))
        value = row.get(field)
        if n and value not in (None, ""):
            weighted += float(value) * n
            tokens += n
    return weighted / tokens if tokens else None


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main_merge(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    state_path = output_root / "protocol_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"protocol run state not found: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "complete":
        raise ValueError(f"protocol run is {state.get('status')!r}; finish/resume all groups before merge")
    protocol = load_yaml(Path(state["protocol_config"]))
    groups = [
        {"name": name, "datasets": value["datasets"], "repeat_size": value["repeat_size"],
         "selected_indices": value["selected_indices"]}
        for name, value in state["groups"].items()
    ]

    expected_by_dataset: dict[str, set[tuple[str, int]]] = {}
    rows_by_dataset: dict[str, list[dict]] = {}
    samples_by_dataset: dict[str, list[dict]] = {}
    code_eval_by_dataset: dict[str, dict[str, list[dict]]] = {}
    group_coverage: dict[str, list[dict]] = {}
    for group in groups:
        group_state = state["groups"][group["name"]]
        group_rows: list[dict] = []
        group_samples: list[dict] = []
        for run_record in group_state["runs"]:
            attempt = output_root / run_record["attempt_dir"]
            rows, samples = load_run_rows(attempt, run_record.get("sample_index_remap"))
            group_rows.extend(rows)
            group_samples.extend(samples)
        unique = ensure_unique_rows(group_rows, f"protocol group {group['name']}")
        expected = {
            (str(pid), int(sample))
            for dataset_keys in group_state["expected_keys"].values()
            for pid, sample in dataset_keys
        }
        if set(unique) != expected:
            raise ValueError(f"group {group['name']} coverage mismatch: expected {len(expected)}, got {len(unique)}; missing={sorted(expected-set(unique))[:10]}, extra={sorted(set(unique)-expected)[:10]}")
        for dataset, keys in group_state["expected_keys"].items():
            expected_by_dataset.setdefault(dataset, set())
            duplicate = expected_by_dataset[dataset].intersection((pid, int(s)) for pid, s in keys)
            if duplicate:
                raise ValueError(f"protocol groups overlap in canonical dataset {dataset}: {sorted(duplicate)[:10]}")
            expected_by_dataset[dataset].update((pid, int(s)) for pid, s in keys)
        for row in group_rows:
            dataset = next((name for name in group["datasets"] if str(row["problem_id"]).startswith(name + "_")), None)
            if dataset is None:
                raise ValueError(f"could not map problem ID to protocol dataset: {row['problem_id']}")
            rows_by_dataset.setdefault(dataset, []).append(row)
        for sample in group_samples:
            pid = str(sample.get("problem_id", sample.get("id", "")))
            dataset = next((name for name in group["datasets"] if pid.startswith(name + "_")), None)
            if dataset is not None:
                samples_by_dataset.setdefault(dataset, []).append(sample)
        for run_record in group_state["runs"]:
            attempt = output_root / run_record["attempt_dir"]
            for eval_path in attempt.rglob("code_solutions.eval_results.json"):
                dataset = next((name for name in group["datasets"]
                                if any(part.startswith(name + "_") for part in eval_path.parts)), None)
                if dataset is None:
                    continue
                data = json.loads(eval_path.read_text(encoding="utf-8"))
                combined_eval = code_eval_by_dataset.setdefault(dataset, {})
                for task_id, completions in data.get("eval", {}).items():
                    combined_eval.setdefault(str(task_id), []).extend(completions)
        group_coverage[group["name"]] = [{"dataset": ds, "samples": len(keys)} for ds, keys in group_state["expected_keys"].items()]

    timestamp = state.get("created_at", now_utc()).replace(":", "").replace("-", "")
    timestamp = timestamp.replace("+00:00", "Z")
    canonical_root = output_root / "canonical"
    canonical_root.mkdir(parents=True, exist_ok=True)
    for dataset, expected in expected_by_dataset.items():
        rows = rows_by_dataset.get(dataset, [])
        actual = ensure_unique_rows(rows, f"canonical {dataset}")
        if set(actual) != expected:
            raise ValueError(f"canonical {dataset} is incomplete: expected {len(expected)}, got {len(actual)}")
        dataset_leaf = canonical_root / f"{dataset}_{state['backend']}" / timestamp
        indices = sorted({index for group in groups if dataset in group["datasets"]
                          for index in group["selected_indices"][dataset]})
        dataset_size = int(state["dataset_sizes"][dataset])
        if indices and indices != list(range(dataset_size)):
            if indices == list(range(indices[0], indices[-1] + 1)):
                dataset_leaf = dataset_leaf / f"TASK_{indices[0]}_{indices[-1] + 1}"
        dataset_dir = dataset_leaf
        dataset_dir.mkdir(parents=True, exist_ok=True)
        _write_csv(dataset_dir / "detailed_results.csv", CSV_FIELDS,
                   sorted(rows, key=lambda r: (r["problem_id"], int(r["sample_idx"]))))
        stats = rows_to_stats(rows)
        with (dataset_dir / "evaluation_stats.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STATS_FIELDS)
            writer.writeheader()
            writer.writerows(stats)
            total_n = len(rows)
            total_c = sum(1 for row in rows if str(row["is_correct"]).lower() in ("true", "1"))
            pass_tokens = sum(int(float(row.get("pass_stat_tokens") or 0)) for row in rows)
            writer.writerow({})
            writer.writerow({
                "problem_id": "Total Accuracy", "accuracy": f"{total_c / total_n:.3f}",
                "correct_count": total_c, "total_samples": total_n,
                "avg_output_length": f"{sum(int(float(row.get('output_tokens') or 0)) for row in rows) / total_n:.2f}",
                "mean_passes_per_token": "" if _weighted_row_field(rows, "mean_passes_per_token") is None else f"{_weighted_row_field(rows, 'mean_passes_per_token'):.6f}",
                "fraction_tokens_at_least_2_passes": "" if _weighted_row_field(rows, "fraction_tokens_at_least_2_passes") is None else f"{_weighted_row_field(rows, 'fraction_tokens_at_least_2_passes'):.6f}",
                "pass_stat_tokens": pass_tokens,
            })
        sample_rows = samples_by_dataset.get(dataset, [])
        sample_map: dict[tuple[str, int], dict] = {}
        for sample in sample_rows:
            pid = str(sample.get("problem_id", sample.get("id", "")))
            idx = int(sample.get("sample", sample.get("sample_idx", 0)))
            key = (pid, idx)
            if key in sample_map:
                raise ValueError(f"duplicate raw sample object {key} while merging {dataset}")
            sample_map[key] = sample
        if set(sample_map) != expected:
            raise ValueError(f"raw samples coverage mismatch for {dataset}: expected {len(expected)}, got {len(sample_map)}")
        with (dataset_dir / "samples.jsonl").open("w", encoding="utf-8") as f:
            for key in sorted(sample_map):
                f.write(json.dumps(sample_map[key], ensure_ascii=False) + "\n")
        if dataset in {"humaneval", "mbpp"}:
            eval_rows = code_eval_by_dataset.get(dataset)
            if eval_rows is None:
                raise ValueError(f"missing EvalPlus results while merging {dataset}")
            if set(eval_rows) != {pid[len(dataset) + 1:] for pid, _ in expected}:
                raise ValueError(f"EvalPlus task coverage mismatch while merging {dataset}")
            expected_by_task: dict[str, int] = {}
            for pid, sample_idx in expected:
                task_id = pid[len(dataset) + 1:]
                expected_by_task[task_id] = expected_by_task.get(task_id, 0) + 1
            for task_id, expected_count in expected_by_task.items():
                if len(eval_rows[task_id]) != expected_count:
                    raise ValueError(f"EvalPlus sample count mismatch for {dataset}/{task_id}: expected {expected_count}, got {len(eval_rows[task_id])}")
            for task_results in eval_rows.values():
                task_results.sort(key=lambda row: int(row.get("completion_id", 0)))

            def pass_at_one(kind: str) -> float:
                task_scores = []
                for task_results in eval_rows.values():
                    if not task_results:
                        raise ValueError(f"empty EvalPlus completions while merging {dataset}")
                    if kind == "base":
                        passed = [str(row.get("base_status", "")).lower() == "pass"
                                  for row in task_results]
                    else:
                        passed = [str(row.get("base_status", "")).lower() == "pass"
                                  and (row.get("plus_status") is None
                                       or str(row.get("plus_status", "")).lower() == "pass")
                                  for row in task_results]
                    task_scores.append(sum(passed) / len(task_results))
                return sum(task_scores) / len(task_scores)

            sidecar = {
                "eval": eval_rows,
                "pass_at_k": {
                    "base": {"pass@1": pass_at_one("base")},
                    "plus": {"pass@1": pass_at_one("plus")},
                },
            }
            write_json(dataset_dir / "code_solutions.eval_results.json", sidecar)
            with (dataset_dir / "code_solutions.jsonl").open("w", encoding="utf-8") as f:
                for key in sorted(sample_map):
                    sample = sample_map[key]
                    pid, _ = key
                    task_id = pid[len(dataset) + 1:]
                    f.write(json.dumps({"task_id": task_id, "solution": sample.get("output", "")},
                                       ensure_ascii=False) + "\n")
        write_json(dataset_dir / "run_status.json", {
            "status": "complete", "protocol": state["protocol_name"],
            "model_path": state["model_path"], "backend": state["backend"],
            "random_seed": state["seed"], "dataset": dataset,
            "expected_samples": len(expected), "completed_samples": len(rows),
            "repeat_size": next((g["repeat_size"] for g in groups if dataset in g["datasets"]), None),
            "protocol_config_sha256": state["protocol_config_sha256"],
            "eval_config_sha256": state["eval_config_sha256"],
            "protocol_source": state["protocol_config"], "group_coverage": group_coverage,
            "canonical_merge_at": now_utc(),
        })
        source_eval = Path(state["resolved_eval_config"])
        if source_eval.exists():
            shutil.copy2(source_eval, dataset_dir / source_eval.name)
        (dataset_dir / "effective_config.yaml").write_text(yaml_data({
            "protocol": state["protocol_name"], "backend": state["backend"],
            "seed": state["seed"], "source_eval_config": state["resolved_eval_config"],
            "groups": [g["name"] for g in groups if dataset in g["datasets"]],
        }), encoding="utf-8")
        print(f"MERGED {dataset}: {len(expected)} samples → {dataset_dir}")

    write_json(canonical_root / "protocol_manifest.json", {
        "protocol": state["protocol_name"], "protocol_config": state["protocol_config"],
        "protocol_config_sha256": state["protocol_config_sha256"],
        "eval_config_sha256": state["eval_config_sha256"],
        "model_path": state["model_path"], "backend": state["backend"], "seed": state["seed"],
        "expected_problem_ids": state["expected_problem_ids"],
        "expected_samples": {dataset: len(keys) for dataset, keys in expected_by_dataset.items()},
        "created_at": state["created_at"], "merged_at": now_utc(),
    })
    state["canonical_root"] = str(canonical_root)
    state["merged_at"] = now_utc()
    write_json(state_path, state)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run or resume the configured protocol groups")
    run.add_argument("--config", type=Path, required=True, help="protocol YAML")
    run.add_argument("--model_path", required=True, help="model checkpoint path or Hub ID")
    run.add_argument("--backend", choices=("tah", "tah_engine", "hf", "sglang"), required=True)
    run.add_argument("--seed", type=int, required=True, help="seed passed unchanged to each eval sub-run")
    run.add_argument("--output_root", type=Path, required=True)
    run.add_argument("--job_nums", type=int, default=1)
    run.add_argument("--tp_size_per_job", type=int, default=1)
    run.add_argument("--logger_level", default="WARNING")
    run.add_argument("--plan-only", action="store_true", help="write configs/commands without launching eval.py")
    run.set_defaults(func=main_run)
    merge = sub.add_parser("merge", help="verify all groups and assemble canonical dataset runs")
    merge.add_argument("--output_root", type=Path, required=True)
    merge.set_defaults(func=main_merge)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(f"run_protocol.py: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
