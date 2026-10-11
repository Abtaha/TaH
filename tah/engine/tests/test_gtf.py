"""Binding G-TF: every WP2 production TF record plus full-vocabulary replay.

Run only after WP2 GPU capture. cpu-tiny is explicitly rejected. Disk
coordinates alone do not establish the exhaustive full-vocabulary gate.
"""
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tah.engine.fixtures import FixtureSuite
from tah.engine.generate_fixtures import labels_for, load_reference
from tah.engine.model import EagerTaHModel
from tah.engine.parity import assert_gtf, compare_output, require_production
from tah.engine.reference_capture import capture


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def replay_layout(sequences, item, record):
    seq = item["input_ids"]
    companion = sequences[(sequences.index(item) + 4) % len(sequences)]["input_ids"][:113]
    layouts = {
        "single": ([seq], 0, "left", 0),
        "left-extra": ([seq], 0, "left", 7),
        "mixed-left": ([seq, companion], 0, "left", 0),
        "mixed-reordered": ([companion, seq], 1, "left", 11),
        "mixed-right": ([companion, seq], 1, "right", 7),
    }
    return layouts[record["variant"]]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("CUDA_VISIBLE_DEVICES") == "", reason="CUDA hidden: binding G-TF requires GPU production fixtures")
@pytest.mark.parametrize("name", ["real", "mlp3", "mlp1", "label3"])
def test_gtf_every_wp2_fixture_set(name):
    suite = FixtureSuite(os.environ.get("TAH_GTF_FIXTURES", "data/engine/fixtures/reference"))
    require_production(suite)
    metadata = suite.manifest["checkpoints"][name]
    root = Path(metadata["path"])
    for file, info in metadata["files"].items():
        assert sha256(root / file) == info["sha256"], f"checkpoint changed since WP2: {file}"
    for file, digest in suite.manifest["reference_source_sha256"].items():
        assert sha256(file) == digest, f"reference changed since WP2: {file}"
    eval_config = Path(suite.manifest["eval_config"]["path"])
    assert sha256(eval_config) == suite.manifest["eval_config"]["sha256"]
    override = yaml.safe_load(eval_config.read_text()) if name == "real" else None
    engine = EagerTaHModel.from_pretrained(root, override, device="cuda")
    oracle = load_reference(SimpleNamespace(device="cuda", eval_config=eval_config), name, root)
    sequences = json.loads((suite.root / "inputs.json").read_text())["sequences"]
    rows = []
    for item in sequences:
        records = [r for r in suite.records(kind="teacher_forced", checkpoint=name) if r["sequence_id"] == item["id"]]
        template = suite.load(next(r for r in records if r["variant"] == "single"))
        output = engine(template["input_ids"].to("cuda"),
                        iter_count_labels=template.get("iter_count_labels"),
                        hidden_positions=template["hidden_positions"])
        # Keep only the CPU engine result while replaying the oracle layouts.
        for key, value in vars(output).items():
            setattr(output, key, value.cpu())
        for record in records:
            assert sha256(suite.root / record["file"]) == record["sha256"]
            stored = suite.load(record)
            batch, target, side, extra = replay_layout(sequences, item, record)
            labels = [labels_for(tokens) for tokens in batch] if name == "label3" else None
            trace, full_logits = capture(oracle, batch, pad_id=oracle.config.eos_token_id,
                                        side=side, extra=extra, labels=labels,
                                        topk=suite.manifest["capture_settings"]["topk"],
                                        chunk_size=record["chunk_size"])
            # A replay on a different supported GPU may have its own noise.
            # Compare to both WP2's saved coordinates and full local replay;
            # neither path adds a tolerance floor or recalibrates thresholds.
            replay = trace[target]
            assert torch.equal(stored["input_ids"], replay["input_ids"])
            row = compare_output(stored, output, full_logits[target], metadata["threshold"])
            row.update(sequence_id=item["id"], variant=record["variant"])
            rows.append(row)
            del trace, full_logits, replay, stored
        del output, template
    noise = suite.manifest["self_noise"][name]["teacher_forced"]["summary"]
    stats = assert_gtf(rows, noise)
    destination = Path(os.environ.get("TAH_GTF_RESULTS", "data/engine/wp3/gtf"))
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{name}.json").write_text(json.dumps({
        "gate": "G-TF", "production_gates_eligible": True,
        "device": torch.cuda.get_device_name(), "summary": stats,
        "noise_limits": noise, "comparisons": rows,
    }, indent=2))
    del engine, oracle
    gc.collect()
    torch.cuda.empty_cache()
