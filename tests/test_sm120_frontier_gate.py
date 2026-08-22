from __future__ import annotations

import json
import struct
from pathlib import Path

from tools.sm120_frontier_gate import compare_frontiers
from tools.sm120_quant_calibrate import generator_actions


def test_generator_actions_accepts_exactly_one_public_schema() -> None:
    assert generator_actions({"moves": [[0, 1], [1, 0]]}).tolist() == [[0, 1], [1, 0]]
    assert generator_actions({"actions": [[1, 0]]}).tolist() == [[1, 0]]
    for payload in ({}, {"moves": [[0]], "actions": [[0]]}):
        try:
            generator_actions(payload)
        except ValueError:
            pass
        else:
            raise AssertionError("ambiguous generator schema must fail")


def test_frontier_gate_compares_reconstructed_states_not_metadata(tmp_path: Path) -> None:
    generator = tmp_path / "generator.json"
    generator.write_text(json.dumps({"moves": [[1, 0], [0, 1]]}), encoding="utf-8")
    test_csv = tmp_path / "test.csv"
    test_csv.write_text("id,initial_state\n0,0;1\n", encoding="utf-8")
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    baseline.mkdir()
    candidate.mkdir()
    record = struct.Struct("<QII")
    # Different parent/route records can still reconstruct the same state.
    (baseline / "depth_0.candidate_meta.bin").write_bytes(record.pack(0, 0, 0))
    (candidate / "depth_0.candidate_meta.bin").write_bytes(record.pack(0, 0, 0))
    rows = compare_frontiers(baseline, candidate, generator, test_csv)
    assert rows[0]["jaccard"] == 1.0
    assert rows[0]["baseline_sha256"] == rows[0]["candidate_sha256"]
