from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import cayleypy_submit as submit  # noqa: E402


class ParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_json(self, name: str, value: object) -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        return path

    def test_loads_bare_and_batch_json(self) -> None:
        envelope = {"schema_version": 1, "client_submission_id": "x"}
        version, rows = submit.load_envelopes(self.write_json("one.json", envelope), None)
        self.assertEqual((version, rows), (1, [envelope]))
        version, rows = submit.load_envelopes(
            self.write_json("batch.json", {"schema_version": 1, "results": [envelope]}), None
        )
        self.assertEqual((version, rows), (1, [envelope]))

    def test_rejects_mixed_versions(self) -> None:
        path = self.root / "mixed.jsonl"
        path.write_text('{"schema_version":1}\n{"schema_version":2}\n', encoding="utf-8")
        with self.assertRaisesRegex(submit.ClientError, "INPUT_MIXED_SCHEMA"):
            submit.load_envelopes(path, None)

    def test_expands_csv_from_fill_once_config_and_replays(self) -> None:
        common = {
            "author": {"name": "Alice", "verification": "claimed"},
            "competition": "toy-cayley",
            "hardware": {"platform": "kaggle", "gpu_names": ["Tesla T4", "Tesla T4"], "accelerator_count": 2, "world_size": 2},
            "kaggle": {"owner": "alice", "slug": "solver", "version": 1, "notebook_sha256": "a" * 64, "run_url": "https://www.kaggle.com/code/alice/solver"},
            "model": {"filename": "model.pt", "format": "resmlp-layernorm", "sha256": "b" * 64, "manifest": {"dtype": "fp16", "output_dim": 2, "num_classes": 3, "state_len": 3}},
            "profile": {"requested_beam": 65536, "effective_beam": 65536, "alignment_delta": 0, "profile_power": 16, "world_size": 2, "selected_profile": "p16", "profile_evidence_version": 1, "evidence": "measured", "model_class": "output_move_count"},
            "runtime": {"solution_mode": "first", "max_depth": 8, "max_collected_solutions": 1, "touch_bfs_radius": 0, "b_micro": 2048, "shard_count": 4, "shard_capacity_scale_ppm": 1050000, "stream1_concurrency": 4, "stream3_ring_slots": 4, "stream4_batch_candidates": 98304, "stream4_trigger_candidates": 98304, "stream4_active_sort_slots": 4},
            "solver_commit": "c" * 40,
            "run_id": "run-one",
            "timings": {"solve_us": 10, "wall_us": 20},
        }
        config = {"schema_version": 1, "common": common, "puzzle_contexts": {"1": {"puzzle_type": "cycle-3", "initial_state": [2, 0, 1], "central_state": [0, 1, 2], "generators": {"clockwise": [1, 2, 0], "counterclockwise": [2, 0, 1]}}}}
        config_path = self.write_json("publisher-config.json", config)
        csv_path = self.root / "solutions.csv"
        csv_path.write_text(
            "puzzle_id,solution,final_orientation,search_mode,collection_index,collection_status,solved_depth,touch_depth,reflected_source_solution,searched_solution\n"
            "1,clockwise,original,off,0,first_solution,1,0,,\n",
            encoding="utf-8",
        )
        version, rows = submit.load_envelopes(csv_path, submit.load_config(config_path))
        self.assertEqual(version, 1)
        self.assertEqual(rows[0]["solution"]["path"], ["clockwise"])
        self.assertEqual(rows[0]["solution"]["length"], 1)
        self.assertEqual(rows[0]["proof"]["reached_state_sha256"], rows[0]["proof"]["central_state_sha256"])
        self.assertEqual(len(rows[0]["idempotency_key"]), 64)


class ArchiveTests(unittest.TestCase):
    def test_canonical_and_gzip_are_deterministic(self) -> None:
        value = {"z": "Δ", "a": [2, 1]}
        self.assertEqual(submit.canonical_bytes(value), b'{"a":[2,1],"z":"\xce\x94"}')
        self.assertEqual(submit.gzip_bytes(b"payload"), submit.gzip_bytes(b"payload"))

    def test_partition_preserves_order_and_limits(self) -> None:
        rows = [{"schema_version": 1, "puzzle_id": i, "padding": "x" * 40} for i in range(4)]
        parts = submit.partition_batches(1, rows, max_compressed=115, max_raw=180)
        recovered: list[int] = []
        for part in parts:
            self.assertLessEqual(len(part.compressed), 115)
            self.assertLessEqual(len(part.raw), 180)
            recovered.extend(item["puzzle_id"] for item in json.loads(part.raw)["results"])
        self.assertEqual(recovered, [0, 1, 2, 3])
        self.assertEqual([part.index for part in parts], list(range(len(parts))))
        self.assertTrue(all(part.count == len(parts) for part in parts))

    def test_single_oversized_envelope_fails(self) -> None:
        with self.assertRaisesRegex(submit.ClientError, "ENVELOPE_TOO_LARGE"):
            submit.partition_batches(1, [{"schema_version": 1, "padding": "x" * 200}], 60, 100)


if __name__ == "__main__":
    unittest.main()
