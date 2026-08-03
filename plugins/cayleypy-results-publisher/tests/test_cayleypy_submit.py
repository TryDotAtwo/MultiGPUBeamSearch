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


    def test_rejects_secret_like_fields_before_network(self) -> None:
        path = self.write_json("unsafe.json", {"schema_version": 1, "author": {"api_token": "must-not-send"}})
        with self.assertRaisesRegex(submit.ClientError, "INPUT_SECRET_FIELD"):
            submit.load_envelopes(path, None)

    def test_tsv_uses_the_same_exact_contract(self) -> None:
        source = ROOT / "templates" / "solutions.csv"
        target = self.root / "solutions.tsv"
        target.write_text(source.read_text(encoding="utf-8").replace(",", "\t"), encoding="utf-8")
        config = submit.load_config(ROOT / "templates" / "publisher-config-kaggle.json")
        version, rows = submit.load_envelopes(target, config)
        self.assertEqual((version, len(rows)), (1, 1))

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


class FakeTransport:
    def __init__(self, responses: list[submit.HttpResponse]) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, str, bytes | None, dict[str, str]]] = []

    def request(self, method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> submit.HttpResponse:
        self.requests.append((method, url, body, dict(headers or {})))
        if not self.responses:
            raise AssertionError("unexpected request")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class TransportTests(unittest.TestCase):
    def envelope(self, version: int = 1) -> dict[str, object]:
        return {
            "schema_version": version,
            "client_submission_id": "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e7f",
            "submitted_at": "2026-07-29T09:30:00.000Z",
            "competition": "toy-cayley",
            "puzzle_type": "cycle-3",
            "puzzle_id": 1,
        }

    def test_submit_uses_pinned_version_route_and_saves_safe_manifest(self) -> None:
        envelope = self.envelope()
        envelope["private_marker"] = "DO_NOT_LEAK"
        parts = submit.partition_batches(1, [envelope])
        receipt = {
            "submission_id": envelope["client_submission_id"],
            "idempotency_key": "d" * 64,
            "status_url": submit.OFFICIAL_ENDPOINT_BASE + "/v1/submissions/" + str(envelope["client_submission_id"]),
        }
        transport = FakeTransport([submit.HttpResponse(202, {}, json.dumps({"receipts": [receipt]}).encode())])
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "receipts.json"
            manifest = submit.submit_parts(parts, submit.SubmitConfig(manifest_path=manifest_path), transport)
            saved = manifest_path.read_text(encoding="utf-8")
        method, url, body, headers = transport.requests[0]
        self.assertEqual((method, url), ("POST", submit.OFFICIAL_ENDPOINT_BASE + "/v1/results"))
        self.assertEqual(headers["Content-Type"], "application/gzip")
        self.assertEqual(headers["User-Agent"], "cayleypy-results-publisher/0.1")
        self.assertEqual(headers["X-CayleyPy-Archive-Index"], "0")
        self.assertEqual(manifest["receipts"][0]["submission_id"], receipt["submission_id"])
        self.assertNotIn("DO_NOT_LEAK", saved)
        self.assertIsNotNone(body)

    def test_partial_server_errors_are_persisted_and_fail(self) -> None:
        parts = submit.partition_batches(1, [self.envelope()])
        response = {"receipts": [], "errors": [{"index": 0, "code": "schema"}]}
        transport = FakeTransport([submit.HttpResponse(202, {}, json.dumps(response).encode())])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipts.json"
            with self.assertRaisesRegex(submit.ClientError, "SUBMIT_PARTIAL"):
                submit.submit_parts(parts, submit.SubmitConfig(manifest_path=path), transport)
            self.assertTrue(path.is_file())

    def test_poll_accepts_staged_or_github_presence_after_status_cleanup(self) -> None:
        submission_id = str(self.envelope()["client_submission_id"])
        manifest = {
            "manifest_version": 1,
            "endpoint_origin": submit.OFFICIAL_ENDPOINT_BASE,
            "receipts": [{
                "submission_id": submission_id,
                "idempotency_key": "d" * 64,
                "status_url": submit.OFFICIAL_ENDPOINT_BASE + "/v1/submissions/" + submission_id,
                "github_url": "https://raw.githubusercontent.com/TryDotAtwo/cayleypy-beam-results/refs/heads/ingest/staging/results/v1/x.json",
                "state": "accepted",
            }],
            "parts": [],
        }
        staged = FakeTransport([submit.HttpResponse(200, {}, json.dumps({"state": "staged"}).encode())])
        summary = submit.poll_manifest(manifest, submit.PollConfig(timeout_seconds=0), staged)
        self.assertEqual(summary.published, 1)
        manifest["receipts"][0]["state"] = "accepted"
        cleaned = FakeTransport([submit.HttpResponse(404, {}, b"{}"), submit.HttpResponse(200, {}, b"{}")])
        summary = submit.poll_manifest(manifest, submit.PollConfig(timeout_seconds=0), cleaned)
        self.assertEqual(summary.published, 1)

    def test_poll_counts_duplicate_as_success(self) -> None:
        submission_id = str(self.envelope()["client_submission_id"])
        manifest = {"manifest_version": 1, "endpoint_origin": submit.OFFICIAL_ENDPOINT_BASE, "parts": [], "receipts": [{"submission_id": submission_id, "idempotency_key": "d" * 64, "status_url": submit.OFFICIAL_ENDPOINT_BASE + "/v1/submissions/" + submission_id, "github_url": "https://example.invalid", "state": "accepted"}]}
        transport = FakeTransport([submit.HttpResponse(200, {}, json.dumps({"state": "duplicate"}).encode())])
        summary = submit.poll_manifest(manifest, submit.PollConfig(timeout_seconds=0), transport)
        self.assertEqual((summary.published, summary.duplicate, summary.rejected, summary.unresolved), (0, 1, 0, 0))
    def test_poll_rejects_cross_origin_status_url(self) -> None:
        manifest = {"manifest_version": 1, "endpoint_origin": submit.OFFICIAL_ENDPOINT_BASE, "parts": [], "receipts": [{"submission_id": "x", "idempotency_key": "d" * 64, "status_url": "https://evil.example/v1/submissions/x", "github_url": "https://example.invalid", "state": "accepted"}]}
        with self.assertRaisesRegex(submit.ClientError, "STATUS_URL_UNSAFE"):
            submit.poll_manifest(manifest, submit.PollConfig(timeout_seconds=0), FakeTransport([]))


class TemplateAndResilienceTests(unittest.TestCase):
    def test_init_copies_valid_kaggle_and_native_templates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for source, version in (("kaggle", 1), ("native", 2)):
                config_path = root / f"{source}.json"
                submit.init_config(source, config_path)
                config = submit.load_config(config_path)
                self.assertEqual(config.schema_version, version)
                report = submit.preflight(ROOT / "templates" / "solutions.csv", config_path)
                self.assertEqual(report.schema_version, version)
                self.assertEqual(report.envelope_count, 1)

    def test_full_json_templates_preflight_without_config(self) -> None:
        for name, version in (("kaggle-v1.json", 1), ("native-v2.json", 2)):
            report = submit.preflight(ROOT / "templates" / name, None)
            self.assertEqual(report.schema_version, version)
            self.assertEqual(report.envelope_count, 1)

    def test_retry_uses_same_body_and_then_accepts(self) -> None:
        envelope = TransportTests().envelope()
        parts = submit.partition_batches(1, [envelope])
        receipt = {"submission_id": envelope["client_submission_id"], "idempotency_key": "d" * 64, "status_url": submit.OFFICIAL_ENDPOINT_BASE + "/v1/submissions/" + str(envelope["client_submission_id"])}
        transport = FakeTransport([submit.HttpResponse(429, {"Retry-After": "0"}, b"{}"), submit.HttpResponse(202, {}, json.dumps({"receipts": [receipt]}).encode())])
        with tempfile.TemporaryDirectory() as tmp:
            submit.submit_parts(parts, submit.SubmitConfig(Path(tmp) / "receipts.json", retry_base_seconds=0), transport)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(transport.requests[0][2], transport.requests[1][2])

    def test_timeout_is_retried_idempotently(self) -> None:
        envelope = TransportTests().envelope()
        parts = submit.partition_batches(1, [envelope])
        receipt = {"submission_id": envelope["client_submission_id"], "idempotency_key": "d" * 64, "status_url": submit.OFFICIAL_ENDPOINT_BASE + "/v1/submissions/" + str(envelope["client_submission_id"])}
        transport = FakeTransport([submit.ClientError("HTTP_TIMEOUT"), submit.HttpResponse(202, {}, json.dumps({"receipts": [receipt]}).encode())])
        with tempfile.TemporaryDirectory() as tmp:
            submit.submit_parts(parts, submit.SubmitConfig(Path(tmp) / "receipts.json", max_retries=1, retry_base_seconds=0), transport)
        self.assertEqual(len(transport.requests), 2)
        self.assertEqual(transport.requests[0][2], transport.requests[1][2])
    def test_partial_receipt_maps_to_non_error_envelope(self) -> None:
        first = TransportTests().envelope()
        second = copy.deepcopy(first)
        second["puzzle_id"] = 2
        second["client_submission_id"] = "018f7a24-8f6b-7c8e-9d1b-2a3b4c5d6e80"
        parts = submit.partition_batches(1, [first, second])
        receipt = {"submission_id": second["client_submission_id"], "idempotency_key": "e" * 64, "status_url": submit.OFFICIAL_ENDPOINT_BASE + "/v1/submissions/" + str(second["client_submission_id"])}
        transport = FakeTransport([submit.HttpResponse(202, {}, json.dumps({"receipts": [receipt], "errors": [{"index": 0, "code": "schema"}]}).encode())])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipts.json"
            with self.assertRaisesRegex(submit.ClientError, "SUBMIT_PARTIAL"):
                submit.submit_parts(parts, submit.SubmitConfig(path), transport)
            manifest = submit.load_manifest(path)
        self.assertIn("/2/", manifest["receipts"][0]["github_url"])
if __name__ == "__main__":
    unittest.main()
