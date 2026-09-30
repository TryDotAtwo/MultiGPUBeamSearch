"""Offline retry identity and bounded receipt regressions; no live network."""
from copy import deepcopy
import gzip
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
import uuid

import pytest
from tools.cayleypy_public import results as publisher
from tools.cube555.prepare_retry import prepare


def envelopes(count=3):
    golden = json.loads((Path(__file__).resolve().parents[2] / "configs/cayleypy_results_v1_golden.json").read_text(encoding="utf-8"))["cases"][0]["envelope"]
    items = []
    for index in range(count):
        item = deepcopy(golden)
        item["solution"]["collection_index"] = index
        item["client_submission_id"] = f"018f7a24-8f6b-7000-8000-{index + 1:012x}"
        item["idempotency_key"] = publisher._hash_json(publisher._semantic_payload(item))
        items.append(item)
    return items


class Response:
    def __init__(self, value, status=202):
        self.status = status
        self.value = value
    def read(self, limit):
        return self.value[:limit]
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


@pytest.fixture
def local_transport(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(publisher, "_endpoint_resolves_only_to_public_addresses", lambda url: True)
    return tmp_path


def send(archive):
    return publisher.publish_result_archive("https://ingest.example/v1/results", archive,
        result_count=len(publisher.read_result_archive(archive)), archive_index=0, archive_count=1)


def test_lost_response_then_retry_retains_exact_bytes_and_one_acceptance_per_key(local_transport, monkeypatch):
    items = envelopes()
    archive = publisher.build_result_archives(items)[0]
    accepted, calls = {}, []
    def transport(request, timeout):
        calls.append(request.data)
        for item in publisher.read_result_archive(request.data):
            accepted.setdefault(item["idempotency_key"], str(uuid.uuid4()))
        if len(calls) == 1:
            raise TimeoutError("response lost after durable acceptance")
        return Response(json.dumps({"receipts": [{"submission_id": value, "idempotency_key": key} for key, value in accepted.items()]}).encode())
    monkeypatch.setattr(publisher, "urlopen", transport)
    assert send(archive).retryable
    assert send(archive).ok
    assert calls == [archive, archive]
    assert len(accepted) == len(items)
    ledger = json.loads((local_transport / "publish_receipts.json").read_text())
    assert len(ledger["archives"][sha256(archive).hexdigest()]["receipts"]) == len(items)


def test_partial_202_is_not_success_and_verified_receipts_can_filter_retry(local_transport, monkeypatch):
    items = envelopes()
    archive = publisher.build_result_archives(items)[0]
    accepted = {"idempotency_key": items[0]["idempotency_key"], "submission_id": str(uuid.uuid4())}
    monkeypatch.setattr(publisher, "urlopen", lambda request, timeout: Response(json.dumps({"receipts": [accepted], "errors": [{"index": 1, "code": "submission_persist_failed"}]}).encode()))
    status = send(archive)
    assert not status.ok and status.retryable and status.status_code == 202
    source = local_transport / "source.gz"; source.write_bytes(archive)
    snapshot = local_transport / "d1.json"; snapshot.write_text('[{"success":true,"results":[]}]')
    result = prepare(source, snapshot, local_transport / "retry", receipts_path=local_transport / "publish_receipts.json")
    assert result["skipped_result_count"] == 1 and result["pending_result_count"] == 2


@pytest.mark.parametrize("body", [b"", b"not JSON", b'{"receipts":[]}', b'{"receipts":[],"private":"secret"}', b"x" * (publisher.MAX_RECEIPT_BYTES + 1)], ids=["empty", "invalid_json", "empty_receipts", "private_field", "oversized"])
def test_unverified_success_never_closes_acceptance(local_transport, monkeypatch, body):
    archive = publisher.build_result_archives(envelopes(1))[0]
    monkeypatch.setattr(publisher, "urlopen", lambda request, timeout: Response(body))
    result = send(archive)
    assert not result.ok and result.retryable
    assert "secret" not in result.safe_error


@pytest.mark.parametrize("body,detail", [(b'{"error":"rate_limit_unavailable","token":"secret"}', ' (rate_limit_unavailable)'), (b'{"error":"private-secret"}', ''), (b"x" * 3000, '')], ids=["allowlisted", "unknown_code", "oversized"])
def test_503_records_only_allowlisted_server_code(local_transport, monkeypatch, body, detail):
    archive = publisher.build_result_archives(envelopes(1))[0]
    def fail(request, timeout):
        raise HTTPError(request.full_url, 503, "secret", {}, BytesIO(body))
    monkeypatch.setattr(publisher, "urlopen", fail)
    result = send(archive)
    assert result.retryable and result.safe_error == "results endpoint returned HTTP 503" + detail


def test_count_mismatch_is_blocked_before_network(local_transport, monkeypatch):
    archive = publisher.build_result_archives(envelopes(1))[0]
    monkeypatch.setattr(publisher, "urlopen", lambda *args, **kwargs: pytest.fail("invalid archive sent"))
    result = publisher.publish_result_archive("https://ingest.example/v1/results", archive, result_count=2, archive_index=0, archive_count=1)
    assert not result.ok and not result.retryable


def test_prepare_preserves_all_identities_and_excludes_existing_keys(tmp_path):
    items = envelopes(7)
    archive = publisher.build_result_archives(items)[0]
    source = tmp_path / "original.gz"; source.write_bytes(archive)
    snapshot = tmp_path / "accepted.json"
    snapshot.write_text(json.dumps([{"success": True, "results": [{"idempotency_key": items[0]["idempotency_key"], "submission_id": str(uuid.uuid4()), "state": "staged"}, {"idempotency_key": items[1]["idempotency_key"], "state": "retryable"}]}]))
    a = prepare(source, snapshot, tmp_path / "a", batch_results=2)
    b = prepare(source, snapshot, tmp_path / "b", batch_results=2)
    assert a == b
    restored = [item for part in a["archives"] for item in publisher.read_result_archive((tmp_path / "a" / part["file"]).read_bytes())]
    assert restored == items[2:]
    assert source.read_bytes() == archive == (tmp_path / "a/original.json.gz").read_bytes()
    assert a["pending_result_count"] + a["skipped_result_count"] == len(items)
    assert a["publication_authorized"] is False and a["network_requests"] == 0
    assert [part["result_count"] for part in a["archives"]] == [2, 2, 1]


@pytest.mark.parametrize("snapshot", [[], [{"success": False, "results": []}], [{"success": True}], [{"success": True, "results": [{"idempotency_key": "a"}] * 10000}]])
def test_failed_or_truncated_status_snapshot_cannot_authorize_retry(tmp_path, snapshot):
    source = tmp_path / "source.gz"; source.write_bytes(publisher.build_result_archives(envelopes(1))[0])
    accepted = tmp_path / "snapshot.json"; accepted.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError):
        prepare(source, accepted, tmp_path / "retry")
    assert not (tmp_path / "retry").exists()


@pytest.mark.parametrize("kind", ["unexpected_key", "changed_receipt_id"])
def test_bad_receipts_never_replace_verified_identity(local_transport, monkeypatch, kind):
    item = envelopes(1)[0]
    archive = publisher.build_result_archives([item])[0]
    receipt = {"idempotency_key": item["idempotency_key"], "submission_id": str(uuid.uuid4())}
    monkeypatch.setattr(publisher, "urlopen", lambda request, timeout: Response(json.dumps({"receipts": [receipt]}).encode()))
    assert send(archive).ok
    saved = (local_transport / "publish_receipts.json").read_bytes()
    bad = {**receipt, "idempotency_key": "f" * 64} if kind == "unexpected_key" else {**receipt, "submission_id": str(uuid.uuid4())}
    monkeypatch.setattr(publisher, "urlopen", lambda request, timeout: Response(json.dumps({"receipts": [bad]}).encode()))
    result = send(archive)
    assert not result.ok and result.retryable
    assert (local_transport / "publish_receipts.json").read_bytes() == saved
