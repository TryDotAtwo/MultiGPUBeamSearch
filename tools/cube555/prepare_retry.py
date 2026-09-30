"""Offline-only retry preparation: verify originals and exclude D1/receipt keys."""
import argparse
from hashlib import sha256
import json
from pathlib import Path

from tools.cayleypy_public.results import (
    MAX_ARCHIVE_RESULTS, _validate_publish_envelope, build_result_archives,
    read_result_archive,
)


def accepted_rows(snapshot):
    """Read a successful Wrangler SELECT export, never infer absence from errors."""
    if not isinstance(snapshot, list) or not snapshot:
        raise ValueError("expected a successful D1 SELECT export")
    rows = []
    for statement in snapshot:
        if not isinstance(statement, dict) or statement.get("success") is not True:
            raise ValueError("D1 snapshot is incomplete or unsuccessful")
        items = statement.get("results")
        if not isinstance(items, list) or len(items) >= 10000:
            raise ValueError("D1 snapshot may be truncated")
        rows.extend(items)
    if any(not isinstance(row, dict) or not isinstance(row.get("idempotency_key"), str) for row in rows):
        raise ValueError("invalid D1 accepted row")
    return rows


def prepare(archive_path, snapshot_path, output, *, batch_results=100, receipts_path=None):
    if type(batch_results) is not int or not 1 <= batch_results <= MAX_ARCHIVE_RESULTS:
        raise ValueError("batch_results must be in [1, 2000]")
    archive = Path(archive_path).read_bytes()
    items = read_result_archive(archive)
    for item in items:
        _validate_publish_envelope(item)
    keys = [item["idempotency_key"] for item in items]
    if len(set(keys)) != len(keys):
        raise ValueError("source archive contains duplicate semantic keys")
    snapshot_bytes = Path(snapshot_path).read_bytes()
    rows = accepted_rows(json.loads(snapshot_bytes))
    accepted = {row["idempotency_key"] for row in rows}
    if receipts_path is not None:
        ledger = json.loads(Path(receipts_path).read_text(encoding="utf-8"))
        if ledger.get("schema_version") != 1 or not isinstance(ledger.get("archives"), dict):
            raise ValueError("invalid verified receipt ledger")
        for part in ledger["archives"].values():
            accepted.update(row["idempotency_key"] for row in part["receipts"])
    pending = [item for item in items if item["idempotency_key"] not in accepted]
    skipped = [item for item in items if item["idempotency_key"] in accepted]
    parts = []
    for offset in range(0, len(pending), batch_results):
        parts.extend(build_result_archives(pending[offset:offset + batch_results]))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "original.json.gz").write_bytes(archive)
    manifest = {
        "schema_version": 1, "mode": "offline_only", "network_requests": 0,
        "publication_authorized": False,
        "source_archive_sha256": sha256(archive).hexdigest(),
        "accepted_snapshot_sha256": sha256(snapshot_bytes).hexdigest(),
        "source_result_count": len(items), "pending_result_count": len(pending),
        "skipped_result_count": len(skipped), "batch_results": batch_results,
        "skipped_idempotency_keys": [item["idempotency_key"] for item in skipped],
        "archives": [],
        "before_send": "Refresh the scoped D1 SELECT and regenerate in a new directory; retain original envelope fields and match receipts by idempotency_key, not client_submission_id. Existing rejected/dead-letter/retryable rows require separate recovery, not a new submission.",
    }
    recovered_keys = []
    for index, part in enumerate(parts):
        records = read_result_archive(part)
        name = f"retry-{index:03d}.json.gz"
        (output / name).write_bytes(part)
        part_keys = [item["idempotency_key"] for item in records]
        recovered_keys.extend(part_keys)
        manifest["archives"].append({
            "file": name, "sha256": sha256(part).hexdigest(), "bytes": len(part),
            "result_count": len(records), "archive_index": index, "archive_count": len(parts),
            "idempotency_keys": part_keys,
            "client_submission_ids": [item["client_submission_id"] for item in records],
        })
    if recovered_keys != [item["idempotency_key"] for item in pending]:
        raise AssertionError("retry packaging lost or reordered results")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--accepted-snapshot", type=Path, required=True)
    parser.add_argument("--verified-receipts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-results", type=int, default=100)
    args = parser.parse_args()
    manifest = prepare(args.archive, args.accepted_snapshot, args.output,
        batch_results=args.batch_results, receipts_path=args.verified_receipts)
    print(json.dumps({key: manifest[key] for key in ["mode", "source_result_count", "pending_result_count", "skipped_result_count"]}))


if __name__ == "__main__":
    main()
