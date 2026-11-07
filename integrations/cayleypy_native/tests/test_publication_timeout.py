import json
from pathlib import Path
from types import SimpleNamespace

from multigpubeamsearch import publication


def _install_fake_publisher(tmp_path, monkeypatch, final_state="published"):
    payload = tmp_path / "publication.json"
    payload.write_text(
        json.dumps({"results": [{"solution": str(index)} for index in range(101)]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(publication, "prepare_publication", lambda *args, **kwargs: payload)
    monkeypatch.setattr(
        publication,
        "client_module",
        lambda _cache: SimpleNamespace(
            __file__=str(Path(publication.__file__)),
            canonical_bytes=lambda value: json.dumps(value).encode(),
        ),
    )
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        phase = command[2]
        manifest_path = Path(command[command.index("--manifest") + 1])
        if phase == "submit":
            part = Path(command[3])
            results = json.loads(part.read_text(encoding="utf-8"))["results"]
            part_number = len([item for item in calls if item[2] == "submit"]) - 1
            first_index = part_number * 100
            manifest_path.write_text(json.dumps({
                "manifest_version": 1,
                "endpoint_origin": "https://example.workers.dev",
                "input_sha256": f"part-{part_number}",
                "parts": [{"index": 0, "sha256": f"part-{part_number}", "state": "accepted"}],
                "receipts": [{
                    "submission_id": f"submission-{first_index + index}",
                    "status_url": f"https://example.workers.dev/v1/submissions/{first_index + index}",
                    "github_url": f"https://raw.githubusercontent.com/TryDotAtwo/cayleypy-beam-results/refs/heads/ingest/staging/{first_index + index}.json",
                    "state": "accepted",
                } for index, _ in enumerate(results)],
            }), encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="accepted", stderr="")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for receipt in manifest["receipts"]:
            receipt["state"] = final_state
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return SimpleNamespace(returncode=0 if final_state == "published" else 4,
                               stdout=final_state, stderr="")

    monkeypatch.setattr(publication.subprocess, "run", run)
    return payload, calls


def test_publisher_submits_all_parts_before_polling_and_persists_terminal_receipts(
    tmp_path, monkeypatch
):
    payload, calls = _install_fake_publisher(tmp_path, monkeypatch)

    status = publication.publish_result(None, None, None)

    assert status["status"] == "published"
    assert [command[2] for command in calls] == ["submit", "submit", "poll"]
    assert all("--wait" not in command for command in calls[:2])
    assert len(json.loads(Path(status["poll_manifest"]).read_text())["receipts"]) == 101
    for receipt_path in status["receipts"]:
        manifest = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        assert all(receipt["state"] == "published" for receipt in manifest["receipts"])


def test_publisher_does_not_report_accepted_receipts_as_published(tmp_path, monkeypatch):
    _payload, calls = _install_fake_publisher(tmp_path, monkeypatch, final_state="accepted")

    status = publication.publish_result(None, None, None)

    assert status["status"] == "pending_or_failed"
    assert calls[-1][2] == "poll"
