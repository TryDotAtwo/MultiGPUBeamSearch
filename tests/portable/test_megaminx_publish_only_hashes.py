from pathlib import Path


def test_publish_only_rechecks_archive_payload_hashes_before_http():
    text=Path("portable/megaminx_cluster/run.sh").read_text()
    assert "verify_archive_payloads" in text
    assert text.index("verify_archive_payloads") < text.index("validate_and_publish")
