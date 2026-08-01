from pathlib import Path


def test_run_sh_exposes_publish_only_without_submitting_or_solving():
    text = Path("portable/megaminx_cluster/run.sh").read_text()
    assert '"${1:-}" = "--publish-only"' in text
    assert 'portable.megaminx_cluster.scripts.validate_and_publish' in text
    assert '--run-dir "$2"' in text
