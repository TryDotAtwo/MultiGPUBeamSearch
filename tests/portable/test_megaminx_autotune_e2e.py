from pathlib import Path

from tools.megaminx_archive_contract import require_allowed_path


def test_fixed_archive_contract_contains_exact_autotune_payloads():
    for relative in (
        "autotune.sh",
        "scripts/autotune_job.sh",
        "portable/megaminx_cluster/autotune/__init__.py",
        "portable/megaminx_cluster/autotune/contracts.py",
        "portable/megaminx_cluster/autotune/search_space.py",
        "portable/megaminx_cluster/autotune/probe.py",
        "portable/megaminx_cluster/autotune/evidence.py",
        "portable/megaminx_cluster/autotune/controller.py",
        "portable/megaminx_cluster/autotune/calibration.json",
    ):
        require_allowed_path(relative)


def test_release_workflow_stages_and_marks_autotune_entrypoints_executable():
    text = Path(".github/workflows/megaminx-native-release.yml").read_text(encoding="utf-8")
    assert '"$stage/portable/megaminx_cluster/autotune"' in text
    assert 'portable/megaminx_cluster/autotune.sh' in text
    assert 'portable/megaminx_cluster/scripts/autotune_job.sh' in text
    assert 'portable/megaminx_cluster/autotune/*.py' in text
    assert 'portable/megaminx_cluster/autotune/calibration.json' in text
    assert 'chmod +x "$stage/autotune.sh" "$stage/scripts/autotune_job.sh"' in text


def test_archive_builder_requires_autotune_payload():
    text = Path("tools/build_megaminx_native_release.py").read_text(encoding="utf-8")
    assert '"autotune.sh"' in text
    assert '"scripts/autotune_job.sh"' in text
    assert '"portable/megaminx_cluster/autotune/calibration.json"' in text


def test_readme_has_pasteable_8xa100_command_and_resume_contract():
    text = Path("portable/megaminx_cluster/README.md").read_text(encoding="utf-8")
    assert "./autotune.sh --gpus 0,1,2,3,4,5,6,7" in text
    assert "--min-beam 30000000" in text
    assert "--time-budget 6h" in text
    assert "--bfs-hash-budget-mib 256" in text
    assert "resume" in text.lower()
    assert "registry.fragment.json" in text
