from pathlib import Path


def test_workflow_stages_python_scripts_under_importable_package_path():
    text = Path(".github/workflows/megaminx-native-release.yml").read_text()
    assert '"$stage/portable/megaminx_cluster/scripts"' in text
    assert 'portable/megaminx_cluster/scripts/*.py "$stage/portable/megaminx_cluster/scripts/"' in text
