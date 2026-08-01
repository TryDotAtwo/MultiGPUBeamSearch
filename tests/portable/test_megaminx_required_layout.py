from tools.build_megaminx_native_release import REQUIRED_PATHS
from pathlib import Path


def test_release_requires_and_stages_readme_and_preflight_wrapper():
    assert "README.md" in REQUIRED_PATHS
    assert "scripts/preflight.sh" in REQUIRED_PATHS
    workflow = Path(".github/workflows/megaminx-native-release.yml").read_text()
    assert 'README.md "$stage/README.md"' in workflow
    assert 'scripts/preflight.sh "$stage/scripts/"' in workflow
