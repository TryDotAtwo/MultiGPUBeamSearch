from pathlib import Path

from portable.megaminx_cluster.submit import default_archive_root


def test_installed_module_resolves_archive_root_above_portable_package():
    module = Path("release/portable/megaminx_cluster/submit.py")
    assert default_archive_root(module) == module.resolve().parents[2]
