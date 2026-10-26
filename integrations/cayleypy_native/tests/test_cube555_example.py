import json
from types import SimpleNamespace
import pytest
from multigpubeamsearch import cube555


@pytest.fixture
def graph(monkeypatch):
    contract = SimpleNamespace(generators=((1, 0),), center=(0, 1))
    monkeypatch.setattr(cube555.GraphContract, "from_graph", lambda *args: contract)
    return object()


def test_result_is_replayed_and_saved(tmp_path, graph):
    result = SimpleNamespace(path_found=True, path=[0], native_metadata={"beam": 100})
    path = cube555.save_result(result, graph, [1, 0], tmp_path)
    saved = json.loads(path.read_text())
    assert saved["path"] == [0] and saved["replay_valid"] is True


@pytest.mark.parametrize("path", [[], [1], None])
def test_invalid_solution_is_not_saved(tmp_path, graph, path):
    with pytest.raises(ValueError):
        cube555.save_result(SimpleNamespace(path_found=True, path=path), graph, [1, 0], tmp_path)
    assert not list(tmp_path.iterdir())


def test_unsolved_is_not_a_validated_solution(tmp_path, graph):
    result = SimpleNamespace(path_found=False, path=None, native_metadata={})
    saved = json.loads(cube555.save_result(result, graph, [1, 0], tmp_path).read_text())
    assert saved["replay_valid"] is None and saved["path_found"] is False


def test_bad_download_is_not_installed(tmp_path, monkeypatch):
    import io
    monkeypatch.setattr(cube555.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"broken"))
    with pytest.raises(ValueError, match="corrupted"):
        cube555._inputs(tmp_path)
    assert not (tmp_path / "cube555-inputs.zip").exists()
