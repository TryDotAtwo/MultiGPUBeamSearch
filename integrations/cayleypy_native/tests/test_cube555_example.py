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


def test_finish_result_saves_without_publishing_when_disabled(tmp_path, graph, monkeypatch):
    def unexpected_publish(*args, **kwargs):
        raise AssertionError("publisher should not run")

    monkeypatch.setattr("multigpubeamsearch.publication.publish_result", unexpected_publish)
    result = SimpleNamespace(path_found=False, path=None, native_metadata={"devices": [0, 1]})
    report = cube555.finish_result(result, graph, [1, 0], publish=False, directory=tmp_path)
    assert json.loads(report["saved"].read_text())["path_found"] is False
    assert report["publication_status"] == "disabled"


def test_finish_result_publishes_and_saves_receipt(tmp_path, graph, monkeypatch):
    notebook = tmp_path / "notebook.ipynb"
    notebook.write_text("{}")
    result = SimpleNamespace(path_found=True, path=[0], native_metadata={"devices": [0, 1]})
    monkeypatch.setattr("multigpubeamsearch.publication.publish_result", lambda *a, **k: {
        "status": "published", "payload": "payload.json", "receipts": ["receipt.json"],
        "poll_manifest": "poll.json", "published": 1, "duplicate": 0,
        "rejected": 0, "unresolved": 0,
    })
    report = cube555.finish_result(
        result, graph, [1, 0], publish=True, directory=tmp_path, puzzle_id=1020,
        author="alice", kaggle_owner="alice", kaggle_slug="cube555", kaggle_version=7,
        notebook_path=notebook, solver_commit="deadbeef",
    )
    saved_status = json.loads((tmp_path / "publication_status.json").read_text())
    assert report["publication_status"] == "published"
    assert saved_status["kaggle_version"] == 7
    assert saved_status["solver_commit"] == "deadbeef"
