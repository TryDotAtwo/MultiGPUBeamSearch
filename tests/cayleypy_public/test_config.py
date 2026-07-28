import pytest

from tools.cayleypy_public.config import PublicRunConfig


BASE = {
    "author_name": "alice", "checkpoint_path": "/kaggle/input/m/model.pth",
    "puzzle_info_json": "/kaggle/input/c/puzzle_info.json",
    "test_csv": "/kaggle/input/c/test.csv",
    "sample_submission_csv": "/kaggle/input/c/sample_submission.csv",
    "puzzle_id_start": 7, "puzzle_id_end": 9, "beam_width": 2**21,
    "max_depth": 100, "reflect_mode": "off", "reflect_source_csv": None,
    "solution_mode": "first", "collect_until_depth": 100,
    "max_collected_solutions": 1000, "touch_bfs_radius": 4,
    "publish_results": True, "results_ingest_url": "https://results.example/",
}


def test_config_accepts_exact_public_contract():
    cfg = PublicRunConfig.from_mapping(BASE)
    assert cfg.puzzle_ids == (7, 8, 9)
    assert cfg.model_dtype == "fp16"


def test_collect_depth_cannot_exceed_max_depth():
    values = {**BASE, "solution_mode": "collect", "collect_until_depth": 101}
    with pytest.raises(ValueError, match="COLLECT_UNTIL_DEPTH"):
        PublicRunConfig.from_mapping(values)


def test_only_requires_reflection_source():
    with pytest.raises(ValueError, match="REFLECT_SOURCE_CSV"):
        PublicRunConfig.from_mapping({**BASE, "reflect_mode": "only"})


@pytest.mark.parametrize("field,value", [("beam_width", 0), ("max_depth", 0), ("max_collected_solutions", 0)])
def test_positive_config_fields_reject_zero(field, value):
    with pytest.raises(ValueError, match=field.upper()):
        PublicRunConfig.from_mapping({**BASE, field: value})


def test_touch_radius_rejects_out_of_range_value():
    with pytest.raises(ValueError, match="TOUCH_BFS_RADIUS"):
        PublicRunConfig.from_mapping({**BASE, "touch_bfs_radius": 13})
