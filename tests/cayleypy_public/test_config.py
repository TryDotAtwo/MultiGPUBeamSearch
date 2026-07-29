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
    "competition": "santa-2023", "kaggle_owner": "alice",
    "kaggle_slug": "public-cayley", "kaggle_version": 1,
    "solver_commit": "a" * 40,
    "kaggle_notebook_sha256": "b" * 64,
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

@pytest.mark.parametrize(
    "field",
    [
        "puzzle_id_start", "puzzle_id_end", "beam_width", "max_depth",
        "max_collected_solutions", "collect_until_depth", "touch_bfs_radius",
    ],
)
def test_numeric_config_fields_reject_bool(field):
    with pytest.raises(ValueError, match=field.upper()):
        PublicRunConfig.from_mapping({**BASE, field: True})


def test_publish_results_requires_real_bool():
    with pytest.raises(ValueError, match="PUBLISH_RESULTS"):
        PublicRunConfig.from_mapping({**BASE, "publish_results": "false"})

@pytest.mark.parametrize("field,value", [
    ("author_name", ""), ("author_name", 7), ("results_ingest_url", 7),
    ("competition", ""), ("kaggle_owner", ""), ("kaggle_slug", ""),
])
def test_public_identity_fields_are_strict(field, value):
    with pytest.raises(ValueError, match=field.upper()):
        PublicRunConfig.from_mapping({**BASE, field: value})


def test_blank_endpoint_is_only_allowed_when_publishing_disabled():
    with pytest.raises(ValueError, match="RESULTS_INGEST_URL"):
        PublicRunConfig.from_mapping({**BASE, "results_ingest_url": ""})
    config = PublicRunConfig.from_mapping({**BASE, "publish_results": False, "results_ingest_url": ""})
    assert config.results_ingest_url == ""

@pytest.mark.parametrize("field", ["model_source", "model_dtype", "checkpoint_format", "unknown_key"])
def test_public_config_rejects_hidden_or_unknown_controls(field: str) -> None:
    with pytest.raises(ValueError, match=field.upper() if field != "unknown_key" else "unknown config"):
        PublicRunConfig.from_mapping({**BASE, field: "unsafe"})


def test_publication_requires_complete_provenance() -> None:
    for field in (
        "competition", "kaggle_owner", "kaggle_slug", "kaggle_version",
        "solver_commit", "kaggle_notebook_sha256",
    ):
        values = dict(BASE)
        values.pop(field)
        with pytest.raises(ValueError, match="publication provenance"):
            PublicRunConfig.from_mapping(values)


def test_nonpublishing_run_may_omit_external_provenance() -> None:
    values = {
        key: value
        for key, value in BASE.items()
        if key not in {
            "competition", "kaggle_owner", "kaggle_slug", "kaggle_version",
            "solver_commit", "kaggle_notebook_sha256",
        }
    }
    values.update(publish_results=False, results_ingest_url="")
    config = PublicRunConfig.from_mapping(values)
    assert config.competition is None
    assert config.kaggle_notebook_sha256 is None


@pytest.mark.parametrize(
    "field,value",
    [("solver_commit", "A" * 40), ("kaggle_notebook_sha256", "x" * 64), ("kaggle_version", 0)],
)
def test_publication_provenance_formats_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field.upper()):
        PublicRunConfig.from_mapping({**BASE, field: value})
