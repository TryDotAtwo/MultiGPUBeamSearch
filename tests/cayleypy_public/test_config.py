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


def test_blank_endpoint_is_only_allowed_when_publishing_disabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("CAYLEYPY_RESULTS_INGEST_URL", raising=False)
    with pytest.raises(ValueError, match="RESULTS_INGEST_URL"):
        PublicRunConfig.from_mapping({**BASE, "results_ingest_url": ""})
    config = PublicRunConfig.from_mapping({**BASE, "publish_results": False, "results_ingest_url": ""})
    assert config.results_ingest_url == ""


def test_publish_endpoint_falls_back_to_namespaced_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "CAYLEYPY_RESULTS_INGEST_URL",
        "  https://results.example/ingest  ",
    )
    config = PublicRunConfig.from_mapping({**BASE, "results_ingest_url": ""})
    assert config.results_ingest_url == "https://results.example/ingest"


def test_explicit_publish_endpoint_takes_precedence_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CAYLEYPY_RESULTS_INGEST_URL",
        "https://environment.example/ingest",
    )
    config = PublicRunConfig.from_mapping(
        {**BASE, "results_ingest_url": "https://configured.example/ingest"}
    )
    assert config.results_ingest_url == "https://configured.example/ingest"


def test_environment_endpoint_does_not_enable_disabled_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "CAYLEYPY_RESULTS_INGEST_URL",
        "https://environment.example/ingest",
    )
    config = PublicRunConfig.from_mapping(
        {**BASE, "publish_results": False, "results_ingest_url": ""}
    )
    assert config.publish_results is False
    assert config.results_ingest_url == ""


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://results.example/ingest",
        "https://user:password@results.example/ingest",
        "https://results.example/ingest?token=secret",
        "https://results.example/ingest#fragment",
        "https://127.0.0.1/ingest",
        "https://[::1]/ingest",
        "https://169.254.169.254/ingest",
        "https://localhost/ingest",
        "https://worker.localhost/ingest",
        "not-a-url",
    ],
)
def test_publish_endpoint_must_be_safe_public_https(endpoint: str) -> None:
    with pytest.raises(ValueError, match="RESULTS_INGEST_URL"):
        PublicRunConfig.from_mapping({**BASE, "results_ingest_url": endpoint})


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://results.example/ingest",
        "https://8.8.8.8/ingest",
        "https://[2606:4700:4700::1111]/ingest",
    ],
)
def test_publish_endpoint_accepts_public_domain_and_global_ip(endpoint: str) -> None:
    config = PublicRunConfig.from_mapping({**BASE, "results_ingest_url": endpoint})
    assert config.results_ingest_url == endpoint


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
    ("field", "sentinel"),
    [
        ("author_name", "replace-with-author"),
        ("competition", " replace-with-competition "),
        ("kaggle_owner", "REPLACE_WITH_KAGGLE_OWNER"),
        ("kaggle_slug", "replace_with_kaggle_notebook_slug"),
        ("kaggle_username", "replace-with-kaggle-username"),
    ],
)
def test_publication_rejects_identity_and_provenance_placeholders(
    field: str,
    sentinel: str,
) -> None:
    with pytest.raises(ValueError, match=field.upper()):
        PublicRunConfig.from_mapping({**BASE, field: sentinel})


def test_nonpublishing_run_may_retain_author_placeholder() -> None:
    config = PublicRunConfig.from_mapping(
        {
            **BASE,
            "author_name": "replace-with-author",
            "publish_results": False,
            "results_ingest_url": "",
        }
    )
    assert config.author_name == "replace-with-author"


@pytest.mark.parametrize(
    "field,value",
    [("solver_commit", "A" * 40), ("kaggle_notebook_sha256", "x" * 64), ("kaggle_version", 0)],
)
def test_publication_provenance_formats_fail_closed(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field.upper()):
        PublicRunConfig.from_mapping({**BASE, field: value})
