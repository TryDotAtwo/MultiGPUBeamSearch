"""Resolve public CayleyPy competition/model assets inside a Molab sandbox."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile


def _kagglehub():
    try:
        import kagglehub
    except ImportError:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "kagglehub"],
            check=True,
        )
        import kagglehub
    return kagglehub


def _unpack_if_needed(source: Path, destination: Path) -> Path:
    if source.is_dir():
        return source
    if not source.is_file() or not zipfile.is_zipfile(source):
        raise ValueError(f"asset is neither a directory nor a ZIP archive: {source}")
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".unpacked"
    if not marker.exists():
        with zipfile.ZipFile(source) as archive:
            archive.extractall(destination)
        marker.write_text(source.name, encoding="utf-8")
    return destination


def _exact_glob(root: Path, pattern: str, label: str) -> Path:
    matches = sorted(path for path in root.rglob(pattern) if path.is_file())
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {label} matching {pattern!r}; observed={matches!r}")
    return matches[0]


def _optional_exact_glob(root: Path, pattern: str | None, label: str) -> Path | None:
    return None if pattern is None else _exact_glob(root, pattern, label)


def _public_archive_url(kind: str, reference: str) -> str:
    routes = {
        "kaggle_competition": "https://www.kaggle.com/api/v1/competitions/data/download-all/",
        "kaggle_dataset": "https://www.kaggle.com/api/v1/datasets/download/",
        "kaggle_model": "https://www.kaggle.com/api/v1/models/",
        "kaggle_notebook_output": "https://www.kaggle.com/api/v1/kernels/output/",
    }
    suffix = "/download" if kind == "kaggle_model" else ""
    return routes[kind] + reference + suffix


def _download_public_archive(kind: str, reference: str, cache_root: Path) -> Path:
    safe_name = sha256(f"{kind}:{reference}".encode("utf-8")).hexdigest()[:16]
    archive_path = cache_root / f"{safe_name}.zip"
    if not archive_path.exists():
        request = Request(
            _public_archive_url(kind, reference),
            headers={"User-Agent": "cayleypy-molab/1.0"},
        )
        try:
            with urlopen(request, timeout=120) as response, archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        except (HTTPError, URLError) as error:
            archive_path.unlink(missing_ok=True)
            code = getattr(error, "code", "network_error")
            raise RuntimeError(
                "SETUP_REQUIRED: Kaggle rejected the public asset download "
                f"({kind} {reference!r}, status={code}). If this competition requires "
                "rules acceptance, add Kaggle credentials to the Molab Secrets panel."
            ) from error
    return _unpack_if_needed(archive_path, cache_root / safe_name)


def _download(kind: str, reference: str, cache_root: Path) -> Path:
    hub = _kagglehub()
    downloaders: dict[str, Callable[[str], str]] = {
        "kaggle_competition": hub.competition_download,
        "kaggle_dataset": hub.dataset_download,
        "kaggle_model": hub.model_download,
        "kaggle_notebook_output": hub.notebook_output_download,
    }
    try:
        downloader = downloaders[kind]
    except KeyError as error:
        raise ValueError(f"unsupported Molab asset kind: {kind!r}") from error
    try:
        downloaded = Path(downloader(reference))
    except Exception as error:
        if error.__class__.__name__ not in {"UnauthenticatedError", "KaggleApiHTTPError"}:
            raise
        return _download_public_archive(kind, reference, cache_root)
    safe_name = sha256(f"{kind}:{reference}".encode("utf-8")).hexdigest()[:16]
    return _unpack_if_needed(downloaded, cache_root / safe_name)


def prepare_molab_public_config(raw: dict[str, Any], workspace: Path) -> dict[str, Any]:
    """Materialize notebook assets and return the existing public CLI contract."""
    asset_root = workspace / "assets"
    asset_root.mkdir(parents=True, exist_ok=True)
    competition_kind = str(raw["COMPETITION_SOURCE_KIND"])
    competition_source = str(raw["COMPETITION_SOURCE"])
    if competition_kind == "local_directory":
        competition_root = Path(competition_source)
    else:
        competition_root = _download(competition_kind, competition_source, asset_root)
    puzzle_info = _exact_glob(competition_root, "puzzle_info.json", "puzzle_info.json")
    data_root = puzzle_info.parent
    test_csv = _exact_glob(data_root, "test.csv", "test.csv")
    sample_submission = _exact_glob(data_root, "sample_submission.csv", "sample_submission.csv")

    model_kind = str(raw["MODEL_ASSET_KIND"])
    model_ref = str(raw["MODEL_ASSET_REF"])
    if model_kind == "local_checkpoint":
        checkpoint = Path(model_ref)
        model_root = checkpoint.parent
    else:
        model_root = _download(model_kind, model_ref, asset_root)
        checkpoint = _exact_glob(model_root, str(raw["CHECKPOINT_GLOB"]), "checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    metadata = _optional_exact_glob(
        model_root, raw.get("CHECKPOINT_METADATA_GLOB"), "checkpoint metadata",
    )
    generator = _optional_exact_glob(
        model_root, raw.get("CHECKPOINT_GENERATOR_GLOB"), "checkpoint generator",
    )

    notebook_digest = sha256(
        json.dumps(raw, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "author_name": raw["AUTHOR_NAME"],
        "checkpoint_path": str(checkpoint),
        "checkpoint_metadata_json": None if metadata is None else str(metadata),
        "checkpoint_generator_json": None if generator is None else str(generator),
        "checkpoint_source_root": str(model_root),
        "puzzle_info_json": str(puzzle_info),
        "test_csv": str(test_csv),
        "sample_submission_csv": str(sample_submission),
        "puzzle_id_start": raw["PUZZLE_ID_START"],
        "puzzle_id_end": raw["PUZZLE_ID_END"],
        "beam_width": raw["BEAM_WIDTH"],
        "max_depth": raw["MAX_DEPTH"],
        "reflect_mode": raw["REFLECT_MODE"],
        "reflect_source_csv": raw.get("REFLECT_SOURCE_CSV"),
        "solution_mode": raw["SOLUTION_MODE"],
        "collect_until_depth": raw["COLLECT_UNTIL_DEPTH"],
        "max_collected_solutions": raw["MAX_COLLECTED_SOLUTIONS"],
        "touch_bfs_radius": raw["TOUCH_BFS_RADIUS"],
        "publish_results": raw["PUBLISH_RESULTS"],
        "results_ingest_url": raw["RESULTS_INGEST_URL"],
        "competition": raw["COMPETITION"],
        "kaggle_owner": "molab",
        "kaggle_slug": f"cayleypy-{raw['COMPETITION']}-example",
        "kaggle_version": 1,
        "kaggle_username": raw["AUTHOR_NAME"],
        "solver_commit": raw["SOLVER_COMMIT"],
        "kaggle_notebook_sha256": notebook_digest,
        "enable_debug": raw["ENABLE_DEBUG"],
        "enable_depth_logs": raw["ENABLE_DEPTH_LOGS"],
        "enable_debug_logs": raw["ENABLE_DEBUG_LOGS"],
        "debug_stream_timing": raw["DEBUG_STREAM_TIMING"],
        "debug_inference_trace": raw["DEBUG_INFERENCE_TRACE"],
        "debug_path_trace": raw["DEBUG_PATH_TRACE"],
        "debug_final_validate": raw["DEBUG_FINAL_VALIDATE"],
        "debug_final_exchange_trace": raw["DEBUG_FINAL_EXCHANGE_TRACE"],
        "debug_final_histogram_trace": raw["DEBUG_FINAL_HISTOGRAM_TRACE"],
        "debug_stream4_histogram_trace": raw["DEBUG_STREAM4_HISTOGRAM_TRACE"],
        "debug_depth_flow_trace": raw["DEBUG_DEPTH_FLOW_TRACE"],
        "debug_pipeline_stats": raw["DEBUG_PIPELINE_STATS"],
        "depth_log_every": raw["DEPTH_LOG_EVERY"],
        "puzzle_log_every": raw["PUZZLE_LOG_EVERY"],
    }
