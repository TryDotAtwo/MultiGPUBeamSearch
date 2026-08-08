from dataclasses import dataclass
from ipaddress import ip_address
import os
from pathlib import Path
import re
from typing import Literal, Mapping
from urllib.parse import urlsplit


_ALLOWED_KEYS = frozenset({
    "author_name", "checkpoint_path", "puzzle_info_json", "test_csv",
    "sample_submission_csv", "puzzle_id_start", "puzzle_id_end", "beam_width",
    "max_depth", "reflect_mode", "reflect_source_csv", "solution_mode",
    "collect_until_depth", "max_collected_solutions", "touch_bfs_radius",
    "publish_results", "results_ingest_url", "competition", "kaggle_owner",
    "kaggle_slug", "kaggle_version", "kaggle_username", "solver_commit",
    "kaggle_notebook_sha256",
})
_FORBIDDEN_PUBLIC_KEYS = frozenset({"model_source", "model_dtype", "checkpoint_format"})
_HEX_40 = re.compile(r"^[0-9a-f]{40}$")
_HEX_64 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER_PREFIX = re.compile(r"^replace(?:[-_\s]+)with(?:[-_\s]+)", re.IGNORECASE)
_RESULTS_INGEST_URL_ENV = "CAYLEYPY_RESULTS_INGEST_URL"


def _validated_results_ingest_url(value: str) -> str:
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in value
    ):
        raise ValueError("RESULTS_INGEST_URL must be a safe public HTTPS URL")
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        hostname_key = hostname.rstrip(".").lower() if hostname is not None else ""
        try:
            address = ip_address(hostname_key)
        except ValueError:
            public_host = hostname_key != "localhost" and not hostname_key.endswith(".localhost")
        else:
            public_host = address.is_global
        safe = (
            parsed.scheme == "https"
            and parsed.hostname is not None
            and parsed.username is None
            and public_host
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        safe = False
    if not safe:
        raise ValueError(
            "RESULTS_INGEST_URL must be HTTPS without credentials, query, or fragment"
        )
    return value


def _is_placeholder(value: str | None) -> bool:
    return value is not None and _PLACEHOLDER_PREFIX.match(value) is not None


@dataclass(frozen=True)
class PublicRunConfig:
    author_name: str
    checkpoint_path: Path
    puzzle_info_json: Path
    test_csv: Path
    sample_submission_csv: Path
    puzzle_id_start: int
    puzzle_id_end: int
    beam_width: int
    max_depth: int
    reflect_mode: Literal["off", "after_original", "only"]
    reflect_source_csv: Path | None
    solution_mode: Literal["first", "collect"]
    collect_until_depth: int
    max_collected_solutions: int
    touch_bfs_radius: int
    publish_results: bool
    results_ingest_url: str
    model_dtype: Literal["fp16"] = "fp16"
    competition: str | None = None
    kaggle_owner: str | None = None
    kaggle_slug: str | None = None
    kaggle_version: int | None = None
    kaggle_username: str | None = None
    solver_commit: str | None = None
    kaggle_notebook_sha256: str | None = None

    @property
    def puzzle_ids(self) -> tuple[int, ...]:
        return tuple(range(self.puzzle_id_start, self.puzzle_id_end + 1))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PublicRunConfig":
        if not isinstance(values, Mapping):
            raise ValueError("config must be an object")
        unknown = set(values).difference(_ALLOWED_KEYS)
        if unknown:
            forbidden = sorted(unknown.intersection(_FORBIDDEN_PUBLIC_KEYS))
            if forbidden:
                raise ValueError(
                    "public checkpoint-only config does not accept "
                    + ", ".join(name.upper() for name in forbidden)
                )
            raise ValueError("unknown config fields: " + ", ".join(sorted(unknown)))

        def required(name: str) -> object:
            try:
                return values[name]
            except KeyError as error:
                raise ValueError(f"missing required config field {name.upper()}") from error

        def nonempty_string(name: str, *, optional: bool = False) -> str | None:
            value = values.get(name) if optional else required(name)
            if value is None and optional:
                return None
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name.upper()} must be a non-empty string")
            return value.strip()

        def integer(name: str) -> int:
            value = required(name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name.upper()} must be an integer")
            return value

        reflect_mode = required("reflect_mode")
        solution_mode = required("solution_mode")
        if reflect_mode not in {"off", "after_original", "only"}:
            raise ValueError("REFLECT_MODE must be off, after_original, or only")
        if solution_mode not in {"first", "collect"}:
            raise ValueError("SOLUTION_MODE must be first or collect")

        start = integer("puzzle_id_start")
        end = integer("puzzle_id_end")
        if start > end:
            raise ValueError("PUZZLE_ID range must be inclusive and non-empty")

        positive_values = {
            name: integer(name)
            for name in ("beam_width", "max_depth", "max_collected_solutions")
        }
        for name, value in positive_values.items():
            if value <= 0:
                raise ValueError(f"{name.upper()} must be positive")

        collect_until_depth = integer("collect_until_depth")
        if collect_until_depth < 0 or collect_until_depth > positive_values["max_depth"]:
            raise ValueError("COLLECT_UNTIL_DEPTH must be within MAX_DEPTH")

        touch_bfs_radius = integer("touch_bfs_radius")
        if not 0 <= touch_bfs_radius <= 12:
            raise ValueError("TOUCH_BFS_RADIUS must be between 0 and 12")

        reflect_source = required("reflect_source_csv")
        if reflect_source is not None and not isinstance(reflect_source, (str, Path)):
            raise ValueError("REFLECT_SOURCE_CSV must be a path or null")
        if reflect_mode == "only" and reflect_source is None:
            raise ValueError("REFLECT_SOURCE_CSV is required when REFLECT_MODE is only")

        publish_results = required("publish_results")
        if not isinstance(publish_results, bool):
            raise ValueError("PUBLISH_RESULTS must be a bool")
        raw_ingest_url = values.get("results_ingest_url")
        if raw_ingest_url is None or raw_ingest_url == "":
            configured_ingest_url = None
        elif not isinstance(raw_ingest_url, str):
            raise ValueError("RESULTS_INGEST_URL must be a non-empty string")
        else:
            configured_ingest_url = raw_ingest_url.strip() or None
        if publish_results and configured_ingest_url is None:
            ingest_url = os.environ.get(_RESULTS_INGEST_URL_ENV, "").strip() or None
        else:
            ingest_url = configured_ingest_url
        if publish_results and ingest_url is None:
            raise ValueError(
                "RESULTS_INGEST_URL or CAYLEYPY_RESULTS_INGEST_URL must be set "
                "when PUBLISH_RESULTS is true"
            )
        if ingest_url is not None:
            ingest_url = _validated_results_ingest_url(ingest_url)

        competition = nonempty_string("competition", optional=True)
        kaggle_owner = nonempty_string("kaggle_owner", optional=True)
        kaggle_slug = nonempty_string("kaggle_slug", optional=True)
        kaggle_username = nonempty_string("kaggle_username", optional=True)
        solver_commit = nonempty_string("solver_commit", optional=True)
        notebook_sha256 = nonempty_string("kaggle_notebook_sha256", optional=True)
        kaggle_version = values.get("kaggle_version")
        if kaggle_version is not None and (
            not isinstance(kaggle_version, int)
            or isinstance(kaggle_version, bool)
            or kaggle_version <= 0
        ):
            raise ValueError("KAGGLE_VERSION must be a positive integer or null")
        if solver_commit is not None and _HEX_40.fullmatch(solver_commit) is None:
            raise ValueError("SOLVER_COMMIT must be 40 lowercase hexadecimal characters")
        if notebook_sha256 is not None and _HEX_64.fullmatch(notebook_sha256) is None:
            raise ValueError(
                "KAGGLE_NOTEBOOK_SHA256 must be 64 lowercase hexadecimal characters"
            )
        author_name = nonempty_string("author_name")
        assert author_name is not None
        if publish_results:
            provenance = {
                "COMPETITION": competition,
                "KAGGLE_OWNER": kaggle_owner,
                "KAGGLE_SLUG": kaggle_slug,
                "KAGGLE_VERSION": kaggle_version,
                "SOLVER_COMMIT": solver_commit,
                "KAGGLE_NOTEBOOK_SHA256": notebook_sha256,
            }
            missing = sorted(name for name, value in provenance.items() if value in {None, ""})
            if missing:
                raise ValueError(
                    "publication provenance is required: " + ", ".join(missing)
                )
            publication_identity = {
                "AUTHOR_NAME": author_name,
                "COMPETITION": competition,
                "KAGGLE_OWNER": kaggle_owner,
                "KAGGLE_SLUG": kaggle_slug,
                "KAGGLE_USERNAME": kaggle_username,
            }
            placeholders = sorted(
                name for name, value in publication_identity.items() if _is_placeholder(value)
            )
            if placeholders:
                raise ValueError(
                    "publication placeholders must be replaced: " + ", ".join(placeholders)
                )
        return cls(
            author_name=author_name,
            checkpoint_path=Path(str(required("checkpoint_path"))),
            puzzle_info_json=Path(str(required("puzzle_info_json"))),
            test_csv=Path(str(required("test_csv"))),
            sample_submission_csv=Path(str(required("sample_submission_csv"))),
            puzzle_id_start=start,
            puzzle_id_end=end,
            beam_width=positive_values["beam_width"],
            max_depth=positive_values["max_depth"],
            reflect_mode=reflect_mode,
            reflect_source_csv=Path(reflect_source) if reflect_source is not None else None,
            solution_mode=solution_mode,
            collect_until_depth=collect_until_depth,
            max_collected_solutions=positive_values["max_collected_solutions"],
            touch_bfs_radius=touch_bfs_radius,
            publish_results=publish_results,
            results_ingest_url=ingest_url or "",
            competition=competition,
            kaggle_owner=kaggle_owner,
            kaggle_slug=kaggle_slug,
            kaggle_version=kaggle_version,
            kaggle_username=kaggle_username,
            solver_commit=solver_commit,
            kaggle_notebook_sha256=notebook_sha256,
        )
