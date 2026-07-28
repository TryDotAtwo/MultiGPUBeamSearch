from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping


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

    @property
    def puzzle_ids(self) -> tuple[int, ...]:
        return tuple(range(self.puzzle_id_start, self.puzzle_id_end + 1))

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PublicRunConfig":
        def required(name: str) -> object:
            try:
                return values[name]
            except KeyError as error:
                raise ValueError(f"missing required config field {name.upper()}") from error

        reflect_mode = required("reflect_mode")
        solution_mode = required("solution_mode")
        if reflect_mode not in {"off", "after_original", "only"}:
            raise ValueError("REFLECT_MODE must be off, after_original, or only")
        if solution_mode not in {"first", "collect"}:
            raise ValueError("SOLUTION_MODE must be first or collect")

        start = required("puzzle_id_start")
        end = required("puzzle_id_end")
        if not isinstance(start, int) or not isinstance(end, int) or start > end:
            raise ValueError("PUZZLE_ID range must be inclusive and non-empty")

        positive_fields = ("beam_width", "max_depth", "max_collected_solutions")
        for field in positive_fields:
            value = required(field)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field.upper()} must be positive")

        collect_until_depth = required("collect_until_depth")
        if not isinstance(collect_until_depth, int) or collect_until_depth < 0 or collect_until_depth > required("max_depth"):
            raise ValueError("COLLECT_UNTIL_DEPTH must be within MAX_DEPTH")

        touch_bfs_radius = required("touch_bfs_radius")
        if not isinstance(touch_bfs_radius, int) or not 0 <= touch_bfs_radius <= 12:
            raise ValueError("TOUCH_BFS_RADIUS must be between 0 and 12")

        reflect_source = required("reflect_source_csv")
        if reflect_source is not None and not isinstance(reflect_source, (str, Path)):
            raise ValueError("REFLECT_SOURCE_CSV must be a path or null")
        if reflect_mode == "only" and reflect_source is None:
            raise ValueError("REFLECT_SOURCE_CSV is required when REFLECT_MODE is only")

        return cls(
            author_name=str(required("author_name")),
            checkpoint_path=Path(str(required("checkpoint_path"))),
            puzzle_info_json=Path(str(required("puzzle_info_json"))),
            test_csv=Path(str(required("test_csv"))),
            sample_submission_csv=Path(str(required("sample_submission_csv"))),
            puzzle_id_start=start,
            puzzle_id_end=end,
            beam_width=required("beam_width"),  # type: ignore[arg-type]
            max_depth=required("max_depth"),  # type: ignore[arg-type]
            reflect_mode=reflect_mode,  # type: ignore[arg-type]
            reflect_source_csv=Path(reflect_source) if reflect_source is not None else None,
            solution_mode=solution_mode,  # type: ignore[arg-type]
            collect_until_depth=collect_until_depth,
            max_collected_solutions=required("max_collected_solutions"),  # type: ignore[arg-type]
            touch_bfs_radius=touch_bfs_radius,
            publish_results=bool(required("publish_results")),
            results_ingest_url=str(required("results_ingest_url")),
        )
