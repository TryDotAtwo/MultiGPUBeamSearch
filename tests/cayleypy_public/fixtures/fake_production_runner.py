"""Deterministic subprocess fixture for runner integration tests."""

import os
from pathlib import Path


def main() -> int:
    result = os.environ.get("BEAM_SOLVE_BUCKET_RESULT_TSV")
    if result:
        path = Path(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "puzzle_id\tdepth_index\tfound_depth\ttotal_depth\tknown_length\tdelta\towner_rank\tsolution_path\n"
            "7\t0\t1\t1\t0\t0\t0\tcounterclockwise\n", encoding="utf-8"
        )
        print("collection_status=capacity_reached")
    else:
        print("solution_path=counterclockwise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
