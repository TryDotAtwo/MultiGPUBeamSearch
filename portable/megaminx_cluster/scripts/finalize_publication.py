"""Create publication_context.json from immutable preflight and solve artifacts."""

from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from portable.megaminx_cluster.publication_context import build_publication_context


def enrich_reflected_results(results: Sequence[Mapping[str, object]], supplied_original: Sequence[str] | None) -> list[dict[str, object]]:
    original = list(supplied_original) if supplied_original is not None else None
    for item in results:
        if item.get("search") == "original":
            original = list(item["path"])
            break
    enriched: list[dict[str, object]] = []
    for source in results:
        item = dict(source)
        if item.get("search") == "reflected":
            if not original:
                raise ValueError("reflected publication lacks its original source solution")
            source_text = ".".join(original)
            item["reflected_source_path"] = original
            item["reflected_source_sha256"] = sha256(source_text.encode("utf-8")).hexdigest()
        enriched.append(item)
    return enriched


def _supplied_original() -> list[str] | None:
    value = os.environ.get("MEGAMINX_ORIGINAL_SOLUTION")
    if not value:
        return None
    text = Path(value).read_text(encoding="utf-8").strip()
    return text.split(".") if text else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--cluster-name", required=True)
    parser.add_argument("--author-name", required=True)
    parser.add_argument("--search-mode", choices=("off", "after", "only"), required=True)
    parser.add_argument("--depth", type=int, required=True)
    parser.add_argument("--wall-us", type=int, required=True)
    args = parser.parse_args(argv)
    root = args.archive_root.resolve()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8-sig"))
    manifest["release_manifest_sha256"] = sha256((root / "MANIFEST.json").read_bytes()).hexdigest()
    preflight = json.loads((run_dir / "preflight.json").read_text(encoding="utf-8-sig"))
    selected = json.loads((run_dir / "selected_profile.json").read_text(encoding="utf-8-sig"))
    validated_path = run_dir / "validated_results.json"
    validated = json.loads(validated_path.read_text(encoding="utf-8-sig"))
    validated["results"] = enrich_reflected_results(validated["results"], _supplied_original())
    validated_path.write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    puzzle_id = int(validated["puzzle_id"])
    initial = None
    with (run_dir / "data/test.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["initial_state_id"]) == puzzle_id:
                initial = [int(value) for value in row["initial_state"].split(",")]
                break
    if initial is None:
        raise ValueError(f"puzzle id not found in run-owned test.csv: {puzzle_id}")
    definition = json.loads((root / "data/puzzle_info.json").read_text(encoding="utf-8-sig"))
    proof = {"initial_state": initial, "central_state": definition["central_state"], "generators": definition["generators"]}
    context = build_publication_context(manifest, preflight, selected, proof, run_id=f"slurm-{args.job_id}", job_id=args.job_id, cluster_name=args.cluster_name, author_name=args.author_name, search_mode=args.search_mode, depth=args.depth, solve_us=args.wall_us, wall_us=args.wall_us)
    (run_dir / "publication_context.json").write_text(json.dumps(context, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
