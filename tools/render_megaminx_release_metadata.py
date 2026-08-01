"""Render dynamic, public release provenance into an archive metadata record."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping


def render_metadata(template: Mapping[str, object], tag: str, asset: str, commit: str, model_manifest: Path, weights: Path) -> dict[str, object]:
    if not tag or not asset or len(commit) != 40:
        raise ValueError("release tag, asset, or solver commit is invalid")
    manifest = json.loads(model_manifest.read_text(encoding="utf-8-sig"))
    payloads = [path for path in sorted(weights.rglob("*")) if path.is_file() and path.resolve() != model_manifest.resolve()]
    if not payloads:
        raise ValueError("weight payload is empty")
    digest = sha256()
    for path in payloads:
        digest.update(path.read_bytes())
    return {**dict(template), "release_tag": tag, "release_asset": asset, "solver_commit": commit, "model_manifest": manifest, "model_sha256": digest.hexdigest()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", required=True); parser.add_argument("--asset", required=True); parser.add_argument("--commit", required=True)
    parser.add_argument("--model-manifest", type=Path, required=True); parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    result = render_metadata(json.loads(args.template.read_text(encoding="utf-8-sig")), args.tag, args.asset, args.commit, args.model_manifest, args.weights)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
