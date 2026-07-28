"""CLI entrypoint reserved for the generated public notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.cayleypy_public.config import PublicRunConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    config = PublicRunConfig.from_mapping(json.loads(arguments.config.read_text(encoding="utf-8")))
    print(json.dumps({"puzzle_ids": config.puzzle_ids, "beam_width": config.beam_width}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
