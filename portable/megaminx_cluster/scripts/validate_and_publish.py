"""Publish an already validated run without rerunning the solver."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from portable.megaminx_cluster.publish import poll_receipt, publish_existing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--poll", action="store_true")
    args = parser.parse_args(argv)
    result = publish_existing(args.run_dir.resolve(), args.url)
    if not result.ok:
        print(f"publish_failed={result.safe_error}", file=sys.stderr)
        return 3 if result.retryable else 2
    if args.poll:
        for receipt in result.receipts:
            poll_receipt(receipt, args.run_dir / f"status-{receipt.submission_id}.json")
    print("published=" + ",".join(receipt.submission_id for receipt in result.receipts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
