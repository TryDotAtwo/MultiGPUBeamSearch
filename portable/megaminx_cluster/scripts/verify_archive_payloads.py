"""Reverify immutable archive payloads before publication retry."""
from __future__ import annotations
import argparse,json
from pathlib import Path
from portable.megaminx_cluster.scripts.preflight import verify_payload_hashes

def main()->int:
    parser=argparse.ArgumentParser(); parser.add_argument("--archive-root",type=Path,required=True); args=parser.parse_args(); root=args.archive_root.resolve()
    manifest=json.loads((root/"MANIFEST.json").read_text(encoding="utf-8-sig")); verify_payload_hashes(root,manifest["payloads"]); return 0
if __name__=="__main__": raise SystemExit(main())
