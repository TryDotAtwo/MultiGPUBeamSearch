"""Fixed public file contract for native cluster archives."""
from __future__ import annotations
from pathlib import PurePosixPath
import re

ROOT_FILES={"run.sh","autotune.sh","cluster.env.example","publication.env.example","README.md","MANIFEST.json","SHA256SUMS"}
DATA_FILES={"data/test.csv","data/puzzle_info.json"}
PROFILE_FILES={"profiles/registry.json"}
SCRIPT_FILES={"scripts/job.sh","scripts/preflight.sh","scripts/autotune_job.sh"}

def require_allowed_path(relative:str)->None:
    path=PurePosixPath(relative); text=path.as_posix()
    allowed=(text in ROOT_FILES|DATA_FILES|PROFILE_FILES|SCRIPT_FILES or text=="bin/production_runner" or
             (len(path.parts)==2 and path.parts[0]=="lib" and re.fullmatch(r"lib[^/]+\.so(?:\.[0-9]+)*",path.name) is not None) or
             (path.parts[0]=="weights" and (path.name=="manifest.json" or path.suffix in {".fp16",".bf16",".pt"})) or
             (path.parts[:2]==("portable","megaminx_cluster") and path.suffix==".py" and len(path.parts) in {3,4} and (len(path.parts)==3 or path.parts[2] in {"scripts","autotune"})) or
             text in {"portable/megaminx_cluster/autotune/calibration.json", "FullBeamNice/generators/p900.json"})
    if not allowed: raise ValueError(f"archive path is outside fixed allowlist: {text}")
