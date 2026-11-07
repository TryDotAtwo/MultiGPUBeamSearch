"""Assemble the small private Kaggle dataset used by paired paper sweeps."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "paper/kaggle_review_sweeps/bundle_stage"
RUNTIME_SRC = ROOT / "test_results/cube4_full_transformer_inference_archive/cube4_full_transformer_inference"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    runtime = STAGE / "pilgrim_runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    shutil.copy2(RUNTIME_SRC / "test.py", runtime / "test.py")
    if (runtime / "pilgrim").exists():
        shutil.rmtree(runtime / "pilgrim")
    shutil.copytree(RUNTIME_SRC / "pilgrim", runtime / "pilgrim")
    shutil.copytree(RUNTIME_SRC / "unified_training", runtime / "unified_training", dirs_exist_ok=True)
    for cache in runtime.rglob("__pycache__"):
        shutil.rmtree(cache)
    (runtime / "generators").mkdir(exist_ok=True)
    (runtime / "targets").mkdir(exist_ok=True)
    shutil.copy2(ROOT / "FullBeamNice/generators/p900.json", runtime / "generators/p900.json")
    shutil.copy2(ROOT / "FullBeamNice/targets/p900-t000.pt", runtime / "targets/p900-t000.pt")

    generators = json.loads((runtime / "generators/p900.json").read_text())["actions"]
    target = torch.load(runtime / "targets/p900-t000.pt", map_location="cpu", weights_only=False)
    state = target.clone().to(torch.long)
    for move in (0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11):
        state = state[torch.tensor(generators[move], dtype=torch.long)]
    torch.save(state.to(torch.uint8), runtime / "paper_state.pt")

    scalar = {"group_id": 900, "target_id": 0, "hd1": 2048, "hd2": 512,
              "nrd": 8, "dropout": 0.0, "model_name": "megaminx2048-512-8-e4000",
              "model_mode": "MLP2RB", "model_id": 4000, "n_gens": 24}
    q = json.loads((ROOT / "FullBeamNice/logs/model_p900-t000-q-sym_1777988767.json").read_text())
    (runtime / "model_output1.json").write_text(json.dumps(scalar, indent=2) + "\n")
    (runtime / "model_output24.json").write_text(json.dumps(q, indent=2) + "\n")

    for cache in STAGE.rglob("__pycache__"):
        shutil.rmtree(cache)
    checksum_files = [p for p in STAGE.rglob("*") if p.is_file()
                      and "__pycache__" not in p.parts and p.suffix != ".pyc"
                      and p.name not in {"benchmark_manifest.json", "dataset-metadata.json", "bundle_payload.zip"}]
    checksums = {p.relative_to(STAGE).as_posix(): sha256(p) for p in checksum_files}
    manifest = {
        "manifest_version": 1, "move_count": 24, "output_dims": [1, 24],
        "fixed_state": "pilgrim_runtime/paper_state.pt",
        "checkpoints": {
            "1": {"model_glob": "**/weights_megaminx2048_512_8_e4000.pth",
                  "sha256": "7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759",
                  "architecture": "Pilgrim MLP hd1=2048 hd2=512 nrd=8 scalar head",
                  "output_dim": 1, "training_run_id": "megaminx2048-512-8-e4000",
                  "native_head": True},
            "24": {"model_glob": "**/p900-t000-q-sym_1777988767_best.pth",
                   "sha256": "e7bda332b53acc9363edd8ec682a211c1a8b8a315ec26ff8e8928efa5d2ca670",
                   "architecture": "Pilgrim QMLP2RB hd1=1536 hd2=512 nrd=2 24-action head",
                   "output_dim": 24, "training_run_id": "p900-t000-q-sym-1777988767",
                   "native_head": True},
        },
        "sha256": checksums,
    }
    (STAGE / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    metadata = {"title": "Paper Paired Megaminx Benchmark Bundle",
                "id": "trydotatwo/paper-paired-megaminx-benchmark-bundle",
                "licenses": [{"name": "CC0-1.0"}]}
    (STAGE / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
