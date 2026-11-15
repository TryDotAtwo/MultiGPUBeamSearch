"""Verify source-bound collective-goal evidence and replay recovery paths."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import subprocess
import tarfile

root = Path(__file__).resolve().parents[3]
out = Path(__file__).resolve().parent
sha = lambda data: hashlib.sha256(data).hexdigest()
info = json.loads((out / "archive-verification.json").read_text(encoding="utf-8"))
data = (out / "remote-evidence.tar.gz").read_bytes()
assert len(data) == info["bytes"] and sha(data) == info["sha256"]
with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
    files = {}
    for member in tar:
        if member.isdir():
            continue
        assert member.isfile() and member.name.startswith("cnot-goal-results/")
        name = member.name.removeprefix("cnot-goal-results/")
        assert name not in files and ".." not in Path(name).parts
        contents = tar.extractfile(member).read()
        assert len(contents) == member.size
        files[name] = contents

manifest = json.loads(files["source-manifest.json"])
commit = "7c25b3c6ecd0becdcbd8d819a71bb575eaad3912"
assert manifest["base_commit"] == commit
base = subprocess.check_output(["git", "archive", "--format=tar", commit, "--",
                                *sorted(manifest["baseline"])], cwd=root)
with tarfile.open(fileobj=io.BytesIO(base)) as tar:
    actual = {entry.name: sha(tar.extractfile(entry).read()) for entry in tar if entry.isfile()}
assert actual == manifest["baseline"] and len(actual) == 541
assert len(manifest["overlays"]) == 3
for name, digest in manifest["overlays"].items():
    assert sha((root / name).read_bytes().replace(b"\r\n", b"\n")) == digest, name
baseline_hash = manifest["baseline"]["integrations/cayleypy_native/multigpubeamsearch/cnot.py"]
assert files["baseline-cnot-sha256.txt"].decode().split()[0] == baseline_hash
assert files["source-before.txt"] == files["source-after.txt"]
assert b"verified source files 543" in files["source-before.txt"]

assert files["status.txt"].decode().splitlines() == [
    "RED_TESTS", "RED_EXIT=1", "PYTHON_TESTS", "GPU_ONE", "GPU_TWO", "COMPLETE", "EXIT_CODE=0"]
assert b"4 failed, 1 passed" in files["red.log"]
assert b"goal calculation failed" in files["red.log"]
assert b"675 passed, 17 skipped" in files["python-tests.log"]
assert files["gpus.csv"].count(b"NVIDIA GeForce RTX 3060") == 2
assert b"2.9.0+cu128" in files["software.txt"]

summary = {}
for world in (1, 2):
    ranks = []
    for rank in range(world):
        receipt = json.loads(files[f"receipts/gpu-{world}/rank-{rank}.json"])
        assert (receipt["rank"], receipt["world"], receipt["device"]) == (rank, world, f"cuda:{rank}")
        assert receipt["gpu"] == "NVIDIA GeForce RTX 3060"
        assert receipt["rows"] == [8, 4, 2, 1]
        assert receipt["result"]["status"] == "depth_limit"
        assert [layer["retained"] for layer in receipt["result"]["layers"]] == [5, 5, 5]
        if world == 2:
            assert receipt["result"]["layers"][1]["by_rank"][1][0] == 2
            assert len(receipt["faults"]) == 2
            for fault_rank, fault in enumerate(receipt["faults"]):
                assert fault["fault_rank"] == fault_rank and fault["reused"] is True
                assert f"rank {fault_rank}: ValueError: injected goal failure" in fault["error"]
                recovered = fault["recovered"]
                assert recovered["status"] == "solved" and recovered["gates"] == [0]
                rows = [1, 2]
                for gate in recovered["gates"]:
                    control, offset = divmod(gate, len(rows) - 1)
                    rows[offset + (offset >= control)] ^= rows[control]
                assert rows == [1, 3]
        else:
            assert not receipt["faults"]
        pairs = receipt["pairs"]
        assert len(pairs) == 5 and all(p["old_seconds"] > 0 and p["new_seconds"] > 0 for p in pairs)
        old = statistics.median(p["old_seconds"] for p in pairs)
        new = statistics.median(p["new_seconds"] for p in pairs)
        assert old == receipt["old_median"] and new == receipt["new_median"]
        assert f"CNOT_GOAL_COLLECTIVE_OK rank={rank} world={world}".encode() in files[f"gpu-{world}.log"]
        ranks.append(receipt)
    if world == 2:
        assert ranks[0]["result"] == ranks[1]["result"]
        assert ranks[0]["pairs"] == ranks[1]["pairs"]
        assert ranks[0]["faults"] == ranks[1]["faults"]
    summary[f"gpu_{world}"] = {
        "old_median_seconds": old,
        "new_median_seconds": new,
        "relative_medians_percent": 100 * (old - new) / old,
        "median_paired_percent": statistics.median(100 * (p["old_seconds"] - p["new_seconds"]) / p["old_seconds"] for p in pairs),
    }

print("PASS: 541 base files, 3 overlays, red/green CPU and two-GPU rank failures, "
      "4 independently replayed recovery receipts")
print(json.dumps(summary, indent=2))
