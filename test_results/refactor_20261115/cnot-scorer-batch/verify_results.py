"""Check source-bound CNOT batch-scorer receipts and independently replay paths."""
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
        assert member.isfile() and member.name.startswith("cnot-batch-results/")
        name = member.name.removeprefix("cnot-batch-results/")
        assert name not in files and ".." not in Path(name).parts
        contents = tar.extractfile(member).read()
        assert len(contents) == member.size
        files[name] = contents

manifest = json.loads(files["source-manifest.json"])
commit = "2efd9a9bde065aca88e5f06394dd31d8097dbfc0"
assert manifest["base_commit"] == commit
base = subprocess.check_output(["git", "archive", "--format=tar", commit, "--",
                                *sorted(manifest["baseline"])], cwd=root)
with tarfile.open(fileobj=io.BytesIO(base)) as tar:
    actual = {entry.name: sha(tar.extractfile(entry).read()) for entry in tar if entry.isfile()}
assert actual == manifest["baseline"] and len(actual) == 539
assert len(manifest["overlays"]) == 5
for name, digest in manifest["overlays"].items():
    assert sha((root / name).read_bytes().replace(b"\r\n", b"\n")) == digest, name
assert files["source-before.txt"] == files["source-after.txt"]
assert b"verified source files 541" in files["source-before.txt"]

private = json.loads(files["private-manifest.json"])
assert private["source_commit"] == "c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb"
assert private["source_files"] == 621
assert private["source_tar_sha256"] == "fb317bbaeb11b18888f35db44c34b4b79bbe113d822a463a2cc62b39032f72b1"
assert private["source_manifest_sha256"] == "8002f6a804c36de3566767fb068e768f6ae5b50cd0c5c3afab7bddf6e4c31361"
assert private["bridge_sha256"] == manifest["overlays"]["integrations/cnot_scorer/bridge.rs"]
assert private["prepared_sha256"] == "bf60650600830abfa830da5a164fa1f34d4a455c3efc8ea88a149c5530f7c823"
assert private["fixture_sha256"] == "d6df7487ac5ba6cfc274b87e59a7b38ff334d0cd4cb8ae5d022dd8bce1e659d6"
assert private["fixture_bytes"] == 520324
private_log = (out / "private-source-verification.log").read_text(encoding="utf-8")
assert json.loads(private_log[private_log.index("{"):]) == private

assert files["status.txt"].decode().splitlines() == [
    "SOURCE_OK", "SCORER_BUILD", "PYTHON_TESTS", "CHECKPOINT_TESTS",
    "SEARCH_ONE_GPU", "SEARCH_TWO_GPU", "COMPLETE", "EXIT_CODE=0"]
assert b"Finished `release` profile" in files["scorer-build.log"]
assert b"670 passed, 17 skipped" in files["python-tests.log"]
for gpu in (0, 1):
    assert b"3 passed, 1 skipped" in files[f"scorer-{gpu}.log"]
assert files["gpus.csv"].count(b"NVIDIA GeForce RTX 3060") == 2
assert b"2.9.0+cu128" in files["software.txt"]

summary = {}
for world in (1, 2):
    ranks = []
    for rank in range(world):
        receipt = json.loads(files[f"receipts/gpu-{world}/rank-{rank}.json"])
        assert (receipt["rank"], receipt["world"], receipt["device"]) == (rank, world, f"cuda:{rank}")
        assert receipt["gpu"] == "NVIDIA GeForce RTX 3060"
        assert receipt["checkpoint_sha256"] == "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4"
        assert receipt["rows"] == [1, 7, 4, 10]
        assert receipt["result"]["status"] == "solved"
        assert receipt["result"]["gates"] == [5, 7, 0]
        # Independent construction replay: decode physical IDs and XOR rows.
        rows = [1 << i for i in range(len(receipt["rows"]))]
        for gate in receipt["result"]["gates"]:
            assert 0 <= gate < len(rows) * (len(rows) - 1)
            control, offset = divmod(gate, len(rows) - 1)
            target = offset + (offset >= control)
            rows[target] ^= rows[control]
        assert rows == receipt["rows"]
        pairs = receipt["pairs"]
        assert len(pairs) == 5 and all(p["old_seconds"] > 0 and p["new_seconds"] > 0 for p in pairs)
        old = statistics.median(p["old_seconds"] for p in pairs)
        new = statistics.median(p["new_seconds"] for p in pairs)
        assert old == receipt["old_median"] and new == receipt["new_median"]
        ranks.append(receipt)
    if world == 2:
        assert ranks[0]["result"] == ranks[1]["result"] and ranks[0]["pairs"] == ranks[1]["pairs"]
    pairs = ranks[0]["pairs"]
    summary[f"gpu_{world}"] = {
        "old_median_seconds": old,
        "new_median_seconds": new,
        "relative_medians_percent": 100 * (old - new) / old,
        "median_paired_percent": statistics.median(100 * (p["old_seconds"] - p["new_seconds"]) / p["old_seconds"] for p in pairs),
    }
    for rank in range(world):
        assert f"CNOT_CHECKPOINT_BATCH_OK rank={rank} world={world}".encode() in files[f"search-{world}.log"]

print("PASS: 539 base files, 5 overlays, private source manifest, checkpoint scorer on 2 GPUs, "
      "and 3 independently replayed rank receipts")
print(json.dumps(summary, indent=2))
