"""Check this stage's source identity and bounded remote acceptance receipts."""
import hashlib
import json
from pathlib import Path
import re
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
archive = here / "cnot-rust-evidence.tar.gz"
expected = (here / "cnot-rust-evidence.sha256").read_text().split()[0]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
with tarfile.open(archive, "r:gz") as tar:
    def read(name):
        return tar.extractfile("cnot-rust-results/" + name).read()
    manifest = json.loads(read("source-manifest.json"))
    assert manifest["base_commit"] == "1f8e40b24e3163b27c4fcd621dc18ac0073ff967"
    for name, digest in manifest["overlays"].items():
        actual = hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert actual == digest, (name, actual, digest)
    assert read("source-before.txt") == read("source-after.txt")
    assert b"COMPLETE\nEXIT_CODE=0\n" in read("status.txt")
    assert re.search(rb"\b31 passed\b", read("cnot-cpu.log"))
    assert re.search(rb"\b8 passed\b", read("cnot-cuda.log"))
    assert re.search(rb"\b717 passed, 10 skipped\b", read("python-tests.log"))
    for world in (1, 2):
        gpu = [json.loads(read(f"receipts/gpu-{world}/rank-{rank}.json")) for rank in range(world)]
        checkpoint = [json.loads(read(f"receipts/checkpoint-{world}/rank-{rank}.json"))
                      for rank in range(world)]
        assert all(row["world"] == world and len(row["cases"]) == 5 for row in gpu)
        assert all(row["cases"] == gpu[0]["cases"] for row in gpu)
        if world == 2:
            assert all(row["reused_after_failure"] for row in gpu)
        for row in checkpoint:
            assert row["world"] == world
            assert row["checkpoint_sha256"] == "92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4"
            qhat, proposal = row["results"]
            assert qhat["result"]["gates"] == [5, 7, 0] and qhat["replay"]
            assert "feature F02 rejected" in proposal["error"]
        assert all(row["results"] == checkpoint[0]["results"] for row in checkpoint)
print("verified source overlays, CPU/GPU receipts, checkpoint and archive", expected)
