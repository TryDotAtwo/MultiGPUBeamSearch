"""Verify CNOT distinct evidence and source overlays; optionally pass a commit."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
archive = here / "cnot-distinct-evidence.tar.gz"
expected = (here / "cnot-distinct-evidence.sha256").read_text().split()[0]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
commit = sys.argv[1] if len(sys.argv) > 1 else None
with tarfile.open(archive, "r:gz") as tar:
    def read(name):
        return tar.extractfile("cnot-distinct-results/" + name).read()
    manifest = json.loads(read("source-manifest.json"))
    assert manifest["base_commit"] == "f6dd04684cb8e9fdcfcc775b240acc9cd28d29c8"
    assert len(manifest["overlays"]) == 6
    for name, digest in manifest["overlays"].items():
        raw = (subprocess.check_output(["git", "show", commit + ":" + name], cwd=root)
               if commit else (root / name).read_bytes())
        assert hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() == digest, name
    assert read("source-before.txt") == read("source-after.txt")
    assert b"COMPLETE\nEXIT_CODE=0\n" in read("status.txt")
    for filename, pattern in (("cnot-cpu.log", rb"\b34 passed\b"),
                              ("cnot-cuda.log", rb"\b26 passed\b"),
                              ("python-tests.log", rb"\b738 passed, 10 skipped\b")):
        assert re.search(pattern, read(filename)), filename
    for rank in range(2):
        assert f"CNOT_RADIX_NCCL_OK rank={rank} cases=3".encode() in read("radix-nccl.log")
    for world in (1, 2):
        gpu = [json.loads(read(f"receipts/gpu-{world}/rank-{rank}.json")) for rank in range(world)]
        checkpoint = [json.loads(read(f"receipts/checkpoint-{world}/rank-{rank}.json")) for rank in range(world)]
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
print("verified source overlays, CNOT distinct and bounded GPU evidence", expected)
