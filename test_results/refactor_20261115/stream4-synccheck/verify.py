"""Verify provenance and print the recorded outcomes without reclassifying failures."""
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE = "26fb7ed824e17a78d0b286f7197e4faa647f6254"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def same_text(a, b):
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


with tarfile.open(HERE / "evidence.tar.gz", "r:gz") as archive:
    members = archive.getmembers()
    assert len({member.name for member in members}) == len(members)
    assert all(member.name == "results" or member.name.startswith("results/") for member in members)
    files = {member.name.removeprefix("results/"): archive.extractfile(member).read()
             for member in members if member.isfile()}

for name in ("chained-diagnostic.cu", "run.sh"):
    assert same_text(files[name], (HERE / name).read_bytes()), name
assert sha(files["chained-diagnostic.cu"]) == "fcc267a0fd202947a06c8720decaec11f45714a3797091874f7f6ef6c9ae75c2"
assert files["status.txt"].decode().rstrip().endswith("EXIT_CODE=0")
assert b"COMPLETE\n" in files["status.txt"]
assert b"OK" in files["source-archive-check.txt"]
assert b"OK" in files["diagnostic-check.txt"]
source_tar = __import__("gzip").decompress((HERE / "source.tar.gz").read_bytes())
assert sha(source_tar) == "a937e350c15b0805cfc872a9f40accae972db581d38602f63ee6b1788bae1bcd"
with tarfile.open(fileobj=io.BytesIO(source_tar), mode="r:") as source_archive:
    source_members = [member for member in source_archive.getmembers() if member.isfile()]
    source_files = {member.name: source_archive.extractfile(member).read() for member in source_members}
assert len(source_files) == len(source_members)
requests = "".join(f"{BASE}:{name}\n" for name in source_files).encode()
batch = io.BytesIO(subprocess.check_output(["git", "cat-file", "--batch"], input=requests, cwd=REPO))
for name, exported in source_files.items():
    header = batch.readline().split()
    assert len(header) == 3 and header[1] == b"blob", name
    original = batch.read(int(header[2]))
    assert batch.read(1) == b"\n", name
    if exported != original:
        # git archive on this Windows checkout exports configured CRLF text.
        assert b"\0" not in exported and b"\0" not in original, name
        assert exported.replace(b"\r\n", b"\n") == original.replace(b"\r\n", b"\n"), name
assert batch.read() == b""
for line in files["key-source-sha256.txt"].decode().splitlines():
    digest, name = line.split(maxsplit=1)
    assert sha(source_files[name]) == digest, name

outcomes = {}
for name in sorted(files):
    if not name.endswith(".exit"):
        continue
    stem = name.removesuffix(".exit")
    code = int(files[name])
    log = files.get(stem + ".log", b"").decode(errors="replace")
    summaries = re.findall(r"ERROR SUMMARY: ([^\n]+)", log)
    outcomes[stem] = {"exit": code, "summaries": summaries,
                      "diagnostic_ok": "DIAGNOSTIC_OK" in log,
                      "barrier_errors": log.count("Barrier error detected")}
print(json.dumps({"base_commit": BASE, "archive_sha256": sha((HERE / "evidence.tar.gz").read_bytes()),
                  "files": len(files), "source_files": len(source_files), "outcomes": outcomes},
                 indent=2, sort_keys=True))
