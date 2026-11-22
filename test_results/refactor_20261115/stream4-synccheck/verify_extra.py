"""Check the additional source-bound GPU diagnostics and preserve failure labels."""
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def same_text(a, b):
    return a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")


def unpack(name, directory):
    data = (HERE / name).read_bytes()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = archive.getmembers()
        assert len({item.name for item in members}) == len(members)
        assert all(item.name == directory or item.name.startswith(directory + "/") for item in members)
        files = {item.name.removeprefix(directory + "/"): archive.extractfile(item).read()
                 for item in members if item.isfile()}
    assert files["status.txt"].decode().rstrip().endswith("EXIT_CODE=0")
    assert b"COMPLETE\n" in files["status.txt"]
    return files, digest(data)


def check_script(files, name):
    assert same_text(files[name], (HERE / name).read_bytes()), name


def result(files, name, expected, marker=None):
    assert int(files[name + ".exit"]) == expected, name
    log = files[name + ".log"].decode(errors="replace")
    if marker:
        assert marker in log, name
    return log


candidate, candidate_sha = unpack("candidate-evidence.tar.gz", "candidate-results")
prefix, prefix_sha = unpack("prefix-evidence.tar.gz", "prefix-results")
partial, partial_sha = unpack("partial-evidence.tar.gz", "partial-results")
union, union_sha = unpack("union-evidence.tar.gz", "union-results")
final, final_sha = unpack("final-shards-evidence.tar.gz", "final-shards-results")
for files, script in [(candidate, "run-candidate.sh"), (prefix, "run-prefix.sh"),
                      (partial, "run-partial.sh"), (union, "run-union.sh"),
                      (final, "run-final-shards.sh")]:
    check_script(files, script)
for files, source in [(prefix, "prefix-diagnostic.cu"), (partial, "partial-diagnostic.cu")]:
    assert same_text(files[source], (HERE / source).read_bytes())
assert digest(prefix["prefix-diagnostic.cu"]) == "391cc30d6d0b643c0147c842a87008e4d219f11b8e9e3ea0b7eabe632a180799"
assert digest(partial["partial-diagnostic.cu"]) == "33a76fd6659623fa26ffe25c3a65f1af289d45099c62e7cba417afae0f3d0889"
assert same_text(union["stream4_logical_union_cuda_tests.cu"], (REPO / "tests/stream4_logical_union_cuda_tests.cu").read_bytes())

overlay = gzip.decompress((HERE / "candidate-overlay.tar.gz").read_bytes())
assert digest(overlay) == "11b58d6ad64df8fe13a43437836396e0773f758e4b2de20a72fcf8a2d925d3bb"
with tarfile.open(fileobj=io.BytesIO(overlay), mode="r:") as tar:
    overlaid = {member.name: tar.extractfile(member).read() for member in tar.getmembers() if member.isfile()}
assert set(overlaid) == {"cuda/stream4.cu", "tests/stream4_dirty_merge_cuda_tests.cu"}
assert same_text(overlaid["tests/stream4_dirty_merge_cuda_tests.cu"], (REPO / "tests/stream4_dirty_merge_cuda_tests.cu").read_bytes())
for line in candidate["overlay-files-sha256.txt"].decode().splitlines():
    checksum, name = line.split(maxsplit=1)
    assert digest(overlaid[name.removeprefix("candidate/")]) == checksum
source_tar = gzip.decompress((HERE / "source.tar.gz").read_bytes())
with tarfile.open(fileobj=io.BytesIO(source_tar), mode="r:") as tar:
    stream4_original = tar.extractfile("cuda/stream4.cu").read()
assert same_text(stream4_original, (REPO / "cuda/stream4.cu").read_bytes())
final_sources = {
    "stream4_dirty_merge_cuda_tests.cu": overlaid["tests/stream4_dirty_merge_cuda_tests.cu"],
    "stream4_logical_union_cuda_tests.cu": union["stream4_logical_union_cuda_tests.cu"],
    "stream4.cu": stream4_original,
}
for line in final["source-sha256.txt"].decode().splitlines():
    checksum, name = line.split(maxsplit=1)
    assert digest(final_sources[Path(name).name]) == checksum, name

direct_cases = 0
for variant in ("baseline-dirty", "candidate-dirty", "candidate-index"):
    for cap in (1, 2, 3, 4, 257):
        for mode in ("bounded", "unbounded"):
            log = result(candidate, f"{variant}-{cap}-{mode}-direct", 0, "dirty_merge_oracle=pass")
            assert "gpu=0" in log and "gpu=1" in log
            direct_cases += 1
for variant in ("candidate-dirty", "candidate-index"):
    for cap in (1, 2, 3, 4, 257):
        log = result(candidate, f"{variant}-{cap}-bounded-synccheck", 86, "Barrier error detected")
        assert "ERROR SUMMARY:" in log
    for cap in (1, 4, 257):
        result(candidate, f"{variant}-{cap}-bounded-memcheck", 0, "ERROR SUMMARY: 0 errors")
    result(candidate, f"{variant}-1-bounded-racecheck", 86, "RACECHECK SUMMARY:")
    result(candidate, f"{variant}-1-unbounded-synccheck", 0, "ERROR SUMMARY: 0 errors")
for cap in (1, 2, 4):
    result(candidate, f"baseline-dirty-{cap}-bounded-synccheck", 86, "Barrier error detected")

for n in (1, 1025):
    for mode in ("plain-prefix", "conditional-prefix", "first-prefix", "last-prefix"):
        result(prefix, f"{mode}-{n}-direct", 0, "DIAGNOSTIC_OK")
        for tool in ("memcheck", "synccheck"):
            log = result(prefix, f"{mode}-{n}-{tool}", 0, "ERROR SUMMARY: 0 errors")
            assert "DIAGNOSTIC_OK" in log
    for mode in ("plain", "conditional", "first"):
        for active in (1, 65, 256):
            result(partial, f"{mode}-partial{active}-{n}-direct", 0, "DIAGNOSTIC_OK")
            log = result(partial, f"{mode}-partial{active}-{n}-synccheck", 0, "ERROR SUMMARY: 0 errors")
            assert "DIAGNOSTIC_OK" in log
    result(partial, f"first-partial1-{n}-memcheck", 0, "ERROR SUMMARY: 0 errors")
for variant in ("baseline-index", "candidate-index"):
    for tool in ("direct", "memcheck", "synccheck"):
        log = result(union, f"{variant}-{tool}", 0, "PASS: logical union and exact physical histograms")
        assert "sort_item_bound=1" in log and "gpu=0" in log and "gpu=1" in log
        if tool != "direct":
            assert "ERROR SUMMARY: 0 errors" in log

for variant in ("baseline-dirty", "baseline-index"):
    for cap in (1, 2, 3, 4, 257):
        for mode in ("bounded", "unbounded"):
            log = result(final, f"{variant}-{cap}-{mode}-direct", 0, "dirty_merge_oracle=pass")
            assert "gpu=0" in log and "gpu=1" in log
    for cap in (1, 257):
        result(final, f"{variant}-{cap}-bounded-memcheck", 0, "ERROR SUMMARY: 0 errors")

print(json.dumps({"candidate_archive": candidate_sha, "prefix_archive": prefix_sha,
                  "partial_archive": partial_sha, "union_archive": union_sha,
                  "final_shards_archive": final_sha,
                  "candidate_direct_runs": direct_cases,
                  "candidate_bounded_synccheck": "FAIL: reproduced; not promoted",
                  "minimal_prefix_and_partial": "PASS in tested scope",
                  "final_baseline_shard_runs": 20}, indent=2, sort_keys=True))
