"""Verify the source-pinned opt-in CNOT search bridge evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

here = Path(__file__).resolve().parent
root = here.parents[2]
commit = sys.argv[1] if len(sys.argv) > 1 else None
base = "0aab5c5111cf0f9f8c7ab823ee8570d5f9dbc8eb"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source(name):
    if commit:
        return subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=root)
    return (root / name).read_bytes().replace(b"\r\n", b"\n")


def evidence(filename, expected):
    path = here / filename
    assert digest(path.read_bytes()) == expected, filename
    return tarfile.open(path, "r:gz")


with evidence("cnot-unseen-bridge-evidence.tar.gz",
              "18cd35b91c67230fda8ec29141e619d33c8c3572e56ee3b019bd1f7680aaf28f") as tar:
    def first(name):
        return tar.extractfile("cnot-unseen-bridge-results/" + name).read()
    previous = json.loads(first("source-manifest.json"))
    assert b"RED_CONFIRMED\n" in first("status.txt")
    assert b"BRIDGE_ONE\nEXIT_CODE=1\n" in first("status.txt")
    assert b"explicit invalid owned CNOT library was not rejected" in first("red.log")
    assert b"explicit invalid owned CNOT library was not rejected" in first("bridge-1.log")

with evidence("cnot-unseen-bridge-continue-evidence.tar.gz",
              "92e0cce7f2496a0c89f9f9f91e594b25b175ec984110837dc5dacf0bc5d0be3d") as tar:
    def failed_build(name):
        return tar.extractfile("cnot-unseen-bridge-continue-results/" + name).read()
    assert b"RUST_BUILD\nEXIT_CODE=101\n" in failed_build("status.txt")
    assert b"#![forbid(unsafe_code)]" in failed_build("rust-build.log")
    assert b"usage of an `unsafe` block" in failed_build("rust-build.log")
    assert b"previous binary lacked bridge marker" in failed_build("stale-binary.txt")

with evidence("cnot-unseen-bridge-fix-evidence.tar.gz",
              "4c0f798699b2b8b76ce288813bbad6ac9cfce9aa77e13de74858df0e337b65c4") as tar:
    def fixed(name):
        return tar.extractfile("cnot-unseen-bridge-fix-results/" + name).read()
    manifest = json.loads(fixed("source-manifest.json"))
    assert json.loads(fixed("previous-manifest.json")) == previous
    assert manifest["base_commit"] == base and manifest["baseline"] == previous["baseline"]
    assert len(manifest["baseline"]) == 604
    for name, expected in manifest["overlays"].items():
        if not name.startswith("docs/plans/"):
            assert digest(source(name)) == expected, name
    assert b"rebuilt binary contains bridge marker" in fixed("rebuilt-binary.txt")
    assert b"Compiling beam-python" in fixed("rust-build.log")
    status = fixed("status.txt")
    assert status.endswith(b"RUST_WORKSPACE\nMEMCHECK\nEXIT_CODE=97\n")
    for name, markers in (
        ("bridge-1.log", (b"CNOT_DEVICE_UNSEEN_OK rank=0 world=1",)),
        ("bridge-2.log", (b"CNOT_DEVICE_UNSEEN_OK rank=0 world=2",
                          b"CNOT_DEVICE_UNSEEN_OK rank=1 world=2")),
        ("existing-2.log", (b"CNOT_RUST_EQUIVALENCE_OK rank=0 world=2",
                            b"CNOT_RUST_EQUIVALENCE_OK rank=1 world=2")),
        ("checkpoint-2.log", (b"CNOT_RUST_CHECKPOINT_OK rank=0 world=2",
                              b"CNOT_RUST_CHECKPOINT_OK rank=1 world=2")),
    ):
        assert all(marker in fixed(name) for marker in markers), name
    assert b"test result: FAILED" not in fixed("rust-workspace.log")
    assert b"test result: ok." in fixed("rust-workspace.log")
    assert not fixed("rust-format.log").strip()
    api_error = fixed("memcheck.log")
    assert b"CUDA_ERROR_INVALID_HANDLE" in api_error and b"cuKernelGetFunction" in api_error
    assert b"beam::cnot::launch_owners" in api_error
    assert b"CNOT_DEVICE_UNSEEN_OK rank=0 world=1" in api_error
    assert b"ERROR SUMMARY: 1 error" in api_error
    records = [json.loads(fixed(f"receipts/bridge-2/rank-{rank}.json")) for rank in (0, 1)]
    for rank, record in enumerate(records):
        assert record["rank"] == rank and record["world"] == 2
        assert record["gpu"] == "NVIDIA GeForce RTX 3060"
        assert record["reused_after_error"] and "rank 0" in record["coordinated_error"]
        assert [(case["wires"], case["mode"]) for case in record["cases"]] == [
            (4, "qhat"), (4, "proposal"), (4, "qhat"), (144, "qhat")]
    assert records[0]["cases"] == records[1]["cases"]
    checkpoints = [json.loads(fixed(f"receipts/checkpoint-2/rank-{rank}.json")) for rank in (0, 1)]
    assert checkpoints[0]["checkpoint_sha256"] == checkpoints[1]["checkpoint_sha256"]
    assert checkpoints[0]["results"] == checkpoints[1]["results"]
    private = json.loads(fixed("private-manifest.json"))
    assert private["source_commit"] == "c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb"
    assert private["fixture_sha256"] == "d6df7487ac5ba6cfc274b87e59a7b38ff334d0cd4cb8ae5d022dd8bce1e659d6"

with evidence("cnot-unseen-bridge-sanitizer-evidence.tar.gz",
              "498c9bd68f58a5843e11d369b1e447afdb12a16670a02edc130f7b3f44e83993") as tar:
    def sanitized(name):
        return tar.extractfile("cnot-unseen-bridge-sanitizer-results/" + name).read()
    assert json.loads(sanitized("source-manifest.json")) == manifest
    assert sanitized("source-before.txt") == sanitized("source-after.txt")
    assert b"COMPLETE\nEXIT_CODE=0\n" in sanitized("status.txt")
    assert sanitized("owner-baseline.exit").strip() == b"97"
    baseline_log = sanitized("owner-baseline.log")
    assert b"CUDA_ERROR_INVALID_HANDLE" in baseline_log and b"cuKernelGetFunction" in baseline_log
    assert b"beam::cnot::launch_owners" in baseline_log and b"OWNERS [1]" in baseline_log
    assert b"ERROR SUMMARY: 1 error" in baseline_log
    memory_log = sanitized("memcheck-no-api-errors.log")
    assert b"CNOT_DEVICE_UNSEEN_OK rank=0 world=1" in memory_log
    assert b"ERROR SUMMARY: 0 errors" in memory_log
    assert b"--report-api-errors no" in sanitized("cnot-unseen-bridge-sanitizer.sh")

# Compare the entire pinned base manifest against Git objects, not the mutable
# checkout. The overlay digests above account for the files intentionally edited.
process = subprocess.Popen(["git", "archive", "--format=tar", base], cwd=root, stdout=subprocess.PIPE)
seen = set()
with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
    for member in tar:
        if member.name in manifest["baseline"]:
            assert member.isfile() and digest(tar.extractfile(member).read()) == manifest["baseline"][member.name], member.name
            seen.add(member.name)
assert process.wait() == 0 and seen == set(manifest["baseline"])

release = json.loads((here.parent / "cnot-borrowed-unseen/release.json").read_text())
assert release["instance"] == 61005582 and release["provider_absence_verified"]
print("verified source-pinned CNOT search bridge, two-rank comparison and sanitizer limits")
