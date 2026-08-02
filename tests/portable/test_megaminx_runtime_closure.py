from pathlib import Path

import pytest

from tools.collect_elf_runtime_closure import collect_closure, parse_ldd


ROOT = Path(__file__).resolve().parents[2]


def test_workflow_stages_existing_repository_assets():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert "cp data/test.csv" in text
    assert "cp data/puzzle_info.json" in text
    assert "cp -a stream1_weights/." in text
    assert (ROOT / "tools/collect_elf_runtime_closure.py").is_file()


def test_workflow_resolves_staged_runtime_sonames_before_packaging():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert 'LD_LIBRARY_PATH="$stage/lib" ldd "$stage/bin/production_runner"' in text
    assert 'grep -F "not found"' in text

def test_ldd_parser_returns_absolute_dependencies_and_ignores_vdso():
    text = """linux-vdso.so.1 (0x0)\nlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x1)\n/lib64/ld-linux-x86-64.so.2 (0x2)\n"""
    assert parse_ldd(text) == (Path("/lib/x86_64-linux-gnu/libc.so.6"), Path("/lib64/ld-linux-x86-64.so.2"))


def test_ldd_parser_fails_closed_on_missing_library():
    with pytest.raises(ValueError, match="not found"):
        parse_ldd("libnccl.so.2 => not found\n")

def test_collector_preserves_requested_soname_when_dependency_resolves_to_version(tmp_path, monkeypatch):
    runner = tmp_path / "runner"
    runner.write_bytes(b"ELF")
    source = tmp_path / "source"
    source.mkdir()
    soname = Path("/runtime/libnccl.so.2")
    versioned = source / "libnccl.so.2.28.3"
    versioned.write_bytes(b"NCCL")
    destination = tmp_path / "lib"
    original_resolve = Path.resolve

    def fake_resolve(path):
        if path == soname:
            return versioned
        return original_resolve(path)

    class Result:
        returncode = 0
        stderr = ""

        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, **_kwargs):
        binary = Path(command[-1])
        return Result("libnccl.so.2 => /runtime/libnccl.so.2 (0x1)\n" if binary.name == "runner" else "")

    monkeypatch.setattr(Path, "resolve", fake_resolve)
    monkeypatch.setattr("tools.collect_elf_runtime_closure.subprocess.run", fake_run)
    collect_closure(runner, destination)

    assert (destination / "libnccl.so.2").read_bytes() == b"NCCL"