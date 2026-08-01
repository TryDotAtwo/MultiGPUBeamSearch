from pathlib import Path

import pytest

from tools.collect_elf_runtime_closure import parse_ldd


ROOT = Path(__file__).resolve().parents[2]


def test_workflow_stages_existing_repository_assets():
    text = (ROOT / ".github/workflows/megaminx-native-release.yml").read_text()
    assert "cp data/test.csv" in text
    assert "cp data/puzzle_info.json" in text
    assert "cp -a stream1_weights/." in text
    assert (ROOT / "tools/collect_elf_runtime_closure.py").is_file()


def test_ldd_parser_returns_absolute_dependencies_and_ignores_vdso():
    text = """linux-vdso.so.1 (0x0)\nlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x1)\n/lib64/ld-linux-x86-64.so.2 (0x2)\n"""
    assert parse_ldd(text) == (Path("/lib/x86_64-linux-gnu/libc.so.6"), Path("/lib64/ld-linux-x86-64.so.2"))


def test_ldd_parser_fails_closed_on_missing_library():
    with pytest.raises(ValueError, match="not found"):
        parse_ldd("libnccl.so.2 => not found\n")
