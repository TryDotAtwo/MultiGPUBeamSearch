"""Boundary and ambiguity checks exercise the real C++ registry."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_registered_pointer_bounds_and_failed_registration(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    source = Path(__file__).with_name("graph_pointer_registry_tests.cpp")
    executable = tmp_path / "graph_pointer_registry_tests"
    subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror",
                    str(source), "-o", str(executable)], check=True, timeout=60)
    subprocess.run([str(executable)], check=True, timeout=10)
