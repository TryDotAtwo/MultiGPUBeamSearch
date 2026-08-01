import pytest

from tools.collect_elf_runtime_closure import dependency_policy


@pytest.mark.parametrize("name", ["libc.so.6", "ld-linux-x86-64.so.2", "libcuda.so.1", "libnvidia-ml.so.1"])
def test_system_and_driver_libraries_are_never_bundled(name):
    assert dependency_policy(name) == "host"


@pytest.mark.parametrize("name", ["libtorch.so", "libc10.so", "libnccl.so.2", "libcudart.so.12"])
def test_redistributable_runtime_libraries_are_bundled(name):
    assert dependency_policy(name) == "bundle"


def test_unknown_runtime_dependency_fails_closed():
    with pytest.raises(ValueError):
        dependency_policy("libmystery.so")
