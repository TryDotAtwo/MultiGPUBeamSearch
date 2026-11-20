"""Rust owns the versioned Stream 5 communicator and its library lifetime."""
from datetime import timedelta
import gc
import os
from pathlib import Path
import shutil
import subprocess

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_stream5 import RustStream5
from multigpubeamsearch.rust_control import extension


MOCK = r'''
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef TEST_OWNED_ABI
#define TEST_OWNED_ABI 1
#endif
#ifndef OMIT_OWNED
#define OMIT_OWNED 0
#endif

static void event(const char *name, int a, int b) {
    const char *path = getenv("CNOT_S5_TEST_LOG");
    if (path) { FILE *f = fopen(path, "a"); if (f) { fprintf(f, "%s %d %d\n", name, a, b); fclose(f); } }
}
int cnot_s5_abi(void) { return TEST_ABI; }
const char *cnot_s5_error(void) { return "injected NCCL failure"; }
int cnot_s5_id_bytes(void) { return 16; }
int cnot_s5_id(void *dst) {
    if (getenv("CNOT_S5_FAIL_ID")) return 1;
    for (int i = 0; i < 16; ++i) ((unsigned char *)dst)[i] = (unsigned char)i;
    event("id", 0, 0); return 0;
}
int cnot_s5_init(void **comm, const void *id, int rank, int world) {
    (void)id; event("init", rank, world);
    if (getenv("CNOT_S5_FAIL_INIT")) return 1;
    *comm = (void *)(uintptr_t)1; return 0;
}
int cnot_s5_close(void *comm, int abort) {
    if (!comm) return 1;
    event("close", abort, 0); return getenv("CNOT_S5_FAIL_CLOSE") ? 1 : 0;
}
#if !OMIT_OWNED
int cnot_s5_owned_abi(void) { return TEST_OWNED_ABI; }
int cnot_s5_owned_init(void **comm, const void *id, int rank, int world) {
    event("owned_init", rank, world);
    return cnot_s5_init(comm, id, rank, world);
}
int cnot_s5_owned_close(void *comm, int abort) {
    event("owned_close", abort, 0);
    return cnot_s5_close(comm, abort);
}
#endif
int cnot_s5_counts(void *comm, void *stream, int rank, int world, void *send, void *recv) {
    (void)comm; (void)stream; (void)rank; (void)world; (void)send; (void)recv; return 0;
}
int cnot_s5_rows(void *comm, void *stream, int rank, int world, void *send, void *recv,
                 const uint32_t *sc, const uint32_t *rc, uint32_t stride) {
    (void)comm; (void)stream; (void)rank; (void)world; (void)send; (void)recv;
    (void)sc; (void)rc; (void)stride; return 0;
}
'''


@pytest.fixture
def mock_library(tmp_path, monkeypatch):
    if os.name != "posix" or shutil.which("cc") is None:
        pytest.skip("mock native library requires a Unix C compiler")
    source, output, log = tmp_path / "mock.c", tmp_path / "mock.so", tmp_path / "calls.log"
    source.write_text(MOCK, encoding="utf-8")
    subprocess.run(["cc", "-shared", "-fPIC", "-DTEST_ABI=1", str(source), "-o", str(output)], check=True)
    monkeypatch.setenv("CNOT_S5_TEST_LOG", str(log))
    return output, log


def test_rust_owns_communicator_and_releases_it_once(mock_library):
    path, log = mock_library
    cls = getattr(extension(), "CnotStream5Library", None)
    assert cls is not None, "Rust Stream 5 ownership is unavailable"
    library = cls(str(path))
    unique_id = library.unique_id()
    assert unique_id == bytes(range(16))
    comm = library.init(unique_id, 1, 2)
    del library
    gc.collect()
    assert not comm.closed
    comm.close(False)
    assert comm.closed
    comm.close(False)
    del comm
    gc.collect()
    assert log.read_text().splitlines() == ["id 0 0", "owned_init 1 2", "init 1 2",
                                            "owned_close 0 0", "close 0 0"]


def test_rust_uses_device_bound_communicator_abi(mock_library):
    path, log = mock_library
    library = extension().CnotStream5Library(str(path))
    communicator = library.init(library.unique_id(), 1, 2)
    communicator.close(False)
    events = log.read_text().splitlines()
    assert "owned_init 1 2" in events, "device-bound initialization was bypassed"
    assert "owned_close 0 0" in events, "device-bound close was bypassed"


def test_rust_exchange_rejects_cpu_buffers_before_native_access(mock_library):
    path, log = mock_library
    library = extension().CnotStream5Library(str(path))
    communicator = library.init(library.unique_id(), 0, 2)
    with pytest.raises((ValueError, RuntimeError), match="CUDA"):
        communicator.exchange(torch.zeros((1, 2), dtype=torch.int64),
                              torch.tensor([1, 0]), object())
    communicator.close(False)
    assert not any(row.startswith(("counts ", "rows ")) for row in log.read_text().splitlines())


def test_rust_aborts_unclosed_communicator(mock_library):
    path, log = mock_library
    cls = getattr(extension(), "CnotStream5Library", None)
    assert cls is not None, "Rust Stream 5 ownership is unavailable"
    library = cls(str(path))
    comm = library.init(library.unique_id(), 0, 2)
    del comm
    gc.collect()
    assert log.read_text().splitlines()[-1] == "close 1 0"


def test_rust_rejects_bad_identity_or_rank_before_native_init(mock_library):
    path, log = mock_library
    cls = getattr(extension(), "CnotStream5Library", None)
    assert cls is not None, "Rust Stream 5 ownership is unavailable"
    library = cls(str(path))
    identity = library.unique_id()
    for args in ((b"too short", 0, 2), (identity, 2, 2), (identity, 0, 0)):
        with pytest.raises((RuntimeError, ValueError)):
            library.init(*args)
    assert log.read_text().splitlines() == ["id 0 0"]


@pytest.mark.parametrize(("abi", "owned_abi", "omit_owned", "message"), [
    (2, 1, 0, "ABI mismatch"),
    (1, 2, 0, "owned ABI mismatch"),
    (1, 1, 1, "owned ABI unavailable"),
])
def test_rust_rejects_incompatible_abi(tmp_path, abi, owned_abi, omit_owned, message):
    if os.name != "posix" or shutil.which("cc") is None:
        pytest.skip("mock native library requires a Unix C compiler")
    source, output = tmp_path / "mock.c", tmp_path / "mock.so"
    source.write_text(MOCK, encoding="utf-8")
    subprocess.run(["cc", "-shared", "-fPIC", f"-DTEST_ABI={abi}",
                    f"-DTEST_OWNED_ABI={owned_abi}", f"-DOMIT_OWNED={omit_owned}",
                    str(source), "-o", str(output)], check=True)
    with pytest.raises(RuntimeError, match=message):
        extension().CnotStream5Library(str(output))


def test_rust_reports_native_errors_without_double_close(mock_library, monkeypatch):
    path, log = mock_library
    library = extension().CnotStream5Library(str(path))
    monkeypatch.setenv("CNOT_S5_FAIL_ID", "1")
    with pytest.raises(RuntimeError, match="injected NCCL failure"):
        library.unique_id()
    monkeypatch.delenv("CNOT_S5_FAIL_ID")
    identity = library.unique_id()
    monkeypatch.setenv("CNOT_S5_FAIL_INIT", "1")
    with pytest.raises(RuntimeError, match="injected NCCL failure"):
        library.init(identity, 0, 2)
    monkeypatch.delenv("CNOT_S5_FAIL_INIT")
    comm = library.init(identity, 0, 2)
    monkeypatch.setenv("CNOT_S5_FAIL_CLOSE", "1")
    with pytest.raises(RuntimeError, match="injected NCCL failure"):
        comm.close(False)
    assert comm.closed
    del comm
    gc.collect()
    assert log.read_text().splitlines().count("close 0 0") == 1


def _fixed_gather_preflight_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    original_new_empty = torch.Tensor.new_empty
    message = "fixture Stream 5 output allocation failed on rank one"
    calls = []
    try:
        with World("cpu") as world:
            class Communicator:
                def validate_all_gather_i64_into(self, tensor, work, output):
                    calls.append("validate")

                def all_gather_i64_into(self, tensor, work, output):
                    calls.append("all_gather")
                    return output

            transport = RustStream5.__new__(RustStream5)
            transport.world = world
            transport.comm = Communicator()
            tensor = torch.tensor([rank + 1], dtype=torch.int64)

            if rank == 1:
                def fail_output_allocation(self, size, *args, **kwargs):
                    if tuple(size) == (2, 1):
                        raise RuntimeError(message)
                    return original_new_empty(self, size, *args, **kwargs)
                torch.Tensor.new_empty = fail_output_allocation

            with pytest.raises(RuntimeError) as error:
                transport.gather_fixed(tensor)
            expected = (
                "prepare Rust Stream 5 fixed gather failed: "
                f"rank 1: RuntimeError: {message}"
            )
            assert str(error.value) == expected
            assert calls == [], "a rank entered validation/NCCL after a peer allocation failure"

            probe = torch.tensor([rank + 1], dtype=torch.int64)
            dist.all_reduce(probe)
            assert probe.item() == 3
    finally:
        torch.Tensor.new_empty = original_new_empty
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_preflight_stream5_gather_allocation_and_keep_world(tmp_path):
    mp.spawn(
        _fixed_gather_preflight_worker,
        args=((tmp_path / "stream5-gather-preflight").resolve().as_uri(),),
        nprocs=2,
    )
