"""Exact score/word cutoff comparison for the Rust radix controller."""
from datetime import timedelta
import os
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch import rust_control
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_rust_selection import SelectionStages
from multigpubeamsearch.cnot_selection import global_winners
from multigpubeamsearch.errors import NativeUnavailable


def require_selection():
    try:
        runner = getattr(rust_control.extension(), "run_cnot_selection", None)
    except NativeUnavailable:
        runner = None
    if runner is None:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            raise AssertionError("matching Rust CNOT selection controller is required")
        pytest.skip("matching Rust CNOT selection controller is unavailable")
    return runner


def require_native_selection():
    try:
        runner = getattr(rust_control.extension(), "run_cnot_native_selection", None)
    except NativeUnavailable:
        runner = None
    if runner is None:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            raise AssertionError("matching Rust-owned CNOT selection stages are required")
        pytest.skip("matching Rust-owned CNOT selection stages are unavailable")
    return runner


def compare(world, scores, words, width):
    old = global_winners(world, scores, words, width)
    new = require_selection()(SelectionStages(world, scores, words), width)
    assert all(torch.equal(a, b) for a, b in zip(old, new))


def compare_native(world, scores, words, width):
    old = global_winners(world, scores, words, width)
    new = require_native_selection()(world, scores, words, width)
    assert all(torch.equal(a, b) for a, b in zip(old, new))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))])
def test_extremes_signed_zero_ties_and_all_widths(device):
    require_selection()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    world = World(device)
    scores = torch.tensor([1e-280, 0., -0., 1.00001, 1.00002, -1e300, -2., 1e300, -0.],
                          dtype=torch.float64, device=device)
    words = torch.tensor([19, 10, 11, 1, 2, 30, 200, 7, 3], device=device)
    for width in range(1, len(words) + 3):
        compare(world, scores, words, width)
    for width in range(1, 8):
        compare(world, torch.zeros(7, dtype=torch.float32, device=device),
                torch.tensor([8, 7, 6, 5, 4, 3, 2], device=device), width)
    compare(world, torch.empty(0, device=device), torch.empty(0, dtype=torch.int64, device=device), 1)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda:0", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"))])
def test_native_extremes_signed_zero_ties_and_all_widths(device):
    require_native_selection()
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    world = World(device)
    scores = torch.tensor([1e-280, 0., -0., 1.00001, 1.00002, -1e300, -2., 1e300, -0.],
                          dtype=torch.float64, device=device)
    words = torch.tensor([19, 10, 11, 1, 2, 30, 200, 7, 3], device=device)
    for width in range(1, len(words) + 3):
        compare_native(world, scores, words, width)
    for width in range(1, 8):
        compare_native(world, torch.zeros(7, dtype=torch.float32, device=device),
                       torch.tensor([8, 7, 6, 5, 4, 3, 2], device=device), width)
    compare_native(world, torch.empty(0, device=device), torch.empty(0, dtype=torch.int64, device=device), 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("missing", ["selection", "radix", "both"])
def test_native_selection_accepts_older_cuda_operators(monkeypatch, missing):
    from multigpubeamsearch import cnot_native

    require_native_selection()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    world = World(device)
    scores = torch.tensor([0., -0., -2., 3., 3., -1e300, 1e300], dtype=torch.float64, device=device)
    keys = torch.tensor([9, 2, 5, 1, 8, 6, 7], dtype=torch.int64, device=device)
    ops = cnot_native._cuda_ops()
    hidden = set()
    if missing in ("selection", "both"):
        hidden.update(("selection_prepare", "selection_records", "selection_order"))
    if missing in ("radix", "both"):
        hidden.update(("radix_histogram", "radix_update"))

    class OlderLibrary:
        def __getattr__(self, name):
            if name in hidden:
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    for width in (1, 2, 5, 7, 9):
        compare_native(world, scores, keys, width)


def distributed(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        world = World("cpu")
        for values, words in (
            (([0., -2., 1e300, -0.], [1e-280, -1e300, 1.00001, 0.]),
             ([11, 10, 15, 9], [8, 3, 4, 5])),
            (([], [0., 0., 0.]), ([], [30, 10, 20])),
        ):
            scores = torch.tensor(values[rank], dtype=torch.float64)
            keys = torch.tensor(words[rank], dtype=torch.int64)
            for width in range(1, sum(map(len, values)) + 2):
                compare(world, scores, keys, width)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available() or sys.platform == "win32",
                    reason="two CPU ranks require Gloo")
def test_two_rank_cutoff_including_empty_owner(tmp_path):
    require_selection()
    mp.spawn(distributed, args=((tmp_path / "cnot-selection-gloo").resolve().as_uri(),), nprocs=2)


def distributed_native(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        world = World("cpu")
        for values, words in (
            (([0., -2., 1e300, -0.], [1e-280, -1e300, 1.00001, 0.]),
             ([11, 10, 15, 9], [8, 3, 4, 5])),
            (([], [0., 0., 0.]), ([], [30, 10, 20])),
        ):
            scores = torch.tensor(values[rank], dtype=torch.float64)
            keys = torch.tensor(words[rank], dtype=torch.int64)
            for width in range(1, sum(map(len, values)) + 2):
                compare_native(world, scores, keys, width)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available() or sys.platform == "win32",
                    reason="two CPU ranks require Gloo")
def test_native_two_rank_cutoff_including_empty_owner(tmp_path):
    require_native_selection()
    mp.spawn(distributed_native, args=((tmp_path / "cnot-native-selection-gloo").resolve().as_uri(),), nprocs=2)
