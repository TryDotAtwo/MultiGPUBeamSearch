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


def compare(world, scores, words, width):
    old = global_winners(world, scores, words, width)
    new = require_selection()(SelectionStages(world, scores, words), width)
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
