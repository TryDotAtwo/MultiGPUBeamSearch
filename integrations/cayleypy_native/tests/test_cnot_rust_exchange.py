"""Rust CNOT record-exchange orchestration preserves the existing protocol."""
from datetime import timedelta
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.errors import NativeUnavailable
from multigpubeamsearch.rust_control import extension


def exchange_fn():
    try:
        function = getattr(extension(), "run_cnot_native_exchange", None)
    except NativeUnavailable:
        function = None
    if function is None:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            raise AssertionError("matching Rust CNOT record exchange is required")
        pytest.skip("matching Rust CNOT record exchange is unavailable")
    return function


def test_one_rank_returns_original_rows_without_inspecting_targets():
    exchange = exchange_fn()
    rows = torch.tensor([[4, 5]], dtype=torch.int64)
    assert exchange(World("cpu"), rows, None) is rows


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    try:
        exchange = exchange_fn()
        world = World("cpu")
        with world:
            cases = [
                (([[10, 11], [20, 21], [22, 23]], [1, 0, 1]),
                 ([[30, 31]], [0])),
                (([], []), ([[40, 41]], [0])),
                (([], []), ([], [])),
                (([[50, 51]], [0]), ([[60, 61]], [1])),
            ]
            expected_rows = [
                ([[20, 21], [30, 31]], [[10, 11], [22, 23]]),
                ([[40, 41]], []), ([], []), ([[50, 51]], [[60, 61]]),
            ]
            for index, case in enumerate(cases):
                values, destinations = case[rank]
                rows = torch.tensor(values, dtype=torch.int64).reshape(-1, 2)
                targets = torch.tensor(destinations, dtype=torch.int64)
                reference = world.exchange(rows, targets, native_packet=True)
                result = exchange(world, rows, targets)
                assert torch.equal(result, reference)
                assert result.tolist() == expected_rows[index][rank]
            # The same process group is still usable after the all-empty exchange.
            value = torch.tensor([rank], dtype=torch.int64)
            dist.all_reduce(value)
            assert value.item() == 1

    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_gloo_ranks_stable_records_empty_splits_and_reuse(tmp_path):
    exchange_fn()
    mp.spawn(_worker, args=((tmp_path / "exchange").resolve().as_uri(),), nprocs=2)
