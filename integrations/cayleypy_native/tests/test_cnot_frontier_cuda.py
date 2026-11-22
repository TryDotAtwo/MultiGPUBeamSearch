"""Native winner packing must preserve every record and empty-rank case."""
import pytest
import torch

from multigpubeamsearch import cnot_native as native


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n", [4, 65])
@pytest.mark.parametrize("rank,count", [(0, 4), (1, 4), (2, 4), (0, 0)])
def test_frontier_packet_and_views_match_reference(n, rank, count):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    words = (n + 63) // 64
    children = torch.arange(4 * n * words, dtype=torch.int64, device=device).reshape(4, n, words)
    closed = children[:1].clone()
    meta = torch.arange(16, dtype=torch.int64, device=device).reshape(4, 4)
    ranks = torch.tensor([0, 1, 0, 1][:count], dtype=torch.int64, device=device)
    local = torch.tensor([3, 1, 0, 2][:count], dtype=torch.int64, device=device)
    lex = torch.tensor([2, 0, 3, 1][:count], dtype=torch.int64, device=device)
    slots = (ranks == rank).nonzero().flatten()
    kept = local[slots]
    rows = children[kept]
    expected_closed = torch.cat((closed, rows))
    records = torch.stack((slots, lex[slots], meta[kept, 0], meta[kept, 3], ranks[slots]), 1)
    expected_wire = torch.cat((records, rows.reshape(len(kept), n * words)), 1)
    expected_targets = slots * 3 // max(1, count)

    ops = native._cuda_ops()
    actual_closed, actual_wire, actual_targets = ops.winner_packet(
        ranks, local, lex, meta, children, closed, rank, 3)
    assert torch.equal(actual_closed, expected_closed)
    assert torch.equal(actual_wire, expected_wire)
    assert torch.equal(actual_targets, expected_targets)

    scrambled = actual_wire.flip(0)
    received = ops.sort_received(scrambled)
    expected_received = scrambled[torch.argsort(scrambled[:, 0])]
    assert torch.equal(received, expected_received)
    history, parents, new_rank = ops.frontier_views(received, n, words)
    assert torch.equal(history, expected_received[:, 2:5].contiguous())
    assert torch.equal(parents, expected_received[:, 5:].reshape(-1, n, words).contiguous())
    assert torch.equal(new_rank, expected_received[:, 1].contiguous())
