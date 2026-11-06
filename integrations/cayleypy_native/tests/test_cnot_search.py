from datetime import timedelta
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_selection import global_winners

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from verify_cnot import WireFixture, validate_case, verify_layers, wide_case


def test_identity_and_goal_outside_shortlist():
    rows = replay_cnot(144, (20591,))
    for input_rows, expected in ((replay_cnot(144, ()), ()), (rows, (20591,))):
        result = search_cnot(pack_rows(input_rows), object(), width=1, shortlist=1, max_depth=1)
        assert result.gates == expected and result.length == len(expected)
    result = search_cnot(pack_rows(rows), object(), width=1, shortlist=1, max_depth=0)
    assert result.status == "depth_limit" and result.gates is None


@pytest.mark.parametrize("mode,k", [("qhat", 7), ("proposal", 7), ("qhat", 12)])
def test_several_layers_against_exhaustive_single_beam(mode, k):
    root = (8, 4, 2, 1)
    scorer = WireFixture(9)
    result = search_cnot(pack_rows(root), scorer, width=5, shortlist=k, max_depth=4, mode=mode, audit=True)
    report = verify_layers(root, scorer, mode, 5, k, result.layers, torch.device("cpu"))
    assert len(report) >= 2 and any(layer["distinct"] > 5 for layer in report)


def test_radix_retains_exact_close_scores_extremes_and_lexicographic_ties():
    world = World("cpu")
    scores = torch.tensor([1e-280, 0., -0., 1.00001, 1.00002, -1e300, -2., 1e300, -0.],
                          dtype=torch.float64)
    words = torch.tensor([19, 10, 11, 1, 2, 30, 200, 7, 3])
    for width in range(1, len(words) + 3):
        owners, ids, new_lex = global_winners(world, scores, words, width)
        expected = sorted(range(len(scores)), key=lambda i: (scores[i].item(), words[i].item()))[:width]
        assert ids.tolist() == expected and owners.tolist() == [0] * len(expected)
        assert new_lex.tolist() == [sorted(words[ids].tolist()).index(words[i]) for i in ids]


def test_rejects_invalid_input_and_nonfinite_scorer():
    with pytest.raises(ValueError, match="singular"):
        search_cnot(pack_rows((1, 1)), object(), width=1, shortlist=1, max_depth=2)
    padding = torch.tensor([[[1], [6]]])
    with pytest.raises(ValueError, match="padding"):
        search_cnot(padding, object(), width=1, shortlist=1, max_depth=2)
    class Invalid(WireFixture):
        def child_values(self, children):
            return torch.full((len(children),), float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        search_cnot(pack_rows((8, 4, 2, 1)), Invalid(0), width=1, shortlist=2, max_depth=3)


def _distributed(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        rows, scorer = wide_case()
        for mode in ("proposal", "qhat"):
            report = validate_case(rows, scorer, mode, 1, 3, 3, torch.device("cpu"), True)
            if rank == 0:
                assert report["physical_ids"] == [18592, 428]
                assert report["routes"] == [dict(source=0, hash_owner=1, target=0, move=428)]
        root = (8, 4, 2, 1)
        result = search_cnot(pack_rows(root), scorer=WireFixture(9), width=5, shortlist=9,
                             max_depth=4, audit=True)
        verify_layers(root, WireFixture(9), "qhat", 5, 9, result.layers, torch.device("cpu"))
        assert result.layers[1]["by_rank"][1][0] == 2
        class RejectOnOne(WireFixture):
            def child_values(self, children):
                if rank == 1:
                    raise RuntimeError("uncovered feature")
                return super().child_values(children)
        with pytest.raises(RuntimeError, match="rank 1: RuntimeError: uncovered feature"):
            search_cnot(pack_rows(root), RejectOnOne(9), width=5, shortlist=9, max_depth=4)
        with pytest.raises(ValueError, match="differ"):
            search_cnot(pack_rows(root), WireFixture(9), width=5 + rank, shortlist=9, max_depth=4)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
def test_two_real_gloo_processes(tmp_path):
    if sys.platform == "win32":
        try:
            dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        except RuntimeError as exc:
            if "unsupported gloo device" not in str(exc):
                raise
            pytest.skip("this Windows torch wheel has no Gloo transport")
    mp.spawn(_distributed, args=((tmp_path / "cnot-gloo").resolve().as_uri(),), nprocs=2)
