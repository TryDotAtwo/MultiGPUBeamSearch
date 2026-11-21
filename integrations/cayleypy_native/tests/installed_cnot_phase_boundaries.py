"""Inject rank-local failures inside real Rust CNOT phases over CPU Gloo."""
from datetime import timedelta
import json
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.cnot_collectives import World


class FlatScorer:
    def child_values(self, children):
        return torch.zeros((len(children),), dtype=torch.float64, device=children.device)


def public_search():
    root = pack_rows(replay_cnot(3, (0,)))
    return search_cnot(
        root,
        FlatScorer(),
        width=6,
        shortlist=6,
        max_depth=3,
        validation="python",
        engine="rust",
    )


def inject_winner_packet_failure(rank, fail_rank):
    original_cat = torch.cat
    state = {"fired": False}

    def injected_cat(tensors, *args, **kwargs):
        if (
            rank == fail_rank
            and not state["fired"]
            and len(tensors) == 2
            and tuple(tensors[0].shape[1:]) == (5,)
            and tuple(tensors[1].shape[1:]) == (3,)
        ):
            state["fired"] = True
            raise RuntimeError("injected winner packet construction failure")
        return original_cat(tensors, *args, **kwargs)

    torch.cat = injected_cat
    return original_cat, state


def inject_history_edge_failure(rank, fail_rank):
    original_longs = World.longs
    state = {"fired": False}

    def injected_longs(world, items):
        if rank == fail_rank and not state["fired"] and list(items) == [0, 0, 0]:
            state["fired"] = True
            raise RuntimeError("injected history edge allocation failure")
        return original_longs(world, items)

    World.longs = injected_longs
    return original_longs, state


def worker(rank, rendezvous, output_dir, phase, fail_rank):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    original, injection = (
        inject_winner_packet_failure(rank, fail_rank)
        if phase == "winner packet"
        else inject_history_edge_failure(rank, fail_rank)
    )
    try:
        with World("cpu"):
            try:
                public_search()
            except RuntimeError as error:
                collective_error = str(error)
            else:
                raise AssertionError(f"injected {phase} failure did not abort the search")
            if rank == fail_rank and not injection["fired"]:
                raise AssertionError(f"the {phase} failure was not injected in its Rust phase")

            # The failed stage must vote before its next data-plane collective.
            # A second public search proves the process group remains usable.
            result = public_search()
            if result.status != "solved" or result.gates != (0,):
                raise AssertionError(f"post-fault public search failed: {result!r}")
            report = {
                "rank": rank,
                "phase": phase,
                "fault_rank": fail_rank,
                "collective_error": collective_error,
                "injection_fired": injection["fired"],
                "recovery_search": "passed",
                "search_status": result.status,
                "gates": list(result.gates),
            }
            Path(output_dir, f"rank-{rank}.json").write_text(
                json.dumps(report, sort_keys=True), encoding="utf-8"
            )
    finally:
        if phase == "winner packet":
            torch.cat = original
        else:
            World.longs = original
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory(prefix="mgbs-cnot-phases-") as temporary:
        root = Path(temporary)
        for phase, fail_rank in (
            ("winner packet", 0),
            ("winner packet", 1),
            ("history edge", 0),
            ("history edge", 1),
        ):
            phase_dir = root / f"{phase.replace(' ', '-')}-rank-{fail_rank}"
            phase_dir.mkdir()
            rendezvous = (phase_dir / "rendezvous").resolve().as_uri()
            mp.spawn(
                worker,
                args=(rendezvous, str(phase_dir), phase, fail_rank),
                nprocs=2,
                join=True,
            )
            reports = [
                json.loads((phase_dir / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            if reports[0]["collective_error"] != reports[1]["collective_error"]:
                raise AssertionError(f"ranks disagreed on {phase} failure: {reports!r}")
            expected = (
                f"prepare CNOT winner packet failed: rank {fail_rank}: RuntimeError: injected winner packet"
                if phase == "winner packet"
                else f"prepare CNOT history edge failed: rank {fail_rank}: RuntimeError: injected history edge"
            )
            if expected not in reports[0]["collective_error"]:
                raise AssertionError(f"wrong error phase or owner: {reports!r}")
            if any(not row["injection_fired"] and row["rank"] == fail_rank for row in reports):
                raise AssertionError(f"missing one-rank injection: {reports!r}")
            if any(row["recovery_search"] != "passed" or row["gates"] != [0] for row in reports):
                raise AssertionError(f"the live process group did not recover: {reports!r}")

    print("CNOT_GLOO_REAL_PHASE_FAILURES_OK")


if __name__ == "__main__":
    main()
