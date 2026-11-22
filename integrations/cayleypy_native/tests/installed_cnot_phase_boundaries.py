"""Inject rank-local failures inside real Rust CNOT phases over CPU Gloo."""
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
import multigpubeamsearch.cnot as cnot_module
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.rust_control import extension


RUNTIME = extension()


class FlatScorer:
    def child_values(self, children):
        return torch.zeros((len(children),), dtype=torch.float64, device=children.device)


def public_search(validation="python", force_compatibility_stages=False):
    root = pack_rows(replay_cnot(3, (0, 1)))
    if not force_compatibility_stages:
        return search_cnot(
            root,
            FlatScorer(),
            width=6,
            shortlist=6,
            max_depth=3,
            validation=validation,
            engine="rust",
        )

    # Exercise the retained Rust-controller/Python-stage compatibility path.
    original = cnot_module._native_stages_runner
    cnot_module._native_stages_runner = lambda: None
    try:
        return search_cnot(
            root,
            FlatScorer(),
            width=6,
            shortlist=6,
            max_depth=3,
            validation=validation,
            engine="rust",
        )
    finally:
        cnot_module._native_stages_runner = original


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
            compatibility = phase.startswith("compatibility ")
            validation = "auto" if phase.endswith("typed input") else "python"
            try:
                public_search(validation, force_compatibility_stages=compatibility)
            except RuntimeError as error:
                collective_error = str(error)
            else:
                raise AssertionError(f"injected {phase} failure did not abort the search")
            if rank == fail_rank and not injection["fired"]:
                raise AssertionError(f"the {phase} failure was not injected in its Rust phase")

            # The failed stage must vote before its next data-plane collective.
            # A second public search proves the process group remains usable.
            result = public_search(validation, force_compatibility_stages=compatibility)
            if (
                result.status != "solved"
                or replay_cnot(3, result.gates) != replay_cnot(3, (0, 1))
            ):
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



def admitted_backend_mode_worker(rank, rendezvous, output_dir):
    """Keep the rank-agreed compatibility mode if process env changes afterward."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    old_value = os.environ.pop("MGBS_CNOT_STRICT_BACKEND", None)
    original_gather_fixed = World.gather_fixed
    injection = {"fired": False}

    def flip_after_admission(world, tensor):
        gathered = original_gather_fixed(world, tensor)
        if rank == 0 and not injection["fired"]:
            # The Rust controller has completed backend admission and the
            # input digest vote; Stream 5 startup follows this call.
            os.environ["MGBS_CNOT_STRICT_BACKEND"] = "1"
            injection["fired"] = True
        return gathered

    World.gather_fixed = flip_after_admission
    try:
        with World("cpu"):
            result = public_search()
            if not injection["fired"]:
                raise AssertionError("strict mode was not changed after backend admission")
            if (
                result.status != "solved"
                or replay_cnot(3, result.gates) != replay_cnot(3, (0, 1))
            ):
                raise AssertionError(
                    f"admitted compatibility mode changed during search: {result!r}"
                )
            Path(output_dir, f"rank-{rank}.json").write_text(
                json.dumps(
                    {"rank": rank, "status": result.status, "gates": list(result.gates)},
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
    finally:
        World.gather_fixed = original_gather_fixed
        if old_value is None:
            os.environ.pop("MGBS_CNOT_STRICT_BACKEND", None)
        else:
            os.environ["MGBS_CNOT_STRICT_BACKEND"] = old_value
        dist.destroy_process_group()

def trace_boundary_worker(rank, rendezvous, output_dir, fail_rank):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        with World("cpu") as world:
            try:
                RUNTIME._probe_cnot_device_stage_vote(world, fail_rank)
            except RuntimeError as error:
                collective_error = str(error)
            else:
                raise AssertionError("injected pre-trace failure did not abort the stage")

            # The peer must not perform a second, nested vote for its local
            # dispatch trace after this stage has already voted its failure.
            world.require_engine_consensus(b"after-device-trace-stage-failure")
            result = public_search()
            if (
                result.status != "solved"
                or replay_cnot(3, result.gates) != replay_cnot(3, (0, 1))
            ):
                raise AssertionError(f"post-trace-fault public search failed: {result!r}")
            Path(output_dir, f"trace-rank-{rank}.json").write_text(
                json.dumps({
                    "rank": rank,
                    "fail_rank": fail_rank,
                    "collective_error": collective_error,
                    "recovery_search": "passed",
                    "gates": list(result.gates),
                }, sort_keys=True),
                encoding="utf-8",
            )
    finally:
        dist.destroy_process_group()



def ranked_native_entry_worker(rank, rendezvous, output_dir, fail_rank):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        root = pack_rows(replay_cnot(3, (0, 1)))
        identity = pack_rows(replay_cnot(3, ()))
        input_matrix = RUNTIME.prepare_cnot_input(root, 6, 6, 3, "qhat")
        admitted_rank = (rank + 1) % 2 if rank == fail_rank else rank
        with World("cpu") as world:
            try:
                RUNTIME.run_cnot_native(
                    world,
                    root,
                    identity,
                    input_matrix,
                    FlatScorer(),
                    "qhat",
                    False,
                    6,
                    6,
                    6,
                    3,
                    None,
                    admitted_topology=(admitted_rank, 2),
                )
            except RuntimeError as error:
                collective_error = str(error)
            else:
                raise AssertionError("mismatched admitted rank reached the CNOT search")

            result = public_search()
            if (
                result.status != "solved"
                or replay_cnot(3, result.gates) != replay_cnot(3, (0, 1))
            ):
                raise AssertionError(f"rank topology failure prevented recovery: {result!r}")

        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "rank": rank,
                    "collective_error": collective_error,
                    "recovery": "passed",
                    "status": result.status,
                    "gates": list(result.gates),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()

def main():
    with tempfile.TemporaryDirectory(prefix="mgbs-cnot-phases-") as temporary:
        root = Path(temporary)
        for phase, fail_rank in (
            ("winner packet", 0),
            ("winner packet", 1),
            ("history edge", 0),
            ("history edge", 1),
            ("compatibility history edge typed input", 0),
            ("compatibility history edge typed input", 1),
            ("compatibility history edge python input", 0),
            ("compatibility history edge python input", 1),
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
            if any(
                row["recovery_search"] != "passed"
                or row["search_status"] != "solved"
                or replay_cnot(3, tuple(row["gates"])) != replay_cnot(3, (0, 1))
                for row in reports
            ):
                raise AssertionError(f"the live process group did not recover: {reports!r}")

        mode_dir = root / "admitted-backend-mode"
        mode_dir.mkdir()
        mode_rendezvous = (mode_dir / "rendezvous").resolve().as_uri()
        mp.spawn(
            admitted_backend_mode_worker,
            args=(mode_rendezvous, str(mode_dir)),
            nprocs=2,
            join=True,
        )
        mode_reports = [
            json.loads((mode_dir / f"rank-{rank}.json").read_text())
            for rank in range(2)
        ]
        if any(
            row["status"] != "solved"
            or replay_cnot(3, tuple(row["gates"])) != replay_cnot(3, (0, 1))
            for row in mode_reports
        ):
            raise AssertionError(
                f"rank-agreed compatibility mode was not pinned: {mode_reports!r}"
            )

        for fail_rank in (0, 1):
            trace_dir = root / f"trace-stage-rank-{fail_rank}"
            trace_dir.mkdir()
            rendezvous = (trace_dir / "rendezvous").resolve().as_uri()
            mp.spawn(
                trace_boundary_worker,
                args=(rendezvous, str(trace_dir), fail_rank),
                nprocs=2,
                join=True,
            )
            reports = [
                json.loads((trace_dir / f"trace-rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            if reports[0]["collective_error"] != reports[1]["collective_error"]:
                raise AssertionError(f"ranks disagreed on DeviceDomain trace boundary: {reports!r}")
            expected = (
                f"rank parent actions and expand failed: rank {fail_rank}: RuntimeError: "
                "injected DeviceDomain stage failure before dispatch trace"
            )
            if expected not in reports[0]["collective_error"]:
                raise AssertionError(f"wrong pre-trace error phase or owner: {reports!r}")
            if any(
                row["recovery_search"] != "passed"
                or replay_cnot(3, tuple(row["gates"])) != replay_cnot(3, (0, 1))
                for row in reports
            ):
                raise AssertionError(f"trace boundary did not recover: {reports!r}")

        for fail_rank in (0, 1):
            binding_dir = root / f"ranked-native-entry-fail-{fail_rank}"
            binding_dir.mkdir()
            rendezvous = (binding_dir / "rendezvous").resolve().as_uri()
            mp.spawn(
                ranked_native_entry_worker,
                args=(rendezvous, str(binding_dir), fail_rank),
                nprocs=2,
                join=True,
            )
            binding_reports = [
                json.loads((binding_dir / f"rank-{rank}.json").read_text())
                for rank in range(2)
            ]
            if binding_reports[0]["collective_error"] != binding_reports[1]["collective_error"]:
                raise AssertionError(
                    f"ranks disagreed on admitted CNOT rank topology: {binding_reports!r}"
                )
            expected = (
                f"bind admitted CNOT rank topology failed: rank {fail_rank}: RuntimeError: "
                "admitted CNOT rank topology"
            )
            if expected not in binding_reports[0]["collective_error"]:
                raise AssertionError(
                    f"wrong admitted topology failure phase: {binding_reports!r}"
                )
            if any(
                row["recovery"] != "passed"
                or row["status"] != "solved"
                or replay_cnot(3, tuple(row["gates"])) != replay_cnot(3, (0, 1))
                for row in binding_reports
            ):
                raise AssertionError(f"rank topology vote did not recover: {binding_reports!r}")

    print("CNOT_GLOO_REAL_PHASE_AND_TRACE_BOUNDARIES_OK")


if __name__ == "__main__":
    main()
