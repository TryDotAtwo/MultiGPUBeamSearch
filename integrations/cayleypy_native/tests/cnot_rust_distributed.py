"""Physical GPU/NCCL equivalence and coordinated-failure acceptance fixture."""
from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from verify_cnot import WireFixture, wide_case
from multigpubeamsearch import cnot
from multigpubeamsearch import cnot_stream5
from multigpubeamsearch import cnot_rust_stages
from multigpubeamsearch import cnot_rust_selection
from multigpubeamsearch import cnot_native
from multigpubeamsearch.cnot_collectives import World


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank = dist.get_rank() if distributed else 0
    size = dist.get_world_size() if distributed else 1
    try:
        legacy_stages = []
        original_stages_init = cnot_rust_stages.CnotStages.__init__

        def trace_stages(self, *args, **kwargs):
            legacy_stages.append(True)
            return original_stages_init(self, *args, **kwargs)

        cnot_rust_stages.CnotStages.__init__ = trace_stages
        legacy_selections = []
        original_selection_init = cnot_rust_selection.SelectionStages.__init__

        def trace_selection(self, *args, **kwargs):
            legacy_selections.append(True)
            return original_selection_init(self, *args, **kwargs)

        cnot_rust_selection.SelectionStages.__init__ = trace_selection
        rust_calls, legacy_calls = [], []
        if distributed:
            original_rust = cnot_stream5.RustStream5.exchange
            original_legacy = cnot_stream5.Stream5.exchange

            def trace_rust(self, send, counts):
                rust_calls.append(len(send))
                return original_rust(self, send, counts)

            def trace_legacy(self, send, counts):
                legacy_calls.append(len(send))
                return original_legacy(self, send, counts)

            cnot_stream5.RustStream5.exchange = trace_rust
            cnot_stream5.Stream5.exchange = trace_legacy
        cases = []
        wide_rows, _ = wide_case()
        for rows, undo, mode, width, k, depth in (
            ((8, 4, 2, 1), 9, "qhat", 5, 9, 4),
            ((8, 4, 2, 1), 9, "proposal", 5, 9, 4),
            ((8, 4, 2, 1), 9, "qhat", 5, 12, 4),
            (wide_rows, 428, "qhat", 1, 3, 3),
            (wide_rows, 428, "proposal", 1, 3, 3),
        ):
            root = cnot.pack_rows(rows, device=device)
            options = dict(width=width, shortlist=k, max_depth=depth, mode=mode, audit=True)
            old = cnot.search_cnot(root, WireFixture(undo), engine="python", **options)
            original_goal = cnot._goal
            original_checked = World.checked
            batch_helpers = {name: getattr(cnot, name) for name in ("_scores", "_expand", "_distinct")}

            def forbidden_goal(*args, **kwargs):
                raise AssertionError("legacy Python goal coordinator was called")

            def forbidden_batch(*args, **kwargs):
                raise AssertionError("legacy Python CNOT batch helper was called")

            def checked(world, stage, action):
                if stage in {"initialize CNOT domain", "rank parent actions and expand",
                             "deduplicate and evaluate distinct children"}:
                    raise AssertionError("legacy Python CNOT stage coordinator was called")
                return original_checked(world, stage, action)

            cnot._goal = forbidden_goal
            World.checked = checked
            for name in batch_helpers:
                setattr(cnot, name, forbidden_batch)
            try:
                new = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
            finally:
                cnot._goal = original_goal
                World.checked = original_checked
                for name, helper in batch_helpers.items():
                    setattr(cnot, name, helper)
            assert asdict(new) == asdict(old), (mode, len(rows), new, old)
            if new.gates is not None:
                assert cnot.replay_cnot(len(rows), new.gates) == rows
            if len(rows) == 144:
                assert new.gates == (18592, 428)
            cases.append(dict(wires=len(rows), mode=mode, options=options,
                              result=asdict(new), replay=new.gates is not None))

        assert not legacy_stages, "Rust search unexpectedly used the Python stage owner"
        assert not legacy_selections, "Rust search unexpectedly used the Python selection owner"
        errors = []
        fallback_legacy_calls = 0
        fallback_stage_calls = 0
        fallback_selection_calls = 0
        batch_fallbacks = []

        class OlderOps:
            def __init__(self, wrapped, missing):
                self.wrapped, self.missing = wrapped, missing

            def __getattr__(self, name):
                if name in self.missing:
                    raise AttributeError(name)
                return getattr(self.wrapped, name)

        original_ops = cnot_native._cuda_ops
        batch_helpers = {name: getattr(cnot, name) for name in ("_scores", "_expand", "_distinct")}

        def forbidden_batch(*args, **kwargs):
            raise AssertionError("legacy Python CNOT batch helper was called")

        try:
            for name in batch_helpers:
                setattr(cnot, name, forbidden_batch)
            for scenario in ("expand", "distinct", "both", "mixed"):
                missing = ({"expand_candidates"} if scenario == "expand" else
                           {"distinct_candidates"} if scenario == "distinct" else
                           {"expand_candidates", "distinct_candidates"} if scenario == "both" else
                           {"expand_candidates"} if rank == 0 else {"distinct_candidates"})
                cnot_native._cuda_ops = lambda: OlderOps(original_ops(), missing)
                fallback = cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), WireFixture(9),
                                            width=5, shortlist=9, max_depth=4, audit=True, engine="rust")
                assert asdict(fallback) == cases[0]["result"], (rank, scenario)
                batch_fallbacks.append(scenario)
        finally:
            cnot_native._cuda_ops = original_ops
            for name, helper in batch_helpers.items():
                setattr(cnot, name, helper)

        if distributed:
            assert rust_calls and legacy_calls, (rank, rust_calls, legacy_calls)
            original_native_backend = cnot._native_stages_runner
            cnot._native_stages_runner = lambda: None if rank == 1 else original_native_backend()
            fallback = cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), WireFixture(9),
                                        width=5, shortlist=9, max_depth=4, audit=True, engine="rust")
            assert asdict(fallback) == cases[0]["result"]
            fallback_stage_calls = len(legacy_stages)
            assert fallback_stage_calls > 0
            fallback_selection_calls = len(legacy_selections)
            assert fallback_selection_calls > 0
            cnot._native_stages_runner = original_native_backend

            original_compatible = cnot_stream5.RustStream5.compatible_library
            cnot_stream5.RustStream5.compatible_library = staticmethod(
                lambda lib: None if rank == 1 else original_compatible(lib))
            prior_legacy, prior_rust = len(legacy_calls), len(rust_calls)
            fallback = cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), WireFixture(9),
                                        width=5, shortlist=9, max_depth=4, audit=True, engine="rust")
            assert asdict(fallback) == cases[0]["result"]
            fallback_legacy_calls = len(legacy_calls) - prior_legacy
            assert fallback_legacy_calls > 0 and len(rust_calls) == prior_rust
            assert len(legacy_selections) == fallback_selection_calls
            cnot_stream5.RustStream5.compatible_library = staticmethod(original_compatible)

            class RejectOnOne(WireFixture):
                def child_values(self, children):
                    if rank == 1:
                        raise RuntimeError("injected CNOT scorer failure")
                    return super().child_values(children)

            try:
                cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), RejectOnOne(9),
                                 width=5, shortlist=9, max_depth=4, engine="rust")
            except RuntimeError as exc:
                assert "rank 1: RuntimeError: injected CNOT scorer failure" in str(exc)
                errors.append(str(exc))
            else:
                raise AssertionError("scorer failure did not reach every rank")
            recovered = cnot.search_cnot(cnot.pack_rows((1, 3), device=device), object(),
                                         width=1, shortlist=1, max_depth=1, engine="rust")
            assert recovered.gates == (0,)

        output = Path(os.environ["CNOT_RUST_EVIDENCE"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(dict(
            rank=rank, world=size, gpu=torch.cuda.get_device_name(device), cases=cases,
            errors=errors, reused_after_failure=bool(errors), rust_transport_calls=len(rust_calls),
            fallback_legacy_calls=fallback_legacy_calls, fallback_stage_calls=fallback_stage_calls,
            fallback_selection_calls=fallback_selection_calls, batch_fallbacks=batch_fallbacks),
            indent=2) + "\n", encoding="utf-8")
        print(f"CNOT_RUST_EQUIVALENCE_OK rank={rank} world={size}", flush=True)
    finally:
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
