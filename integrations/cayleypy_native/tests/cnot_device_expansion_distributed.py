"""Compare and trace the opt-in Rust CNOT device stages on real ranks."""
from collections import Counter
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


def main():
    torch.set_num_threads(1)
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    rank = dist.get_rank() if distributed else 0
    size = dist.get_world_size() if distributed else 1
    output = Path(os.environ["CNOT_RUST_EVIDENCE"])
    output.mkdir(parents=True, exist_ok=True)
    trace = output / f"trace-{rank}.txt"
    trace.write_text("")
    os.environ["BEAM_TEST_CNOT_TRACE"] = str(trace)
    library = os.environ["BEAM_CNOT_TRACE_LIBRARY"]
    key = "MGBS_CNOT_OWNED_LIBRARY"

    def trace_counts():
        return Counter(trace.read_text().splitlines())

    def assert_trace_delta(before, enabled, label):
        delta = trace_counts() - before
        if not enabled:
            assert not delta, f"rank {rank}: fallback path called traced native stages during {label}: {delta}"
            return
        # Each selected device route asks native code to assign the root owner
        # once, then once per nonempty generated-child batch. Exact deltas make
        # either missing call site visible even if the search falls back and
        # still returns the right answer.
        assert delta["owners"] == 1 + delta["children"], (rank, label, delta)

    try:
        cases = []
        wide_rows, _ = wide_case()
        for rows, undo, mode, width, shortlist, depth in (
            ((8, 4, 2, 1), 9, "qhat", 5, 9, 4),
            ((8, 4, 2, 1), 9, "proposal", 5, 9, 4),
            ((8, 4, 2, 1), 9, "qhat", 5, 12, 4),
            (wide_rows, 428, "qhat", 1, 3, 3),
        ):
            root = cnot.pack_rows(rows, device=device)
            options = dict(width=width, shortlist=shortlist, max_depth=depth, mode=mode, audit=True)
            os.environ.pop(key, None)
            reference = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
            os.environ[key] = library
            for _ in range(2):
                before = trace_counts()
                actual = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
                assert asdict(actual) == asdict(reference), (rank, len(rows), mode)
                assert_trace_delta(before, True, f"{len(rows)}-wire {mode} repeat")
                if actual.gates is not None:
                    assert cnot.replay_cnot(len(rows), actual.gates) == rows
            if distributed:
                if rank:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = library
                before = trace_counts()
                mixed = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
                assert asdict(mixed) == asdict(reference), (rank, "mixed", len(rows), mode)
                assert_trace_delta(before, rank == 0, f"{len(rows)}-wire {mode} mixed")
            cases.append(dict(wires=len(rows), mode=mode, result=asdict(reference)))

        counts = Counter(trace.read_text().splitlines())
        for stage in ("owners", "children", "unseen"):
            assert counts[stage] > 0, f"rank {rank}: selected CNOT stage {stage} was not called: {counts}"
        (output / f"rank-{rank}.json").write_text(json.dumps(dict(
            rank=rank, world=size, gpu=torch.cuda.get_device_name(device),
            counts=dict(counts), cases=cases), indent=2) + "\n")
        print(f"CNOT_DEVICE_EXPANSION_OK rank={rank} world={size}", flush=True)
    finally:
        os.environ.pop(key, None)
        os.environ.pop("BEAM_TEST_CNOT_TRACE", None)
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

