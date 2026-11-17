"""Compare the opt-in Rust/device CNOT stage over real rank collectives."""
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
    library = os.environ["BEAM_CNOT_OWNED_LIBRARY"]
    key = "MGBS_CNOT_OWNED_LIBRARY"
    try:
        if os.environ.get("CNOT_DEVICE_UNSEEN_RED") == "1":
            os.environ[key] = library + ".does-not-exist"
            try:
                cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), WireFixture(9),
                                 width=5, shortlist=9, max_depth=4, engine="rust")
            except RuntimeError as error:
                assert "does-not-exist" in str(error)
                print("CNOT_DEVICE_UNSEEN_RED_OK", flush=True)
                return
            raise AssertionError("explicit invalid owned CNOT library was not rejected")
        cases = []
        wide_rows, _ = wide_case()
        for rows, undo, mode, width, k, depth in (
            ((8, 4, 2, 1), 9, "qhat", 5, 9, 4),
            ((8, 4, 2, 1), 9, "proposal", 5, 9, 4),
            ((8, 4, 2, 1), 9, "qhat", 5, 12, 4),
            (wide_rows, 428, "qhat", 1, 3, 3),
        ):
            root = cnot.pack_rows(rows, device=device)
            options = dict(width=width, shortlist=k, max_depth=depth, mode=mode, audit=True)
            os.environ.pop(key, None)
            reference = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
            os.environ[key] = library
            for _ in range(2):
                actual = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
                assert asdict(actual) == asdict(reference), (rank, len(rows), mode)
                if actual.gates is not None:
                    assert cnot.replay_cnot(len(rows), actual.gates) == rows
            if distributed:
                if rank:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = library
                mixed = cnot.search_cnot(root, WireFixture(undo), engine="rust", **options)
                assert asdict(mixed) == asdict(reference), (rank, "mixed", len(rows), mode)
            cases.append(dict(wires=len(rows), mode=mode, result=asdict(reference)))

        # Only one rank has an invalid library. The rank-wide stage check must
        # propagate the error to every rank without entering the next collective.
        if rank == 0:
            os.environ[key] = library + ".does-not-exist"
        else:
            os.environ.pop(key, None)
        try:
            cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), WireFixture(9),
                             width=5, shortlist=9, max_depth=4, engine="rust")
        except RuntimeError as error:
            message = str(error)
            assert "does-not-exist" in message, message
            if distributed:
                assert "rank 0" in message, message
        else:
            raise AssertionError("explicit invalid owned CNOT library was not rejected")
        os.environ[key] = library
        again = cnot.search_cnot(cnot.pack_rows((8, 4, 2, 1), device=device), WireFixture(9),
                                 width=5, shortlist=9, max_depth=4, engine="rust", audit=True)
        assert asdict(again) == cases[0]["result"]
        output = Path(os.environ["CNOT_RUST_EVIDENCE"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(dict(
            rank=rank, world=size, gpu=torch.cuda.get_device_name(device), cases=cases,
            coordinated_error=message, reused_after_error=True), indent=2) + "\n")
        print(f"CNOT_DEVICE_UNSEEN_OK rank={rank} world={size}", flush=True)
    finally:
        os.environ.pop(key, None)
        if distributed:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
