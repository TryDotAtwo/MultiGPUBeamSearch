"""Exercise the real PyO3 validator through public CNOT search on CPU/Gloo."""
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


spec = importlib.util.spec_from_file_location("_beam_runtime", sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)
sys.modules["_beam_runtime"] = runtime
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.rust_control import extension
from verify_cnot import WireFixture, wide_case


assert extension() is runtime
torch.set_num_threads(1)
distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
cuda = "--cuda" in sys.argv[2:]
if cuda:
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
if distributed:
    dist.init_process_group("nccl" if cuda else "gloo")
rank = dist.get_rank() if distributed else 0
records = []
try:
    wide_rows, wide_scorer = wide_case()
    for rows, scorer, width, shortlist, depth, mode in (
        (replay_cnot(144, (20591,)), object(), 1, 1, 1, "qhat"),
        (wide_rows, wide_scorer, 1, 3, 3, "qhat"),
        ((8, 4, 2, 1), WireFixture(9), 5, 9, 4, "qhat"),
        ((8, 4, 2, 1), WireFixture(9), 5, 7, 4, "proposal"),
    ):
        options = dict(width=width, shortlist=shortlist, max_depth=depth, mode=mode, audit=True)
        reference = search_cnot(pack_rows(rows, device=device), scorer, validation="python", **options)
        actual = search_cnot(pack_rows(rows, device=device), scorer, validation="rust", **options)
        assert asdict(actual) == asdict(reference)
        if actual.gates is not None:
            assert replay_cnot(len(rows), actual.gates) == rows
        if rows == wide_rows:
            assert actual.gates == (18592, 428)
            if distributed:
                assert any(edge['move'] > 255 and
                           len({edge['source'], edge['hash_owner'], edge['target']}) > 1
                           for edge in actual.path_routes)
        records.append(dict(rows=list(rows), options=options, reference=asdict(reference),
                            rust=asdict(actual)))

    for validation in ("python", "rust"):
        for original_gate in (False, True):
            original = replay_cnot(4, (0,) if original_gate else ())
            replacement = replay_cnot(4, () if original_gate else (0,))
            backing = torch.empty((1, 8, 1), dtype=torch.int64, device=device)
            root = backing[:, ::2, :]
            root.copy_(pack_rows(original, device=device))
            assert not root.is_contiguous()
            gather = World.gather_fixed
            changed = [False]

            def mutate_after_validation(world, tensor):
                if not changed[0]:
                    root.copy_(pack_rows(replacement, device=device))
                    changed[0] = True
                return gather(world, tensor)

            World.gather_fixed = mutate_after_validation
            try:
                result = search_cnot(root, object(), width=1, shortlist=1, max_depth=1,
                                     validation=validation)
            finally:
                World.gather_fixed = gather
            assert changed[0] and result.gates == ((0,) if original_gate else ())
            assert replay_cnot(4, result.gates) == original
            records.append(dict(snapshot_validation=validation, original_gate=original_gate,
                                strided_input=True, gates=list(result.gates)))

    if distributed:
        try:
            search_cnot(pack_rows((1, 2), device=device), object(), width=1, shortlist=1, max_depth=1,
                        validation="rust" if rank == 0 else "python")
        except ValueError as error:
            assert "differ" in str(error)
            records.append(dict(rejected="validator_mismatch", error=str(error)))
        else:
            raise AssertionError("different rank validators were accepted")

        try:
            rows = (1, 1) if rank == 1 else (1, 2)
            search_cnot(pack_rows(rows, device=device), object(), width=1, shortlist=1, max_depth=1,
                        validation="rust")
        except RuntimeError as error:
            assert "rank 1" in str(error) and "singular" in str(error)
            records.append(dict(rejected="rank_1_singular", error=str(error)))
        else:
            raise AssertionError("singular matrix on rank 1 was accepted")

        for validation in ("python", "rust"):
            for fault_rank, failure in ((0, "property"), (1, "json")):
                class BadMetadata:
                    @property
                    def metadata(self):
                        if rank == fault_rank and failure == "property":
                            raise ValueError("metadata unavailable")
                        if rank == fault_rank and failure == "json":
                            return {"checkpoint_sha256": object()}
                        return {}

                root = pack_rows((1, 2), device=device)
                try:
                    search_cnot(root, BadMetadata(), width=1, shortlist=1, max_depth=1,
                                validation=validation)
                except RuntimeError as error:
                    expected = "metadata unavailable" if failure == "property" else "not JSON serializable"
                    assert f"rank {fault_rank}:" in str(error) and expected in str(error)
                    message = str(error)
                else:
                    raise AssertionError("rank-local metadata failure was accepted")
                recovered = search_cnot(root, object(), width=1, shortlist=1, max_depth=1,
                                        validation=validation)
                assert recovered.status == "solved" and recovered.gates == ()
                records.append(dict(rejected="metadata_" + failure, validation=validation,
                                    fault_rank=fault_rank, error=message, reused=True))
    world_size = dist.get_world_size() if distributed else 1
    if "CNOT_EVIDENCE_DIR" in os.environ:
        output = Path(os.environ["CNOT_EVIDENCE_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank-{rank}.json").write_text(json.dumps(dict(
            rank=rank, world=world_size, device=device.type, records=records), indent=2) + "\n")
    print(f"PYO3_CNOT_PUBLIC_SEARCH_OK rank={rank} world={world_size} device={device.type}")
finally:
    if distributed:
        dist.destroy_process_group()
