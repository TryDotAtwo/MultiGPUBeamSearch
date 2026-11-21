"""Hosted CPU admission tests for required CNOT backends and rank agreement."""
from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot import pack_rows, replay_cnot, search_cnot
from multigpubeamsearch.rust_control import extension


ROOT = Path(__file__).resolve().parents[3]
RUNTIME = extension()
FIXTURE = ROOT / "integrations/cayleypy_native/tests/cnot_device_domain_admission_fixture.c"


class FlatScorer:
    metadata = {}

    def parent_scores(self, parents, mode):
        alphabet = parents.size(1) * (parents.size(1) - 1)
        return torch.zeros((len(parents), alphabet), dtype=torch.float64, device=parents.device)

    def child_values(self, children):
        return torch.zeros(len(children), dtype=torch.float64, device=children.device)


def build_library(directory, name, *defines):
    output = Path(directory) / f"lib{name}.so"
    subprocess.run(
        ["cc", "-shared", "-fPIC", "-I", str(ROOT), *defines, str(FIXTURE), "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


def distributed_contract_worker(rank, rendezvous, output_dir, valid, partial):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        world = World("cpu")
        result = {}
        with world:
            strict = True
            path = valid if rank == 0 else partial
            try:
                contract = world.checked(
                    "admit strict CNOT backends",
                    lambda: RUNTIME._probe_cnot_backend_admission(
                        strict, str(path) if strict else None, True, rank, 4, 2
                    ),
                )
                RUNTIME._require_cnot_backend_consensus(world, contract)
            except Exception as error:
                result = {"kind": type(error).__name__, "error": str(error)}
            world.require_engine_consensus(b"control-group-still-live")
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def assert_rank_pair(tmp, rendezvous_name, valid, partial, expected):
    rendezvous = (Path(tmp) / rendezvous_name).resolve().as_uri()
    mp.spawn(
        distributed_contract_worker,
        args=(rendezvous, str(tmp), valid, partial),
        nprocs=2,
        join=True,
    )
    results = [json.loads((Path(tmp) / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(row == results[0] for row in results), results
    assert expected in results[0]["error"], results


def activation_worker(rank, rendezvous, output_dir, library):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=20))
    try:
        with World("cpu") as world:
            strict = rank == 0
            contract = world.checked(
                "admit strict CNOT backends",
                lambda: RUNTIME._probe_cnot_backend_admission(
                    strict, str(library) if strict else None, strict, rank, 4, 2
                ),
            )
            result = {}
            try:
                RUNTIME._require_cnot_backend_consensus(world, contract)
            except Exception as error:
                result = {"kind": type(error).__name__, "error": str(error)}
            world.require_engine_consensus(b"control-group-still-live")
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def public_activation_worker(rank, rendezvous, output_dir):
    torch.set_num_threads(1)
    if rank == 0:
        os.environ["MGBS_CNOT_STRICT_BACKEND"] = "1"
    else:
        os.environ.pop("MGBS_CNOT_STRICT_BACKEND", None)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=20),
    )
    try:
        with World("cpu") as world:
            try:
                search_cnot(
                    pack_rows(replay_cnot(3, (0,))),
                    FlatScorer(),
                    width=6, shortlist=6, max_depth=3, validation="python", engine="rust",
                )
            except ValueError as error:
                result = {"error": str(error)}
            else:
                raise AssertionError("public CNOT search accepted rank-dependent strict activation")
            assert "CNOT search engine settings differ between ranks" in result["error"]
            world.require_engine_consensus(b"public-search-recovered-after-admission-error")
        Path(output_dir, f"public-rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


def main():
    with tempfile.TemporaryDirectory(prefix="mgbs-cnot-admission-") as temporary:
        root = Path(temporary)
        valid = build_library(root, "valid")
        wrong_base = build_library(root, "wrong_base", "-DMGBS_TEST_DOMAIN_ABI=9")
        wrong_device = build_library(root, "wrong_device", "-DMGBS_TEST_DEVICE_ABI=9")
        missing_unseen = build_library(root, "missing_unseen", "-DMGBS_TEST_MISSING_UNSEEN=1")

        assert RUNTIME._probe_cnot_backend_admission(False, None, False, 0, 4, 1) == "strict=0"
        try:
            RUNTIME._probe_cnot_backend_admission(True, None, True, 0, 4, 1)
        except RuntimeError as error:
            assert "MGBS_CNOT_OWNED_LIBRARY" in str(error)
        else:
            raise AssertionError("strict admission accepted a missing required DeviceDomain path")

        for library, message in ((wrong_base, "CNOT domain ABI mismatch"),
                                 (wrong_device, "CNOT device ABI mismatch"),
                                 (missing_unseen, "borrowed-device selection")):
            try:
                RUNTIME._probe_cnot_backend_admission(True, library, True, 0, 4, 1)
            except RuntimeError as error:
                assert message in str(error), (library, error)
            else:
                raise AssertionError(f"strict admission accepted unsupported library {library}")

        contract = RUNTIME._probe_cnot_backend_admission(True, valid, True, 0, 4, 1)
        assert "capabilities=owners,expansion,full_matrix_distinct" in contract

        mismatch_dir = root / "rank-mismatch"
        mismatch_dir.mkdir()
        assert_rank_pair(mismatch_dir, "rendezvous", valid, missing_unseen,
                         "admit strict CNOT backends failed")

        activation_dir = root / "activation-mismatch"
        activation_dir.mkdir()
        # Here both local probes succeed; only the strict activation differs.
        activation_rendezvous = (activation_dir / "rendezvous").resolve().as_uri()
        mp.spawn(activation_worker, args=(activation_rendezvous, str(activation_dir), valid),
                 nprocs=2, join=True)
        activation_results = [json.loads((activation_dir / f"rank-{rank}.json").read_text())
                              for rank in range(2)]
        assert activation_results[0] == activation_results[1], activation_results
        assert "settings differ between ranks" in activation_results[0]["error"]

        public_activation_dir = root / "public-activation-mismatch"
        public_activation_dir.mkdir()
        public_rendezvous = (public_activation_dir / "rendezvous").resolve().as_uri()
        mp.spawn(public_activation_worker,
                 args=(public_rendezvous, str(public_activation_dir)), nprocs=2, join=True)
        public_activation_results = [
            json.loads((public_activation_dir / f"public-rank-{rank}.json").read_text())
            for rank in range(2)
        ]
        assert public_activation_results[0] == public_activation_results[1], public_activation_results
        assert "CNOT search engine settings differ between ranks" in public_activation_results[0]["error"]

    print("CNOT_BACKEND_ADMISSION_CPU_OK")


if __name__ == "__main__":
    main()
