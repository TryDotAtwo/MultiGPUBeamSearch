"""Admission failures must reach every rank without poisoning the world."""
from datetime import timedelta
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from multigpubeamsearch.cnot import pack_rows, search_cnot


@pytest.mark.parametrize("failure", ["property", "json"])
def test_metadata_exceptions_stay_inside_the_admission_failure_boundary(monkeypatch, failure):
    from multigpubeamsearch import cnot

    class Scorer:
        @property
        def metadata(self):
            if failure == "property":
                raise ValueError("metadata unavailable")
            return {"checkpoint_sha256": object()}

    class CollectiveProbe:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def checked(self, stage, fn):
            try:
                return fn()
            except Exception as error:
                raise RuntimeError(f"{stage} failed: rank 0: {error}") from error

        def gather_fixed(self, _):
            raise AssertionError("a failing rank must not enter the digest exchange")

    monkeypatch.setattr(cnot, "World", CollectiveProbe)
    with pytest.raises(RuntimeError, match="validate CNOT input failed: rank 0:"):
        search_cnot(pack_rows((1, 2)), Scorer(), width=1, shortlist=1, max_depth=1)


def _worker(rank, rendezvous, output_dir, fault_rank, failure):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=8),
    )
    try:
        class Scorer:
            @property
            def metadata(self):
                if rank == fault_rank and failure == "property":
                    raise ValueError("metadata unavailable")
                if rank == fault_rank and failure == "json":
                    return {"checkpoint_sha256": object()}
                return {}

        root = pack_rows((1, 2))
        result = {}
        try:
            search_cnot(root, Scorer(), width=1, shortlist=1, max_depth=1)
        except Exception as error:
            result["kind"] = type(error).__name__
            result["error"] = str(error)
        else:
            result["kind"] = "accepted"

        # Only attempt a new collective when this rank saw the coordinated
        # error; in the unfixed implementation the process group is broken.
        if result["kind"] == "RuntimeError" and "validate CNOT input failed" in result["error"]:
            success = search_cnot(root, object(), width=1, shortlist=1, max_depth=1)
            result["reused"] = success.status == "solved" and success.gates == ()
        Path(output_dir, f"rank-{rank}.json").write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="two CPU ranks require Gloo")
@pytest.mark.parametrize("fault_rank,failure", [(0, "property"), (1, "json")])
def test_rank_local_metadata_failure_is_collective_and_group_is_reusable(tmp_path, fault_rank, failure):
    if sys.platform == "win32":
        try:
            dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")
        except RuntimeError as error:
            if "unsupported gloo device" not in str(error):
                raise
            pytest.skip("this Windows torch wheel has no Gloo transport")
    rendezvous = (tmp_path / "rendezvous").resolve().as_uri()
    mp.spawn(_worker, args=(rendezvous, str(tmp_path), fault_rank, failure), nprocs=2)
    results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    for result in results:
        assert result["kind"] == "RuntimeError", results
        assert f"rank {fault_rank}:" in result["error"], results
        assert ("metadata unavailable" if failure == "property" else "not JSON serializable") in result["error"]
        assert result["reused"] is True
