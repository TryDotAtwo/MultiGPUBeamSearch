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
import multigpubeamsearch.cnot as cnot_module
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


def domain_prepare_failure_worker(rank, rendezvous, output_dir, library):
    """A failed rank-local backend prepare must be voted before later Gloo work."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2,
        timeout=timedelta(seconds=20),
    )
    observations = []
    try:
        with World("cpu") as world:
            for failed_rank in (0, 1):
                try:
                    RUNTIME._probe_cnot_device_domain_vote(
                        world, str(library), rank, 4, failed_rank
                    )
                except Exception as error:
                    failure = {"kind": type(error).__name__, "error": str(error)}
                else:
                    failure = {"kind": "missing-failure", "error": ""}

                # Both ranks must still be in the same Gloo phase after the
                # rank-local failure has been reported to every participant.
                world.require_engine_consensus(
                    f"after-device-domain-failure-{failed_rank}".encode()
                )
                healthy = RUNTIME._probe_cnot_device_domain_vote(
                    world, str(library), rank, 4, -1
                )
                world.require_engine_consensus(healthy.encode())
                observations.append({"failed_rank": failed_rank, **failure,
                                    "healthy_contract": healthy})
        Path(output_dir, f"domain-vote-rank-{rank}.json").write_text(
            json.dumps(observations), encoding="utf-8"
        )
    finally:
        dist.destroy_process_group()


def dispatch_binding_worker(rank, rendezvous, output_dir):
    """Run public CNOT routing with rank-local missing symbols and healthy retries."""
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    original_native_method = World.native_method
    original_start_transport = World.start_transport
    original_native_stages = cnot_module._native_stages_runner
    original_compat_stages = cnot_module._make_compat_stages
    outcomes = []
    root = pack_rows(replay_cnot(3, (0, 1)))
    try:
        with World("cpu") as outer:
            for fail_rank in (0, 1):
                def search(engine):
                    return search_cnot(
                        root,
                        FlatScorer(),
                        width=6,
                        shortlist=6,
                        max_depth=3,
                        validation="python",
                        engine=engine,
                    )

                def hide_native_stage(world, name, target=fail_rank):
                    if rank == target and name == "run_cnot_native":
                        return None
                    return original_native_method(world, name)

                # Strict selection must reject the rank-local missing required
                # stage ABI before Stream 5 or any data-plane transport starts.
                World.native_method = hide_native_stage
                transport_calls = []
                def record_transport_start(world, *, prefer_rust):
                    transport_calls.append(world.rank)
                    return original_start_transport(world, prefer_rust=prefer_rust)

                World.start_transport = record_transport_start
                old_strict = os.environ.get("MGBS_CNOT_STRICT_BACKEND")
                os.environ["MGBS_CNOT_STRICT_BACKEND"] = "1"
                try:
                    try:
                        search("rust")
                    except RuntimeError as error:
                        strict_error = str(error)
                    else:
                        raise AssertionError("strict CNOT accepted a rank-missing stage ABI")
                finally:
                    if old_strict is None:
                        os.environ.pop("MGBS_CNOT_STRICT_BACKEND", None)
                    else:
                        os.environ["MGBS_CNOT_STRICT_BACKEND"] = old_strict
                    World.native_method = original_native_method
                    World.start_transport = original_start_transport
                if "strict CNOT acceptance requires matching Rust controller, stage" not in strict_error:
                    raise AssertionError(f"strict CNOT reported the wrong admission error: {strict_error}")
                if transport_calls:
                    raise AssertionError(f"strict CNOT started transport before rejection: {transport_calls}")
                outcomes.append({
                    "rank": rank,
                    "failed_rank": fail_rank,
                    "case": "strict_missing_stage",
                    "error": strict_error,
                    "transport_calls": len(transport_calls),
                })
                outer.require_engine_consensus(f"strict-rejected-{fail_rank}".encode())

                # Auto must select the common Rust compatibility path from the
                # capability snapshot. Calling the native stage builder would
                # mean the runtime made a second, late route decision.
                stage_builder_calls = []
                compatibility_builds = []
                def forbidden_native_stage_builder():
                    stage_builder_calls.append(True)
                    raise RuntimeError("non-common Rust native stage builder was invoked")

                def track_compatibility_stages(*args, **kwargs):
                    compatibility_builds.append(True)
                    return original_compat_stages(*args, **kwargs)

                World.native_method = hide_native_stage
                cnot_module._native_stages_runner = forbidden_native_stage_builder
                cnot_module._make_compat_stages = track_compatibility_stages
                try:
                    result = search("auto")
                finally:
                    World.native_method = original_native_method
                    cnot_module._native_stages_runner = original_native_stages
                    cnot_module._make_compat_stages = original_compat_stages
                if stage_builder_calls:
                    raise AssertionError("auto selected RustController without a common native stage ABI")
                if not compatibility_builds:
                    raise AssertionError("auto did not execute the common Rust compatibility route")
                if result.status != "solved" or replay_cnot(3, result.gates) != replay_cnot(3, (0, 1)):
                    raise AssertionError(f"Rust compatibility fallback returned an invalid result: {result!r}")
                outcomes.append({
                    "rank": rank,
                    "failed_rank": fail_rank,
                    "case": "auto_compatibility",
                    "status": result.status,
                    "gates": list(result.gates),
                    "native_stage_builder_calls": len(stage_builder_calls),
                    "compatibility_stage_builds": len(compatibility_builds),
                })
                outer.require_engine_consensus(f"auto-compatibility-{fail_rank}".encode())

                # With no common Rust controller or compatibility controller,
                # explicit Rust must fail identically; then auto must recover
                # in the same distributed process group.
                def hide_all_rust_controllers(world, name, target=fail_rank):
                    if rank == target and name in ("run_cnot_native_search", "run_cnot"):
                        return None
                    return original_native_method(world, name)

                World.native_method = hide_all_rust_controllers
                try:
                    try:
                        search("rust")
                    except RuntimeError as error:
                        explicit_error = str(error)
                    else:
                        raise AssertionError("explicit Rust accepted the absence of a common Rust route")
                finally:
                    World.native_method = original_native_method
                if "CNOT Rust engine requested but no compatible Rust controller is available" not in explicit_error:
                    raise AssertionError(f"explicit Rust reported the wrong no-route error: {explicit_error}")
                outcomes.append({
                    "rank": rank,
                    "failed_rank": fail_rank,
                    "case": "rust_without_common_route",
                    "error": explicit_error,
                })
                outer.require_engine_consensus(f"rust-no-route-{fail_rank}".encode())

                result = search("auto")
                if result.status != "solved" or replay_cnot(3, result.gates) != replay_cnot(3, (0, 1)):
                    raise AssertionError(f"healthy auto retry failed after dispatch rejection: {result!r}")
                outcomes.append({
                    "rank": rank,
                    "failed_rank": fail_rank,
                    "case": "healthy_auto_retry",
                    "status": result.status,
                    "gates": list(result.gates),
                })
                outer.require_engine_consensus(f"healthy-auto-retry-{fail_rank}".encode())

        Path(output_dir, f"dispatch-binding-rank-{rank}.json").write_text(
            json.dumps(outcomes, sort_keys=True), encoding="utf-8"
        )
    finally:
        World.native_method = original_native_method
        World.start_transport = original_start_transport
        cnot_module._native_stages_runner = original_native_stages
        cnot_module._make_compat_stages = original_compat_stages
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

        domain_vote_dir = root / "device-domain-rank-failure"
        domain_vote_dir.mkdir()
        domain_vote_rendezvous = (domain_vote_dir / "rendezvous").resolve().as_uri()
        mp.spawn(domain_prepare_failure_worker,
                 args=(domain_vote_rendezvous, str(domain_vote_dir), valid),
                 nprocs=2, join=True)
        domain_vote_results = [
            json.loads((domain_vote_dir / f"domain-vote-rank-{rank}.json").read_text())
            for rank in range(2)
        ]
        assert domain_vote_results[0] == domain_vote_results[1], domain_vote_results
        for failed_rank, result in enumerate(domain_vote_results[0]):
            assert result["failed_rank"] == failed_rank, result
            assert "prepare admitted CNOT DeviceDomain failed" in result["error"], result
            assert "domain_abi=1" in result["healthy_contract"], result

        dispatch_dir = root / "rank-local-dispatch"
        dispatch_dir.mkdir()
        dispatch_rendezvous = (dispatch_dir / "rendezvous").resolve().as_uri()
        mp.spawn(
            dispatch_binding_worker,
            args=(dispatch_rendezvous, str(dispatch_dir)),
            nprocs=2,
            join=True,
        )
        dispatch_reports = [
            json.loads((dispatch_dir / f"dispatch-binding-rank-{rank}.json").read_text())
            for rank in range(2)
        ]
        for fail_rank in (0, 1):
            def pair(case):
                return [
                    next(row for row in rows if row["failed_rank"] == fail_rank and row["case"] == case)
                    for rows in dispatch_reports
                ]

            strict_pair = pair("strict_missing_stage")
            assert strict_pair[0]["error"] == strict_pair[1]["error"], strict_pair
            assert strict_pair[0]["transport_calls"] == strict_pair[1]["transport_calls"] == 0, strict_pair
            auto_pair = pair("auto_compatibility")
            assert all(row["status"] == "solved" for row in auto_pair), auto_pair
            assert all(row["native_stage_builder_calls"] == 0
                       and row["compatibility_stage_builds"] > 0 for row in auto_pair), auto_pair
            rust_pair = pair("rust_without_common_route")
            assert rust_pair[0]["error"] == rust_pair[1]["error"], rust_pair
            retry_pair = pair("healthy_auto_retry")
            assert all(row["status"] == "solved" for row in retry_pair), retry_pair

    print("CNOT_BACKEND_ADMISSION_CPU_OK")


if __name__ == "__main__":
    main()
