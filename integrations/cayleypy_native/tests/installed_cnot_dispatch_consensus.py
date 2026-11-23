"""Installed-wheel regression for rank-consistent CNOT route admission."""
from datetime import timedelta
import hashlib
import os

import torch.distributed as dist

from multigpubeamsearch import cnot
from multigpubeamsearch.cnot import pack_rows, search_cnot
from multigpubeamsearch.cnot_collectives import World


def _all_ranks_equal(label, value):
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, value)
    if any(item != gathered[0] for item in gathered[1:]):
        raise AssertionError(f"{label} differed across ranks: {gathered}")
    return gathered[0]


def _attempt(*, hidden_rank=None, hidden_names=(), engine="auto", strict=False, legacy_digest=False):
    rank = dist.get_rank()
    hidden_names = set(hidden_names)
    events = []
    digest_events = []
    original_native_method = World.native_method
    original_runner = cnot._native_search_runner
    original_start_transport = World.start_transport
    previous_strict = os.environ.get("MGBS_CNOT_STRICT_BACKEND")

    def native_method(self, name):
        binding = original_native_method(self, name)
        if name == "_cnot_public_dispatch_digest":
            if legacy_digest:
                return None
            if callable(binding):
                digest_events.append(name)
        if self.rank == hidden_rank and name in hidden_names:
            return None
        return binding

    def native_runner():
        binding = original_runner()
        if rank == hidden_rank and "run_cnot_native_search" in hidden_names:
            return None
        return binding

    def observe_transport(self, *, prefer_rust):
        # Root is identity in these probes, so no data-plane operation is needed.
        events.append(bool(prefer_rust))

    World.native_method = native_method
    cnot._native_search_runner = native_runner
    World.start_transport = observe_transport
    if strict:
        os.environ["MGBS_CNOT_STRICT_BACKEND"] = "1"
    else:
        os.environ.pop("MGBS_CNOT_STRICT_BACKEND", None)
    try:
        try:
            result = search_cnot(
                pack_rows((1, 2, 4, 8)),
                object(),
                width=1,
                shortlist=1,
                max_depth=2,
                validation="python",
                engine=engine,
            )
        except BaseException as error:
            record = {
                "error": [type(error).__name__, str(error)],
                "status": None,
                "gates": None,
                "transport_preferences": list(events),
                "native_dispatch_digest_calls": len(digest_events),
            }
        else:
            record = {
                "error": None,
                "status": result.status,
                "gates": list(result.gates) if result.gates is not None else None,
                "transport_preferences": list(events),
                "native_dispatch_digest_calls": len(digest_events),
            }
    finally:
        World.native_method = original_native_method
        cnot._native_search_runner = original_runner
        World.start_transport = original_start_transport
        if previous_strict is None:
            os.environ.pop("MGBS_CNOT_STRICT_BACKEND", None)
        else:
            os.environ["MGBS_CNOT_STRICT_BACKEND"] = previous_strict
    return _all_ranks_equal("CNOT dispatch outcome", record)


def _expect_solved(label, record, expected_transport, *, digest_calls=1):
    if record["error"] is not None or record["status"] != "solved" or record["gates"] != []:
        raise AssertionError(f"{label} did not complete the identity search: {record}")
    if record["native_dispatch_digest_calls"] != digest_calls:
        raise AssertionError(
            f"{label} used {record['native_dispatch_digest_calls']} Rust digest calls; "
            f"expected {digest_calls}: {record}"
        )
    if record["transport_preferences"] != expected_transport:
        raise AssertionError(
            f"{label} ignored the rank-agreed dispatch path: "
            f"{record['transport_preferences']} != {expected_transport}"
        )


def _expect_coordinated_failure(label, record):
    if record["error"] is None:
        raise AssertionError(f"{label} unexpectedly selected a route: {record}")
    if record["transport_preferences"]:
        raise AssertionError(f"{label} entered transport before admission failed: {record}")


def _assert_rust_dispatch_digest_matches_compatibility():
    from multigpubeamsearch.rust_control import extension

    digest_builder = getattr(extension(), "_cnot_public_dispatch_digest", None)
    if not callable(digest_builder):
        raise AssertionError("installed Rust extension lacks the CNOT dispatch digest")
    for engine in ("auto", "python", "rust"):
        engine_digest = hashlib.sha256(
            b"cnot-requested-engine-v1:" + engine.encode("utf-8")
        ).digest()
        for max_depth in (0, (1 << 64) - 1, 1 << 64):
            depth_fits = max_depth <= (1 << 64) - 1
            for strict in (False, True):
                expected = hashlib.sha256(
                    b"cnot-public-dispatch-v1:"
                    + bytes((int(strict), int(depth_fits)))
                    + engine_digest
                ).digest()
                actual = digest_builder(engine, max_depth, strict)
                if type(actual) is not bytes or actual != expected:
                    raise AssertionError(
                        f"Rust dispatch digest changed the compatibility contract: "
                        f"engine={engine}, depth={max_depth}, strict={strict}"
                    )
                legacy = cnot._public_dispatch_digest(
                    engine, max_depth, strict, native_digest=None
                )
                if legacy != expected:
                    raise AssertionError(
                        f"legacy wheel fallback changed the digest contract: "
                        f"engine={engine}, depth={max_depth}, strict={strict}"
                    )
    try:
        digest_builder("invalid", 1, False)
    except ValueError as error:
        if "CNOT engine must be auto, python, or rust" not in str(error):
            raise AssertionError(f"wrong invalid-engine error: {error}") from error
    else:
        raise AssertionError("Rust dispatch digest accepted an invalid engine")
    print("CNOT_RUST_PUBLIC_DISPATCH_DIGEST_PASS", flush=True)


def _assert_rust_selector_reads_depth_once():
    from multigpubeamsearch.rust_control import extension

    selector = getattr(extension(), "_select_cnot_dispatch")

    class ProbeDepth(int):
        comparisons = 0

        def __gt__(self, other):
            type(self).comparisons += 1
            if type(self).comparisons > 1:
                raise AssertionError("CNOT dispatch compared depth more than once")
            return super().__gt__(other)

    capabilities = {
        "native_controller": False,
        "checked_control": False,
        "native_stages": False,
        "stream5_start": False,
        "world_exchange": False,
        "fixed_gather": False,
        "row_gather": False,
        "broadcast": False,
        "compatibility_controller": True,
    }
    selected = selector("auto", ProbeDepth(2), False, capabilities)
    if selected != "rust_compatibility":
        raise AssertionError(f"unexpected typed CNOT dispatch path: {selected}")
    if ProbeDepth.comparisons != 1:
        raise AssertionError(f"CNOT selector read depth {ProbeDepth.comparisons} times")
    print("CNOT_RUST_SELECTOR_SINGLE_PARSE_PASS", flush=True)

def main():
    phase = "rust_dispatch_digest"
    try:
        _assert_rust_dispatch_digest_matches_compatibility()
        phase = "rust_selector"
        _assert_rust_selector_reads_depth_once()
        phase = "gloo_init"
        dist.init_process_group("gloo", timeout=timedelta(seconds=90))
        for hidden_rank in (0, 1):
            phase = f"auto_missing_controller_rank_{hidden_rank}"
            record = _attempt(
                hidden_rank=hidden_rank,
                hidden_names={"run_cnot_native_search"},
            )
            # The common Rust compatibility controller is still available.
            _expect_solved(
                f"auto with rank-{hidden_rank} missing Rust controller",
                record,
                [True],
            )

        # With no common Rust controller or compatibility controller, auto
        # must honor the shared PythonReference path instead of reselecting.
        phase = "auto_python_fallback"
        record = _attempt(
            hidden_rank=0,
            hidden_names={"run_cnot_native_search", "run_cnot"},
        )
        _expect_solved("auto PythonReference fallback", record, [False])

        # Strict acceptance must reject a missing required operation before
        # any transport setup. The error text and phase are identical per rank.
        phase = "strict_missing_operation"
        record = _attempt(
            hidden_rank=0,
            hidden_names={"run_cnot_native_start_transport"},
            strict=True,
        )
        _expect_coordinated_failure("strict missing operation", record)

        # Explicit Rust must fail collectively when neither Rust controller is
        # available on every rank.
        phase = "explicit_rust_missing_route"
        record = _attempt(
            hidden_rank=0,
            hidden_names={"run_cnot_native_search", "run_cnot"},
            engine="rust",
        )
        _expect_coordinated_failure("explicit Rust without a common route", record)

        # An older wheel has no Rust digest export. Its Python fallback must
        # produce the same rank-consensus bytes and still finish the search.
        phase = "legacy_digest_fallback"
        record = _attempt(engine="auto", legacy_digest=True)
        _expect_solved(
            "legacy wheel digest fallback",
            record,
            [],
            digest_calls=0,
        )

        # The live default process group must remain usable after all failures.
        phase = "healthy_retry"
        record = _attempt(engine="auto")
        _expect_solved("healthy auto retry", record, [])
        if dist.get_rank() == 0:
            print("INSTALLED_CNOT_DISPATCH_CONSENSUS_PASS", flush=True)
    except BaseException as error:
        error_type = type(error).__name__
        if not error_type.isascii() or not error_type.replace("_", "").isalnum():
            error_type = "Exception"
        print(
            f"CNOT_DISPATCH_CONSENSUS_FAILURE phase={phase} type={error_type}",
            flush=True,
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
