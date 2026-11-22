"""Exercise the public verify_cnot entry under torchrun/Gloo, including rank-local failure recovery."""
import json
import os
from pathlib import Path
import sys
import tempfile

import torch.distributed as dist
from multigpubeamsearch.cnot_cli import initialize_world


ROOT = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
sys.path.insert(0, str(ROOT / "tools"))
import verify_cnot


class SyntheticScorer:
    def parent_scores(self, parents, mode):
        return parents.new_zeros((len(parents), parents.size(1) * (parents.size(1) - 1)), dtype=parents.dtype)

    def child_values(self, children):
        return children.new_zeros((len(children),), dtype=children.dtype)


def invoke(case, output, scorer_factory=None, *, engine="rust", forbid_python_reference=False):
    from multigpubeamsearch import cnot

    original_search = cnot._search
    if forbid_python_reference:
        def forbidden_python_reference(*_args, **_kwargs):
            raise AssertionError("engine=auto entered the Python CNOT reference loop")
        cnot._search = forbidden_python_reference
    try:
        result = verify_cnot.main(
            ["--case", case, "--backend", "gloo", "--engine", engine, "--output", str(output)],
            scorer_factory=scorer_factory,
            manage_process_group=False,
        )
    finally:
        cnot._search = original_search
    if forbid_python_reference and int(os.environ["RANK"]) == 0:
        print("VERIFY_CNOT_AUTO_DEFAULT_RUST_CONTROLLER_PASS", flush=True)
    if case == "wide" and int(os.environ["RANK"]) == 0:
        report = json.loads(output.read_text(encoding="utf-8"))
        if report.get("status") != "solved" or not report.get("bitwise_replay"):
            raise AssertionError(f"healthy verify_cnot run produced an invalid report: {report}")
    return result


def main():
    rank = int(os.environ["RANK"])
    output_root = Path(os.environ["RUNNER_TEMP"])
    output_root.mkdir(parents=True, exist_ok=True)
    initialize_world("gloo")
    observations = []
    diagnostics = []
    diagnostic_path = output_root / f"verify-cnot-gloo-diagnostic-rank-{rank}.json"

    def report(phase, **fields):
        event = {"rank": rank, "phase": phase, **fields}
        diagnostics.append(event)
        diagnostic_path.write_text(json.dumps(diagnostics, sort_keys=True), encoding="utf-8")
        print("CNOT_GLOO_LIFECYCLE_DIAGNOSTIC " + json.dumps(event, sort_keys=True), flush=True)

    # A different requested controller changes which collectives follow. The
    # public entry must reject rank-local engine choices before path selection.
    requested_engine = "auto" if rank == 0 else "python"
    try:
        verify_cnot.main(
            [
                "--case", "wide", "--backend", "gloo",
                "--engine", requested_engine,
                "--output", str(output_root / f"engine-mismatch-{rank}.json"),
            ],
            scorer_factory=lambda *_args, **_kwargs: SyntheticScorer(),
            manage_process_group=False,
        )
    except (RuntimeError, ValueError) as error:
        rejected = "CNOT search engine settings differ" in str(error)
        report(
            "engine_consensus",
            engine=requested_engine,
            status="rejected" if rejected else "wrong_error",
            error_type=type(error).__name__,
            expected_rejection=rejected,
        )
        assert rejected, "engine mismatch did not reach the common consensus error"
        observations.append({"mismatched_engine": requested_engine, "error": str(error)})
    else:
        report("engine_consensus", engine=requested_engine, status="not_rejected")
        raise AssertionError("rank-local CNOT engine choices were not rejected")

    try:
        invoke(
            "wide",
            output_root / f"healthy-after-engine-mismatch-{rank}.json",
            engine="auto",
            forbid_python_reference=True,
        )
    except BaseException as error:
        report("engine_consensus_retry", status="failed", error_type=type(error).__name__)
        raise
    report("engine_consensus_retry", status="passed")

    for failed_rank in (0, 1):
        def factory(*_args, target=failed_rank, **_kwargs):
            if rank == target:
                raise RuntimeError(f"injected scorer load failure rank={target}")
            return SyntheticScorer()

        try:
            invoke("trained", output_root / f"failed-{failed_rank}.json", factory)
        except BaseException as error:
            message = str(error)
            correct_phase = "load trained scorer failed" in message
            correct_rank = (
                f"rank {failed_rank}: RuntimeError: injected scorer load failure rank={failed_rank}"
                in message
            )
            coordinated = isinstance(error, RuntimeError) and correct_phase and correct_rank
            report(
                f"scorer_failure_rank_{failed_rank}",
                status="coordinated" if coordinated else "wrong_error",
                error_type=type(error).__name__,
                expected_phase=correct_phase,
                expected_rank=correct_rank,
            )
            if not coordinated:
                raise AssertionError("scorer failure did not return the common expected phase") from error
            observations.append({"failed_rank": failed_rank, "error": message})
        else:
            report(f"scorer_failure_rank_{failed_rank}", status="not_rejected")
            raise AssertionError(f"rank-local scorer failure on {failed_rank} was not propagated")

        # The public verifier must close its per-search World while retaining
        # the torchrun process group for a healthy retry on the same ranks.
        try:
            invoke(
                "wide",
                output_root / f"healthy-after-{failed_rank}-{rank}.json",
                engine="auto",
                forbid_python_reference=True,
            )
        except BaseException as error:
            report(
                f"scorer_failure_rank_{failed_rank}_healthy_retry",
                status="failed",
                error_type=type(error).__name__,
            )
            raise
        report(f"scorer_failure_rank_{failed_rank}_healthy_retry", status="passed")

    if dist.is_initialized():
        dist.destroy_process_group()
    (output_root / f"verify-cnot-lifecycle-rank-{rank}.json").write_text(
        json.dumps(observations, sort_keys=True), encoding="utf-8"
    )
    if rank == 0:
        print("VERIFY_CNOT_GLOO_LIFECYCLE_PASS", flush=True)


if __name__ == "__main__":
    main()
