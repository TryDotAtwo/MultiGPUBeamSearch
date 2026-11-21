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


def invoke(case, output, scorer_factory=None):
    result = verify_cnot.main(
        ["--case", case, "--backend", "gloo", "--engine", "rust", "--output", str(output)],
        scorer_factory=scorer_factory,
        manage_process_group=False,
    )
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
    for failed_rank in (0, 1):
        def factory(*_args, target=failed_rank, **_kwargs):
            if rank == target:
                raise RuntimeError(f"injected scorer load failure rank={target}")
            return SyntheticScorer()

        try:
            invoke("trained", output_root / f"failed-{failed_rank}.json", factory)
        except RuntimeError as error:
            message = str(error)
            assert "load trained scorer failed" in message, message
            assert f"rank {failed_rank}: RuntimeError: injected scorer load failure rank={failed_rank}" in message, message
            observations.append({"failed_rank": failed_rank, "error": message})
        else:
            raise AssertionError(f"rank-local scorer failure on {failed_rank} was not propagated")

        # The public verifier must close its per-search World while retaining
        # the torchrun process group for a healthy retry on the same ranks.
        invoke("wide", output_root / f"healthy-after-{failed_rank}.json")

    if dist.is_initialized():
        dist.destroy_process_group()
    (output_root / f"verify-cnot-lifecycle-rank-{rank}.json").write_text(
        json.dumps(observations, sort_keys=True), encoding="utf-8"
    )
    if rank == 0:
        print("VERIFY_CNOT_GLOO_LIFECYCLE_PASS", flush=True)


if __name__ == "__main__":
    main()
