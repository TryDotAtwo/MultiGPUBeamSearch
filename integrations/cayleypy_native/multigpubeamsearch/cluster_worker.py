"""One externally scheduled rank. Use torchrun -m or srun python -m."""
import argparse
import signal
import sys

from .cluster import run_cluster_rank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", required=True, help="same shared filesystem path on every node")
    parser.add_argument("--run-id", help="fresh ID for this job; defaults to torchrun or Slurm job/step ID")
    parser.add_argument("--timeout", type=float, default=3600, help="worker wall time in seconds")
    parser.add_argument("--local-rank", "--local_rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- production_runner PUZZLE DEPTH GLOBAL_BEAM")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("supply a native command after --")

    # Torch elastic may signal the worker before its child has finished.
    # Unwind run_cluster_rank to publish the failure and reap that child.
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        directory = run_cluster_rank(command, shared_dir=args.shared_dir, run_id=args.run_id, timeout=args.timeout)
    except (OSError, ValueError, RuntimeError) as error:
        print(error, file=sys.stderr, flush=True)
        return 1
    print(directory, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
