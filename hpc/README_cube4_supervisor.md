# Cube4 supervisor: current acceptance boundary

`tools/cube4_run_supervisor.py` runs a child command in a fresh, exclusive
run directory and POSIX process session. It does not rent, stop or destroy
cloud instances. Windows execution is rejected until Job Object support exists.

Example on a POSIX GPU worker:

```bash
python tools/cube4_run_supervisor.py \
  --run-dir /workspace/runs/unique-run-id \
  --max-elapsed-seconds 120 \
  --max-cost-usd 1 \
  --hourly-rate-usd 0.189 \
  -- /absolute/path/to/prepared-launcher
```

Use an unused run path. Child cwd is that directory; commands and input paths
must not assume the repository cwd. Caller-provided hourly rate should include
all relevant continuing charges. The smaller of elapsed limit and cost/rate
sets the child deadline. Termination grace and scheduling overhead may exceed
that duration: this is not a hard financial guarantee or a billing receipt.
Instance and disk billing continue after a child is killed.

`child.log` contains raw output; `summary.json` is a flushed/file-fsynced,
atomically replaced single-file report. Directory fsync and multi-file
transactional publication are not implemented. Child nonzero exit is failed;
deadline expiry is timed_out. Zero exit is completed_unverified, never proof
of solved or unsolved. `production_accepted` remains false.

Optional paired `--world-size N --puzzle-contract /absolute/contract.json`
enables rank-result verification. The parent contract contains `puzzle_id`,
`initial_state`, `central_state` and named permutation `generators`.
State symbol multisets must agree; generator names must be single legal move
tokens and permutation indices exact integers (not bool or float).
Every child rank must publish `rank-status/rank-R.json` with its exact integer
`rank`, matching integer `puzzle_id`, zero integer `exit_code` and unanimous
`status`. Solved reports require a single-row `submit.csv` with columns
`initial_state_id,path`; the path must independently replay to the central
state. Contract SHA-256 is recorded. Malformed/non-object JSON and duplicate
keys are rejected. Set `BEAM_RANK_STATUS_DIR=rank-status` for native emission
after cleanup. This is supported only for a single-puzzle first-solution run;
repair and solve-bucket modes reject this option. Each rank reserves its own
publication and refuses overwrite. File flush/close and atomic rename are
checked, but fsync/durable filesystem transactions are not implemented.
Submission read is bounded to 1MiB before CSV parsing/replay. Unsolved requires
an integer parent `requested_depth` in [1,2^32) and every rank's exact integer
`completed_depths` equal to that value; missing/early/bool values fail closed.
Frontier exhaustion before the requested depth is not yet a supported terminal
proof. Rank reports are a protocol, not independently authenticated evidence.

Linux acceptance currently covers deadline, cost-derived deadline, failed
exit, false solved text, external SIGTERM, TERM-ignoring descendants in the
owned session, and survival of an unrelated process. A descendant that creates
a new session can escape group cleanup; this is not a sandbox.

Spawn hand-off cancellation is now covered by a deterministic real-Popen
SIGTERM injection: ownership is recorded before interrupted cleanup.

Remaining gates: signal storms, snapshot/publication
failure, broad native rank-status failure coverage, replay wall-time enforcement,
launcher integration, and multi-node resource ownership. These are required
before using this supervisor as the production campaign acceptance boundary.

Bounded actual 2xRTX3060 FP16 acceptance: a three-move fixture solves under
depth5 with both reports and independent replay; depth1 gives unanimous
unsolved with completed_depths=1. This is not H200 throughput, large-beam,
multi-node, repair or collect-all acceptance.
