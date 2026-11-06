# Native search across GPU nodes

Run one `multigpubeamsearch.cluster_worker` per assigned GPU on Linux. The
workers start a single native NCCL search with one global beam. The maximum
world size is 128, matching the fixed arrays used by the native final exchange.

Install the package from the same revision on every node. Use a native
`production_runner` built for the graph shape, executor, and compute
architectures on the participating nodes. Provide the same generators, center,
input states, and model to every rank. A small directory on a filesystem
mounted at the **same path on all nodes** carries the NCCL bootstrap ID and
worker logs. The actual candidate exchange uses NCCL.

## Two nodes with eight GPUs each

This example uses an existing prepared native MLP with its matching runner.
Set these variables on both hosts (or export them before running `srun`).
The puzzle ID, forward depth limit and **global** beam are the three positional
arguments after the runner.

```bash
export RUNNER=/shared/search/bin/production_runner
export BEAM_GENERATOR_PATH=/shared/search/inputs/puzzle_info.json
export BEAM_PUZZLE_INFO_JSON="$BEAM_GENERATOR_PATH"
export BEAM_TEST_CSV=/shared/search/inputs/test.csv
export BEAM_WEIGHT_DIR=/shared/search/weights
export RUNS=/shared/search/runs
```

Inside a two-node Slurm allocation, run the following once. Slurm binds one
visible GPU to each task and terminates the step if any task fails.

```bash
srun --nodes=2 --ntasks=16 --ntasks-per-node=8 --gpus-per-task=1 \
  --kill-on-bad-exit=1 python -m multigpubeamsearch.cluster_worker \
  --shared-dir "$RUNS" --timeout 3600 -- "$RUNNER" 0 20 1000003
```

With torchrun, run this command once on **each** host. Choose one fresh ID and
a reachable address of the first node. Give both agents the same values.

```bash
torchrun --nnodes=2 --nproc-per-node=8 --max-restarts=0 \
  --rdzv-backend=c10d --rdzv-endpoint=gpu-node-01:29400 --rdzv-id=search-001 \
  -m multigpubeamsearch.cluster_worker --shared-dir "$RUNS" --timeout 3600 \
  -- "$RUNNER" 0 20 1000003
```

The module also exports `multigpubeamsearch.run_cluster_rank(command,
shared_dir=..., run_id=..., timeout=...)` for scheduler worker scripts. Keep
the binary's arguments as individual strings. The call returns this rank's
artifact directory on exit code zero. The launcher exit status covers the
whole job. For a custom worker, `scheduler_rank()` returns the communicator
rank, node number, node-local task number, world size and local group size.
Use `rank.cuda_ordinal(torch.cuda.device_count())` to select a CUDA device
inside that task's visible namespace.

## Inputs, placement, and results

- `RANK` identifies the communicator peer. In a conventional 2×8 block
  assignment, global rank 15 has node-local rank 7. It selects CUDA ordinal 7
  with eight visible devices, or ordinal 0 with a per-task one-GPU mask.
  The worker preserves `CUDA_VISIBLE_DEVICES` and the NCCL network settings.
- The top-k in the example is exactly 1,000,003. Each rank allocates capacity
  for 62,501 frontier states. At startup all ranks exchange their available
  device-memory snapshots, plan using the smallest budget, and check the
  resulting common search and exchange geometry before entering the depth loop.
  The run fails if any rank cannot admit its requested workload.
- The worker sets disk-backed history by default. `BEAM_HISTORY_RAM_BYTES`
  for static-hybrid history is a **per-node** budget split over that node's
  ranks. `BEAM_HISTORY_DISK_BYTES` is a **job-wide** allowance split over all
  ranks. Use absolute job-specific paths for node-local scratch or optional
  native inputs. A history directory on node-local storage is sufficient
  because each rank reconstructs its own history entries.
- Each task writes `launch.json`, `native.log`, and native artifacts under
  `$RUNS/<run-id>/attempt-<restart>/rank-<global-rank>/`. The solving rank's
  native log reports `puzzle_solved=1` and the replayed solution. A failed
  worker writes `FAILED` in the attempt directory. Peers observe that file
  and terminate their native children. `--timeout` bounds each worker's wall
  time. `BEAM_NCCL_RENDEZVOUS_TIMEOUT_SECONDS` sets the native wait for
  rank zero's ID (default 60 seconds).
- Use a fresh ID for every launch. The default is the torchrun rendezvous ID
  or the Slurm job and step IDs. An explicit `--run-id` is required for a
  static torchrun configuration that exposes the placeholder `none`. Each
  worker reserves its own rank and local slot before launch.

The external worker uses native runtime planning. Notebook autosizing and model
export run separately through the local CayleyPy adapter. The included CPU
process tests exercise 16 workers, shared rendezvous, and failure cleanup.
Physical throughput and solve quality on 16 or 128 GPUs require a hardware run.

Scheduler environment definitions are in the [PyTorch torchrun documentation](https://docs.pytorch.org/docs/stable/elastic/run.html)
and the [Slurm srun manual](https://slurm.schedmd.com/srun.html).
