# CNOT synthesis on one distributed beam

The input is an invertible binary matrix. Row integers use bit 0 for column 0.
Physical action IDs enumerate controls first and skip the diagonal. Applying
`c → t` XORs row `c` into row `t`. The solver reduces the input to identity,
backtracks, returns gates in **construction order**, and independently replays
every bit from the identity. Width is the total number of retained matrices
across all ranks. Shortlist is the number of actions proposed per parent.

## Run with the trained model

Build the scorer against the pinned, operator-supplied TheQuantumProject
checkout. The binding uses the project's checkpoint loader, feature registry,
coverage checks and three model heads. Supply its model manifest and associated
files to each rank. Private checkpoints and donor source are supplied separately.
Build the bridge with Rust, CMake, a CUDA toolkit and CUDA PyTorch 2.9.0.
The packed-domain and Stream 5 operators build on first use with nvcc, Ninja,
the PyTorch headers and, for distributed CUDA, the CUDA NVTX3 headers,
NCCL headers and the NCCL library.
To provision compute nodes from a build host, set `MGBS_CNOT_DOMAIN_LIBRARY`
and `MGBS_CNOT_STREAM5_LIBRARY` to the matching compiled operators. Build for
each target SM and the PyTorch/CUDA ABI used by the workers.

```bash
python integrations/cnot_scorer/build.py --source "$TQP" --build-dir "$PRIVATE_BUILD"
export PYTHONPATH="$PWD/integrations/cayleypy_native${PYTHONPATH:+:$PYTHONPATH}"
printf '[1,2,7,8]\n' > matrix.json
torchrun --standalone --nproc-per-node=2 -m multigpubeamsearch.cnot_cli matrix.json \
  --library "$PRIVATE_BUILD/target/release/libcnot103_native.so" --manifest "$FIXTURE/search-qhat.json" \
  --width 8 --shortlist 11 --max-depth 3 --mode qhat
```

For 16 GPUs on two nodes, run the following on **both hosts** with the same
reachable rendezvous address, ID, input and scorer. `torchrun` launches eight
processes per host. Each worker selects its node-local device. All 16 workers
exchange candidates and select a single global top B.

```bash
torchrun --nnodes=2 --nproc-per-node=8 --max-restarts=0 \
  --rdzv-backend=c10d --rdzv-endpoint=node0:29400 --rdzv-id=cnot-001 \
  -m multigpubeamsearch.cnot_cli matrix.json --library "$LIB" --manifest "$MANIFEST" \
  --width 257 --shortlist 11 --max-depth 12 --mode proposal
```

For Slurm, set `MASTER_ADDR` to a reachable hostname of the first allocated node
and `MASTER_PORT` to a free port. Launch one process per GPU. A per-task
CUDA mask exposing one GPU selects ordinal 0 even when `SLURM_LOCALID=7`.

```bash
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29400
srun --nodes=2 --ntasks=16 --ntasks-per-node=8 --gpus-per-task=1 \
  python -m multigpubeamsearch.cnot_cli matrix.json --library "$LIB" --manifest "$MANIFEST" \
  --width 257 --shortlist 11 --max-depth 12
```

The command writes JSON on rank zero. `status` is `solved`, `depth_limit` or
`exhausted`. A solved result includes `gates` and `length`. `by_rank` contains
one row per global rank with parent count, selected actions sent, actions sent
to another hash owner, distinct children scored, and frontier states received.
`path_routes` lists retained reduction edges before the terminal identity check.
The model explicitly rejects a state that falls outside its feature coverage.
The adapter reports the offending rank and feature and stops the collective search.

## Data movement and limits

- The CNOT domain uses packed `int64[batch,n,ceil(n/64)]` with zero tail bits,
  for 2–256 wires. A 144-wire matrix takes 3,456 bytes. Source rank and physical
  gate each occupy 16 bits in a route. The scheduler entry points accept up to
  128 ranks, the native permutation runner's current exchange limit.
- Before shortlisting, every parent is checked for a one-gate completion over the
  entire alphabet. `qhat` selects smaller action estimates, `proposal` selects
  larger raw logits. Both break head-score ties by physical ID.
- Selected children travel to the hash owner via the native Stream 5 NCCL count
  and variable-width payload exchange. CUB sorts row ordinals and tests full
  matrix equality. A hash collision changes placement only. The owner discards
  closed matrices and scores each remaining distinct child once.
- A distributed radix cutoff uses raw finite `V` and lexicographic gate history.
  It resolves ties before keeping exactly `min(B, eligible)`. Ranks replicate
  only winning metadata in normal use. Winner matrices travel to balanced slices
  of the next frontier. Empty ranks participate in collectives at every layer.
- Memory grows with the **local shortlisted layer and local closed set**. The
  implementation exchanges materialized matrices and performs layer barriers.
  Throughput at large `n`, width and node count needs workload-specific measurement.
  Enable `--audit` only on small fixtures: it also replicates all eligible rows.

For a bounded acceptance run, use `tools/verify_cnot.py --case trained` with
the same launcher, library and manifest plus `--output receipt.json`. Its
independent Python enumerator checks every eligible matrix, the ordered top B,
and the returned physical gates. `--case trained-deep --mode qhat` checks a
three-gate GL(4,2) input with B=2 and K=3. On two GPUs it requires both ranks
to expand, score and retain in the same saturated layer.
`--case wide` exercises n=144 with a diagnostic
scorer. The latter can also run with `--backend gloo` on CPU to check routing.
For a 16-rank occupancy test use `--case topology`, with B=17 and K=25.
Its second layer requires all ranks to expand, score, and keep states.
For the permutation runner's external launcher, see [cluster setup](CLUSTER.md).
