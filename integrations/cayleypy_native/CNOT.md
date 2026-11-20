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
Errors while reading scorer metadata or encoding the admission configuration are
also reported collectively, before ranks exchange input fingerprints. Ranks can
reuse the process group after such a coordinated admission error.
Local errors while evaluating a one-gate goal are reported to all ranks through
the existing goal exchange; a healthy group can be reused after the error. This
does not recover a failed device or NCCL communicator, or an error that prevents
the status exchange itself.

By default, `search_cnot` uses the Rust packed-input and construction replay
validator when the matching extension loads. `validation="auto"` falls back to
Python when the extension is absent or cannot load; `validation="python"`
forces the reference validator, and `validation="rust"` fails closed. All
ranks must select the same validator.

The default `engine="auto"` uses the Rust search controller and rank-local
stage owner when the matching capability is available. It falls back to the
reference path for an absent or unloadable extension; `engine="python"`
forces that path and `engine="rust"` fails closed. The current Rust stage
owner dispatches packed-matrix operations to existing C++ ATen/CUDA operators.
PyTorch owns the tensors, and arbitrary Python scorers remain supported.

## Rust-owned CNOT control collectives, 2026-11-18

When all ranks have the matching PyO3 module and a Stream 5 library advertising
both the owned-exchange ABI and the additive int64-collectives ABI, the Rust
Stream 5 communicator carries CNOT `sum`, fixed `all_gather`, and history-row
`broadcast` calls as well as the existing variable-record exchange. Capability
selection is collective: if any rank lacks either ABI, all ranks retain the
previous Torch collective path. Tensor storage remains PyTorch-owned; the FFI
borrows its device addresses on the current CUDA stream and records that stream
before NCCL reads or writes the buffers. NCCL unique-ID bootstrap and the diagnostic fallback for older checked
stage APIs remain on the caller's default process group. Rust CNOT goal failures
and the checked v2 stage path gather UTF-8 lengths and padded byte tensors via
the current `World` transport. This transfers diagnostic dispatch only, not
the full CNOT search or checkpoint scorer.

With the matching PyO3 extension, public `TQPScorer` uses a Rust-owned
scorer-library/context handle. Rust performs the existing `inference_batch`
scalar-ABI calls in the same order and validates the same coverage and finite
scores. The Python scorer remains the fallback for older or absent extensions.
PyTorch still owns the row, score and status tensors; native-sys checks their
dtype, device, shape, contiguity and disjoint ranges before borrowing addresses.
This transfers scorer control and lifetime ownership without changing model
kernels, output order, precision or search decisions.

`experimental_native_batch=True` remains opt-in. It submits one borrowed packet
through `mgbs_tqp_score_many`, which retains the same inference sub-packet size.
Without that symbol, explicit selection fails before the checkpoint is opened.
The buffer contract and validation limits are described in
`integrations/cnot_scorer/README.md`.

## Data movement and limits

- The CNOT domain uses packed `int64[batch,n,ceil(n/64)]` with zero tail bits,
  for 2–256 wires. A 144-wire matrix takes 3,456 bytes. Source rank and physical
  gate each occupy 16 bits in a route. The scheduler entry points accept up to
  128 ranks, the native permutation runner's current exchange limit.
- Each search copies its input to one owned contiguous tensor on the same device
  before validation. Its host validation bytes and the search frontier refer to
  that snapshot, even if the caller later changes the original. The device copy
  is at most 8192 bytes. Concurrent writes during initial capture must be ordered
  by the caller; copying is not an atomic cross-stream operation.
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
