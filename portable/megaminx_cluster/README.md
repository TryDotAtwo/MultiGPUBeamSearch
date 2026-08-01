# Megaminx native cluster release

Compiler-free Ubuntu 22.04+ package for one Megaminx puzzle per SLURM job. Pick
the archive whose SM exactly matches every allocated GPU:

| GPU | Archive |
|---|---|
| T4 | `megaminx-sm75-linux-x86_64.tar.zst` |
| A100 | `megaminx-sm80-linux-x86_64.tar.zst` |
| RTX 30 | `megaminx-sm86-linux-x86_64.tar.zst` |
| RTX 40 / L4 | `megaminx-sm89-linux-x86_64.tar.zst` |
| H100 | `megaminx-sm90-linux-x86_64.tar.zst` |
| RTX PRO 6000 Blackwell 96 GB | `megaminx-sm120-linux-x86_64.tar.zst` |

There is no PTX, JIT, compiler, CUDA toolkit, Docker, architecture fallback, or
mixed-GPU mode. A mismatch fails during preflight before the solver starts.

Prerelease status: the committed registry currently has runnable measured profiles only
for 2x T4. The other architecture assets are build targets, not runnable claims;
they fail closed until a sweep for the exact GPU family, VRAM, world size, backend,
model class, and beam range is imported. The H100 command below is the intended
interface after its H100x4 profile evidence is added.

## Run

```bash
tar --use-compress-program=unzstd -xf megaminx-sm90-linux-x86_64.tar.zst
cd megaminx-sm90-linux-x86_64
cp cluster.env.example cluster.env
# Fill only the values required by your SLURM site.
./run.sh --gpus 0,1,2,3 --beam 1000000000 --puzzle 900 --reflect off
```

`--gpus`, `--beam`, and `--puzzle` are mandatory. Omitting `--puzzle` exits
before `sbatch` and asks for one puzzle id. Each invocation creates one run
directory and submits exactly one SLURM job. `torchrun` starts one process per
listed GPU.

Reflection modes:

- `--reflect off`: solve the original puzzle only.
- `--reflect after`: solve the original, then its reflected continuation.
- `--reflect only --original-solution solution.txt`: validate the supplied
  original solution and solve only the reflected continuation.

The requested beam is retained. Only the documented distributed-layout
alignment may increase the effective beam. The package chooses an exact
hardware/world-size/backend/model profile at the nearest half-up `log2` beam
power. Unknown or unmeasured tuples fail closed.

## Publication

Set the v2 Worker endpoint and claimed public author/cluster metadata in the
release configuration. After CPU replay succeeds, publication can be retried
without solving again:

```bash
python3 -m portable.megaminx_cluster.scripts.validate_and_publish \
  --run-dir runs/p900-... \
  --url https://YOUR-WORKER/v2/results \
  --poll
```

The client sends only schema-v2 whitelisted fields, writes the exact submitted
payload and receipt into the run directory, rejects redirects/private
endpoints, and polls the receipt's `/v1/submissions/<id>` status URL. GitHub is
the Worker's responsibility; cluster credentials never include a GitHub token.

## Release gates

Every asset has deterministic timestamps/ownership, `MANIFEST.json`, and
`SHA256SUMS`. The release checker requires exactly one `sm_XX` cubin and rejects
PTX, source/compiler files, symlinks, path traversal, secret-like content, and
an archive name that disagrees with its native SM.
## Automatic profile tuning

On a homogeneous allocation, submit the standalone adaptive tuner:

```bash
./autotune.sh --gpus 0,1,2,3,4,5,6,7 \
  --min-beam 30000000 \
  --time-budget 6h \
  --bfs-hash-budget-mib 256
```

The packaged calibration set uses puzzle IDs `900,950,1000` as short, medium,
and hard anchors. Override it with `--puzzles ID1,ID2,ID3`. The controller first
discovers the maximum stable beam, then applies deterministic successive halving.
It derives one Touch-BFS radius for the session from the puzzle move count,
stored hash width, and raw hash budget; for Megaminx Hash128 the default is radius 5.

Every probe is checkpointed under `autotune-runs/`. Rerun the same generated
SLURM job environment to resume; identity drift fails closed. Inspect
`leaderboard.tsv`, `profile_candidate.json`, and `registry.fragment.json`.
A fragment is runnable only when every anchor says `measured`; partial or failed
sessions remain `unverified`. Installing the fragment into `profiles/registry.json`
is an explicit local action and the tuner never publishes results automatically.
