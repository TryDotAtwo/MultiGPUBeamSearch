# Megaminx cluster autotuner portable verification — 2026-08-01

## Scope

Portable, compiler-free adaptive profile autotuner for native Linux archives.
The initial real qualification target remains 8x NVIDIA A100 40 GB (sm80); no
claim of a real A100 measurement is made in this report.

## Implemented gates

- mandatory homogeneous hardware-only preflight before profile discovery;
- one SLURM job and one controller, with torchrun workflow probes;
- maximum stable beam search from 30,000,000;
- deterministic adaptive successive halving;
- one formula-derived BFS radius per session using actual hash width and a
  default raw hash budget of 256 MiB (Megaminx Hash128 selects radius 5);
- replay/digest, CUDA/NCCL, scratch, timeout, and per-GPU 10% VRAM-margin gates;
- atomic resume checkpoints and append-only JSONL evidence;
- `measured` registry status only after 3 puzzles x 3 successful final repeats;
- exact fixed archive allowlist and native archive packaging.

## Verification

Command:

```text
python -m pytest tests/portable -q
```

Result:

```text
173 passed, 3 skipped in 6.01s
```

The three skips are Linux-bash syntax invocations unavailable from this Windows
host. The same shell payloads are staged into the Linux-only release workflow.

Additional checks:

```text
python -m compileall -q portable/megaminx_cluster
workflow_yaml_ok=1
git diff --check
changed_payload_secret_private_scan=clean
```

Focused Tasks 1-6 gate before packaging:

```text
57 passed, 1 skipped in 4.84s
```

Archive-builder/allowlist gate:

```text
23 passed, 1 skipped in 0.61s
```

## Real 8xA100 qualification command

After downloading and extracting the sm80 archive on the cluster:

```bash
./autotune.sh --gpus 0,1,2,3,4,5,6,7 \
  --min-beam 30000000 \
  --time-budget 6h \
  --bfs-hash-budget-mib 256
```

Retain the printed SLURM job ID and the complete `autotune-runs/...` directory.
Do not install `registry.fragment.json` unless every emitted anchor is
`measured`; incomplete sessions intentionally remain `unverified` and resumable.
