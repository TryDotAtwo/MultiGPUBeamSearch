# Cluster Branch Migration Verification

## Preserved source

- GitHub source branch: `codex/megaminx-native-cluster-release`
- Source and initial canonical tip: `2d5f978fcfc1f018813fc49e9d9aeec32ea5e22c`
- Canonical branch: `cluster/production`
- The canonical branch was created directly at the source object id; source history was not rewritten.
- Detached cluster-fix commit `847fe2a46fcf2135155ca9c368f5813f73ab9f75` is already contained in the source branch.

## Shell syntax gate

All 20 tracked `*.sh` files under `hpc/` were checked with `bash -n` inside
`gpu-dev-cutlass-nsight:2026-05-24`. Result: exit code `0`.

No SLURM submission, network publishing, or production solve was launched.

## Native build gate

- Host GPU: NVIDIA GeForce RTX 3070 Laptop GPU
- Driver: `572.70`
- Container CUDA: `12.8.1`
- CUTLASS: `/opt/cutlass`
- Target: `sm_86`

Fresh build targets:

```text
production_runner
contract_tests
```

Result: build exit code `0`; `contract_tests=pass`.

## Dirty root-checkout preservation

Tracked changes under `hpc`, `tools`, `cuda`, and `src` were saved without
application to the canonical branch:

- Patch: `test_results/root_dirty_cluster_stream_patch_2026-08-09.patch`
- Patch SHA-256: `a11f37fb1a59a9adb55979a06ddc8c0c1cd9a2137b8272b08fee139917dd952d`
- External archive: `D:/100XH100-backups/root-untracked-cluster-stream-tools-2026-08-09.zip`
- Archive files: `9`
- Archive SHA-256: `5a9269a525b91efcf6c6789ce4db25551ba654c6dc69b7edbe9b00e3e00c61fb`

These changes require a separate content review and verification before they
may enter `cluster/production`, `stream1`, or `kaggle/notebooks`.

Live 8xA100/SLURM behavior was not rerun during branch migration. Existing
published history and evidence remain preserved; local gates do not substitute
for a future live cluster acceptance run.
