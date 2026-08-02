# Megaminx native cluster: NCCL CUDA-line pin

Date: 2026-08-02
Branch: codex/megaminx-native-cluster-release

## Live evidence

- Eight A100 40 GB preflight passed on SM80 with NVIDIA driver major 550.
- The previously published archive contained `NCCL version 2.30.7+cuda13.3`.
- Multi-rank startup failed in `ncclCommInitRank` with `unhandled cuda error`.

## Fix

- Pin `libnccl2` and `libnccl-dev` to `2.27.7-1+cuda12.4`.
- Verify both installed Debian package versions exactly.
- Verify staged `lib/libnccl.so.2` contains `NCCL version 2.27.7+cuda12.4` before packaging.
- Retain the existing runtime-closure `ldd` gate.

## Verification

- Focused release automation: 15 passed.
- Full portable suite: 189 passed, 3 skipped.
- `git diff --check`: passed.

The final runtime proof still requires rebuilding the public assets and rerunning the clean-cluster SM80 archive on the user-operated cluster.