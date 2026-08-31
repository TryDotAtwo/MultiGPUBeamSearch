# CayleyPy native public-PR verification — 2026-08-31

This note records the reproducible acceptance evidence for
`TryDotAtwo/MultiGPUBeamSearch#4`. The adapter code under test was commit
`c40a64d0c33ee3c72d0395b7a28e1c419d4a7c32`; the later review fix only clarifies
the already implemented compile-time state ABI and adds this tracked report.

## Local CPU and installation acceptance

The full adapter suite was run on Windows against the companion CayleyPy hook:

```text
python -B -m pytest integrations/cayleypy_native/tests -q -p no:cacheprovider
169 passed, 2 skipped
```

The two skips are POSIX-only subprocess-permission tests. A wheel was then built,
installed into an isolated target directory, and exercised with
`validation/wheel_smoke.py` against both released CayleyPy 0.1.0 and the
companion hook checkout. Both runs imported the installed wheel, selected the
expected CPU fallback, found a path, and independently replayed it:

```text
wheel_smoke=pass
fallback_reason=native backend requires Linux
path=[1]
path_replay=pass
```

## Public CI

GitHub Actions run
[`33415924639`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/33415924639)
passed all four jobs:

- Ubuntu, Python 3.10: 166 passed, 5 skipped; 33 hook-dispatch tests passed.
- Ubuntu, Python 3.12: 166 passed, 5 skipped; 33 hook-dispatch tests passed.
- Windows, Python 3.10: 165 passed, 6 skipped; 33 hook-dispatch tests passed.
- Windows, Python 3.12: 165 passed, 6 skipped; 33 hook-dispatch tests passed.

The Python 3.12 Ubuntu job also performed real HTTPS downloads, SHA256 checks,
safe extraction, and complete offline cache reuse for both immutable source
archives:

```text
native revision: a1db0e6d9bb5458c8a842b37dfa99572d3025667
native SHA256: 53a68eb40275bc0bd50669a5f98e4701cc9f6d1cb12a5fac92506771b41ed073
native archive bytes: 8749964
CUTLASS revision: afa177415c970c797f700dbf4c73032d3874bcd9
CUTLASS SHA256: e62fb364e2524c747e5bda3a5fcb57eb22b604d980f1e9c03553c1f6612c9de8
CUTLASS archive bytes: 31040140
offline reuse: pass
```

The companion CayleyPy PR
[`cayleypy/cayleypy#207`](https://github.com/cayleypy/cayleypy/pull/207)
passed its 10 Linux, Windows, macOS, docs, lint, type-check and coverage jobs at
commit `d74c3e1b1e80d7c89910d8c81bf21fded924299b`.

## Native GPU boundary

The pinned native source predates the adapter and had already passed LRX8 and a
real N88/G24 Tetraminx acceptance run on two Tesla T4 GPUs, including path
replay, multiple moves, prepared-model reuse, exhausted budgets and a small
state build (`STATE_LEN=8`, `STATE_STORAGE_LEN=16`). The adapter does not change
CUDA/native algorithm files.

A fresh public-PR two-T4 rerun was attempted through Kaggle, but SaveKernel,
owned-kernel reads and kernel listing all returned HTTP 403 through both the
configured proxy and host session. No new GPU run id was obtained, so this note
does not relabel the earlier native run as fresh validation of the Python hook or
source downloader. `validation/public_gpu_smoke.py` remains the reproducible
two-GPU acceptance command for a supported Linux CUDA host.
