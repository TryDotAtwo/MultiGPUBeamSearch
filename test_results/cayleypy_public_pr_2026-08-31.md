# CayleyPy native public-PR verification — 2026-08-31

This note records the reproducible acceptance evidence accumulated for
`TryDotAtwo/MultiGPUBeamSearch#4`. Initial CPU/wheel acceptance was performed at
commit `c40a64d0c33ee3c72d0395b7a28e1c419d4a7c32`; later pre-merge review fixes and
their exact-head CI are recorded below.

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
native SHA256: 53a68e2261e799aa5421f925c2a70bd23b40dae262092a3a34fea6026bafd6ad
native archive bytes: 8749964
CUTLASS revision: afa1772203677c5118fcd82537a9c8fefbcc7008
CUTLASS SHA256: e62fb320c2b61e7e0c7a1163c9c5a58f6dd86025adf7df4d124933152482997f
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

## 2026-09-01 pre-merge review fixes

Three later review findings were reproduced with regression tests and fixed:

- already-solved and zero-step searches now return before CUDA, model, build, or
  worker preflight;
- every model manifest/blob byte is rehashed after runtime preparation and
  immediately before worker launch;
- automatic export now compares the class-level Torch FX forward graph with the
  supported Pilgrim/ResMLPDistance inference schema instead of relying on finite
  numeric probes alone, and requires exact supported child module types without
  forward hooks.

The full Windows adapter suite after these fixes passed:

```text
174 passed, 2 skipped
```

The two skips remain the POSIX-only subprocess-permission tests. Review run
[`33452889518`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/33452889518)
at commit `a4cbf7e13f9a18ff97c73a3d6218926c892aef1c` passed all four
Ubuntu/Windows and Python 3.10/3.12 jobs. Every job built and
installed the wheel, exercised released CayleyPy, ran the adapter contracts and
tested the public hook. Ubuntu/Python 3.12 additionally repeated the real HTTPS
downloads, exact pin checks and offline cache reuse shown above. The final
merged-install smoke is recorded in the local merge evidence directory.

## Strict max_steps with touch-BFS

The final pre-merge review found that the native runner interpreted its depth
argument as forward-search depth and could then append a touch-BFS suffix. The
adapter now reserves the effective suffix radius from `max_steps`, clamps a
configured radius to `max_steps - 1`, passes only the remaining forward depth to
the runner, and rejects any returned path longer than the requested bound.
Dispatch, command/environment, metadata and parser regressions cover the full
API path. The complete Windows adapter suite passed:

```text
180 passed, 2 skipped
```

## Bias-free ResMLP input export

Exact built-in `nn.Linear` layers with `bias=False` are valid under the checked
native inference schema. The first ResMLP input layer now uses the exporter's
shared zero-bias conversion instead of dereferencing a missing state-dict key.
A full-process regression verifies the emitted FP16 blob, and the complete
Windows adapter suite passed:

```text
181 passed, 2 skipped
```

## Unambiguous native model semantics

The artifact validator now rejects duplicate, nested or JSON-escaped native
runtime manifest keys, preventing Python's JSON parser and the runner's literal
first-match parser from selecting different model configurations. Auto-export
also rejects a class-level `__call__` override that could bypass the verified
`forward` graph. Focused regressions reproduce both cases; the complete Windows
adapter suite passed:

```text
185 passed, 2 skipped
```

## Call dispatch and finite converted weights

The model preflight now rejects custom class/instance `_call_impl`, compiled
call dispatch and custom attribute lookup in addition to `__call__`. Artifact
validation decodes every FP16/BF16 blob and rejects NaN/Inf, including overflow
created when a finite FP32 tensor or folded BatchNorm value is converted to
FP16. Imported FP16/BF16 artifacts and full auto-export are covered. The
complete Windows adapter suite passed:

```text
191 passed, 2 skipped
```

## Residual shape fallback contract

Auto-export now validates the dimensions of every emitted Linear,
BatchNorm and LayerNorm tensor before launching the exporter. Both Pilgrim and
ResMLPDistance models with bottleneck or expanded residual blocks are classified
as unsupported with `NativeUnavailable`, so `backend="auto"` can use the
original CayleyPy search instead of failing on a mismatched exported artifact.
The regression asserts that no export subprocess starts for all four cases. The
complete Windows adapter suite passed:

```text
195 passed, 2 skipped
```
