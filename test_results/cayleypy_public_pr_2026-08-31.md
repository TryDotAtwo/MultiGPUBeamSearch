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
state build (`STATE_LEN=8`, `STATE_STORAGE_LEN=16`). The final branch also adds
a host-only allocation guard to the current native runner; it does not change
GPU kernels, scoring, deduplication or beam selection. The adapter's default
download remains the earlier tested immutable source snapshot.

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

The final capability preflight also classifies Pilgrim hidden widths that are
not divisible by 8 and models exceeding the 1024-block runtime limit as
`NativeUnavailable`, before export. Custom `__getattr__` and `__getattribute__`
dispatch are both rejected. After these adjacent regressions, the complete
suite passed:

```text
200 passed, 2 skipped
```

Class-level and instance-level `state_dict` overrides, custom
`_save_to_state_dict` implementations and state-dict hooks are now rejected
before weight extraction. The regressions alter an input weight column not
covered by the finite probes and assert that no export subprocess starts. The
complete suite passed:

```text
203 passed, 2 skipped
```

The serialization/call checks now traverse the raw registered module tree, so
a custom root `modules()` implementation cannot hide a child residual block's
state hooks, `state_dict` implementation or call dispatch.

Touch-BFS host expansion is no longer allowed to inherit the native runner's
unlimited default. `NativeOptions.touch_bfs_max_entries` defaults to 1,048,576,
is validated as a positive integer, is always passed as
`BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES`, and is recorded in runtime metadata.
The complete suite passed:

```text
212 passed, 2 skipped
```

## Private launch artifacts and bounded host allocation

A manually supplied `NativeModel` no longer leaves native workers reading the
caller's mutable weight directory. Preparation validates that source, copies
only the exact runtime manifest/blobs to the search run directory, validates the
copy against the original content hash, and the existing prelaunch check
rehashes the private copy. Regressions cover source mutation after preparation,
mutation during copy, exclusion of unrelated caller files, and direct tampering
with the private snapshot.

Touch-BFS preparation now computes the duplicate-free geometric upper bound for
the effective radius and rejects it before CUDA/build when it exceeds
`touch_bfs_max_entries`. This protects users of the immutable pinned native
snapshot. The current C++ runner additionally clamps both solved-neighborhood
and Stream2 suffix next-frontier `reserve` calls to the remaining finite budget
and avoids multiplication overflow. The complete Windows adapter suite passed:

```text
215 passed, 2 skipped
```

The modified runner was configured and built from a read-only checkout in the
existing CUDA 12.8/CUTLASS container with `BEAM_CUDA_ARCHITECTURES=75`; NVCC
compiled `tools/production_runner.cu` and linked the `production_runner` target
successfully. No GPU was exposed to this compile-only container run.

`pip wheel --no-build-isolation --no-deps` produced
`cayleypy_native-0.1.0.dev0-py3-none-any.whl` with SHA256
`cf4888f72a46696224939d107bb5396cfd36958029b91385981229583d183a51`.
The wheel contains `dist-info/licenses/LICENSE`, was installed into an isolated
target, and `validation/wheel_smoke.py` passed against both released CayleyPy
0.1.0 (`public_hook=false`) and the companion checkout (`public_hook=true`). In
both cases the installed adapter produced and replayed CPU fallback path `[2]`.

## Process-global PyTorch hook boundary

Auto-export rejects nonempty PyTorch global forward and pre-forward hook
registries before inspecting a model, then checks them again after the semantic
probes. These hooks are not stored on any child module and cannot be represented
in the native artifact. Regressions use both public global registration APIs,
assert rejection before the exporter starts, and remove the hooks in `finally`
so no process state leaks between tests. The complete suite passed:

```text
217 passed, 2 skipped
```

## Private runner launch snapshot

The adapter no longer verifies a shared prepared runner and later opens that
same mutable path in the worker. Every nontrivial search copies the runner into
its fresh run directory, checks the private copy against the pinned build
SHA256, and passes only that private path to direct or `torchrun` workers. The
regression replaces the shared runner immediately after snapshot creation and
confirms that the launched private executable retains the verified bytes. The
complete Windows adapter suite passed:

```text
217 passed, 2 skipped
```

## Early touch-BFS capability rejection

Graph-specific touch-BFS packing and geometric entry bounds are now checked
immediately after the graph contract and effective depth budget are known. Both
ordinary `graph.beam_search(...)` dispatch and explicit `prepare_native(...)`
reject unsupported graph/radius combinations before CUDA inspection, model
export, native build, or worker launch. Regressions make every later preflight
entry point fail if called. The complete Windows adapter suite passed:

```text
219 passed, 2 skipped
```

## Lazy run-directory creation and fallback cleanup

`backend="auto"` now completes graph and CUDA/runtime capability checks before
creating a per-search cache directory. A CPU-only or otherwise unsupported
runtime therefore cannot fail because the native cache is read-only or
unavailable. Cache-creation errors become explicit preflight fallback reasons,
and later unsupported model/build preparation removes its own fresh UUID run
directory before calling the original CayleyPy search. Strict `native` mode
retains created artifacts for diagnosis. Regressions cover all three paths; the
complete Windows adapter suite passed:

```text
221 passed, 2 skipped
```

## Exact manifest identity

The native artifact SHA256 now starts from the exact `manifest.json` bytes read
for validation, rather than a canonicalized Python object. A JSON-equivalent
byte edit therefore invalidates a prepared private snapshot. The adapter also
rejects escaped runtime string values whose Python interpretation differs from
the native runner's literal text parser. Both cases were reproduced before the
fix and are covered by regressions; the complete Windows adapter suite passed:

```text
224 passed, 2 skipped
```
