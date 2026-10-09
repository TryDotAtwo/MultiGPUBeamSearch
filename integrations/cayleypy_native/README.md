# MultiGPUBeamSearch Python library

## Public package name

Install the multigpubeamsearch wheel and use `import multigpubeamsearch`.
The historical `cayleypy_native` imports forward to the same objects for compatibility.
This is the public Python namespace of MultiGPUBeamSearch; no import alias is needed.

## Automatic API (development update, 2026-10-06)

Install from the complete matching repository with
`pip install ./integrations/cayleypy_native`, or install its built wheel. The wheel
ships its matching native source; CMake/Ninja, CayleyPy and PyTorch are package
dependencies. First search obtains checksum-pinned CUTLASS and builds/caches a
runner for the graph shape and visible CUDA architectures. Linux, an NVIDIA
driver, CUDA toolkit/nvcc and a C++ compiler are platform prerequisites; failures
are explicit. No driver installation or model training happens during search.

```python
from cayleypy import CayleyGraph, PermutationGroups
from multigpubeamsearch import enable_native, NativeOptions

graph = CayleyGraph(PermutationGroups.lrx(8), device="cuda")
enable_native(NativeOptions(num_gpus=2))
result = graph.beam_search(
    start_state=[1, 0, 2, 3, 4, 5, 6, 7],
    beam_width=100_000,
    return_path=True,
)
print(result.path_found, result.path_length, result.get_path_as_string() if result.path_found else None)
```

No predictor means exact Hamming distance to the graph's central state, encoded
as a fixed untrained native MLP. An unchanged `Predictor(graph, "hamming")` also
works; a supplied learned predictor is never silently replaced by Hamming.
Supported learned model exporters remain explicit; arbitrary Python callables
are not automatically converted to native code.

Search defaults to strict `backend="native"` after `enable_native`.
`backend="torch"` calls ordinary CayleyPy; `backend="auto"` explicitly permits
pre-launch capability fallback. `beam_mode` keeps CayleyPy's algorithm meaning;
our adapter supports simple search. Importing the adapter alone changes nothing.

Inference implementation is selected automatically: T4 uses LibTorch; newer
supported GPUs use CUTLASS. A mixed set containing T4 uses LibTorch for the whole
job. `inference_backend` is an optional diagnostic override, separate from search
`backend`. MLPs and ensembles calibrate the selected executor when `autotune=True`:
native MLPs measure CUDA Graph replay, LibTorch MLPs measure the eager path,
and ensembles measure their actual ordered readout. Other registered families
retain conservative profiles until they have an executor-specific probe.
Native VRAM planning still admits the requested beam. Requested GPU count is never
silently reduced. Native allocation alignment may increase effective beam width;
inspect `result.native_metadata` for requested/effective widths, executor,
profile, build/model identities and replay validation.

## Ensembles and inference-first calibration (development)

The current development source is on branch
[`codex/fast-autotune-20261009`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/tree/codex/fast-autotune-20261009).
Install that matching compact checkout with
`pip install ./integrations/cayleypy_native`; the default branch is not a release
of this ensemble/autotune update.

```python
from multigpubeamsearch import NativeEnsemble, NativeOptions, enable_native

enable_native(NativeOptions(num_gpus=2))
# model_a/model_b/model_c are supported models or graph-bound NativeModel artifacts.
predictor = NativeEnsemble((model_a, model_b, model_c), (.6, .3, .1))
result = graph.beam_search(start_state=start, predictor=predictor,
                           beam_width=1_000_000, return_path=True)
```

The ordered ensemble has no fixed two-model limit. Coefficients are explicit
and are not silently normalized. Supported FP16 heads accumulate in FP32;
clamping and score quantization happen after the last head. On SM80 and newer,
the final GEMM performs the weighted accumulation in its CUTLASS epilogue.
Backbones currently use LibTorch; importing this API does not compile arbitrary
Python neural architectures. The public artifact adapter currently accepts MLPs;
additional native C++ families still require public adapter registration.

With autotuning enabled for an MLP or ensemble, Stream1 first measures inference batches on
all selected GPUs. The bounded sweep includes power-of-two batches, the actual
user/beam cap and a local refinement around the coarse winner. Selection uses
the slowest rank, repeated timings and confidence intervals; it finds the best
verified candidate within its time budget, not a guaranteed global optimum.
The downstream stage freezes that inference batch and measures isolated native
services at the exact admitted capacities while varying outer batch, rings,
shards and sort buffers. Complete-depth verification is optional.
For an ordinary MLP, the native row budget remains fixed too; increasing it
would change the inference batch that was just calibrated.
Every candidate must pass native memory admission for the actual requested beam.
Default calibration budgets are 180 seconds for inference and 600 seconds for
the pipeline. No profile measured at 65,536 or 10M states is transferred as
certification of a larger requested frontier. A large optional full-frontier
check may require increasing `calibration_pipeline_seconds`.
Inspect `result.native_metadata['profile']` for the selected parameters and the
measured frontier scope. Isolated component probes are a scheduling proxy,
not a measured complete step or a global optimum. Small finite graphs can
prevent a unique legal fixture for the optional complete-step measurement.

`plan_cluster([8] * 16, beam_width)` computes a 128-rank layout and alignment.
It does not launch multiple nodes or prove hardware performance; every actual
rank must independently pass native memory admission. Current hardware evidence
for this development is from two and eight RTX 3060 GPUs; the exact workload
and measured scopes are recorded in the linked evidence branch below.

Raw receipts and SHA256-verified archives are on
[`codex/fast-autotune-evidence-20261009`](https://github.com/TryDotAtwo/MultiGPUBeamSearch/tree/codex/fast-autotune-evidence-20261009/test_results/fast_autotune_20261009).

The remainder records the older explicit setup API and historical validation.
Its default-auto and pinned-native setup descriptions do not describe the new
automatic entry point above. New execution evidence is recorded separately under
`test_results/cayleypy_api_20261006`; the old T4 acceptance is not evidence for
these updated bytes.

An optional Python wrapper around this repository's native beam search. Enable
it once, then keep calling `graph.beam_search(...)`. CayleyPy supplies the graph
and public result API; the supported native path runs our existing CUDA/NCCL
algorithm as a local subprocess. The public CayleyPy backend hook is used when
available; released CayleyPy 0.1.0 also works through an explicit reversible
in-memory wrapper. Neither library is patched on disk.

This first integration has passed end-to-end validation on **two real Tesla T4**:
LRX8, explicit prepared-model reuse, and a real N88/G24 Tetraminx artifact. Native
paths were independently replayed, including multiple moves and exhausted
budgets. This is integration evidence, not a trained-model Torch speedup claim.

## Installation

Install into your own Python 3.10+ environment. This package is installable from
Git; no PyPI release is implied:

```bash
python -m pip install "git+https://github.com/TryDotAtwo/MultiGPUBeamSearch.git#subdirectory=integrations/cayleypy_native"
```

For a checkout, `python -m pip install ./integrations/cayleypy_native` also works.
A built wheel has no dependency on the checkout's location or existence.

Native execution requires **Linux, CUDA-capable PyTorch, NCCL headers/library,
a CUDA toolkit with nvcc, a compatible C++ compiler and CMake**. Install the
CUDA-enabled PyTorch build appropriate for your system using its official
installation instructions; a CPU-only wheel cannot run native search. PyTorch's
installed NVIDIA NCCL wheel or a system NCCL installation can supply NCCL.
FP16 needs SM75+ (e.g. T4); BF16 needs SM80+. Python build helpers can be installed
with `python -m pip install cmake ninja`. This does not install nvcc or a driver.
Windows, macOS and CPU-only calls in `auto` mode use ordinary CayleyPy and explain why.

Prepare public source dependencies explicitly once:

```python
from cayleypy_native import NativeOptions, setup_sources

options = setup_sources(options=NativeOptions(devices=(0, 1)))
```

`setup_sources()` downloads immutable native and CUTLASS archives to
`~/.cache/cayleypy-native/sources/`, checks their pinned SHA256, safely extracts
them, and verifies cached source bytes on reuse. It returns `NativeOptions`;
it does not build, search, install packages, download weights or modify the
environment. Downloads happen only in this explicit call. The native pin is the
tested `a1db0e6d9bb5458c8a842b37dfa99572d3025667`, not moving `main`.

Use `setup_sources(offline=True)` to require an existing cache, or provide your
own `source_dir` and `cutlass_dir` for a network-free installation. Explicit
directories are caller-managed, not claimed to match the pinned snapshot.
`cache_dir=...` selects another local volume. Corrupt/incomplete caches raise an
error; setup never silently overwrites them. CUTLASS retains its own license,
included in its downloaded source archive. This repository's MIT license covers
the adapter and the native source snapshot used by this integration, not weights
or third-party dependencies.

## Keep the existing call

```python
from cayleypy_native import enable_native, setup_sources

options = setup_sources()  # Explicit setup; omit when only exercising CPU fallback.
enable_native(options)

# graph, start_state and predictor are the objects from your existing program.
result = graph.beam_search(
    start_state=start_state,
    predictor=predictor,
    beam_width=2**20,
    max_steps=100,
    return_path=True,
)
print(result.path_found, result.path_length)
if result.path_found:
    print(result.get_path_as_string())
```

Automatic export accepts a loaded, unmodified `Pilgrim` BatchNorm model or
`ResMLPDistance` LayerNorm model matching the native export schema, either directly
or inside the standard CayleyPy `Predictor`. Models must be in evaluation mode.
The adapter checks the complete class-level forward graph against the supported
schema before exporting. Linear, normalization, activation, embedding and container
children must be the exact supported PyTorch module types, without forward hooks.
The model must retain standard `nn.Module` call and attribute dispatch, including
`_call_impl`, and standard state-dict serialization without hooks.
A supported `nn.Linear(..., bias=False)` is exported with an explicit zero-bias
blob, preserving its output in the fixed native artifact schema.
A customized model uses the original CayleyPy search in `auto` mode and is rejected
in `native` mode.
Use the standard `Predictor` wrapper when torch fallback is expected; a raw
module is forwarded unchanged on fallback and must satisfy upstream's interface.
An upstream CayleyPy MLP, an arbitrary callable, a custom Predictor, or the default
Hamming heuristic is **not** silently translated into another model; it uses the
original CayleyPy search in `auto` mode.

For existing native weight artifacts:

```python
from cayleypy import Predictor
from cayleypy_native import NativeModel

predictor = NativeModel.for_graph(
    graph,
    weights_dir="/work/weights/exported-model",
    fallback=Predictor(graph, "hamming"),  # Optional, explicitly chosen fallback.
)
result = graph.beam_search(
    start_state=start_state,
    predictor=predictor,
    backend="native",  # Require our search; do not fall back.
    beam_width=2**20,
    max_steps=100,
    return_path=True,
)
```

`for_graph` declares the artifact's graph association. Use it only for weights
trained for this exact ordered graph, center and label encoding. It cannot prove
training provenance. If the manifest already has a graph hash, it must match.
NativeModel without `fallback` cannot execute an unsupported case through torch.
The fallback may be the original predictor instead of Hamming; its selection is
explicit and can affect search quality.
Runtime manifest keys must be literal, unique and top-level so the adapter and
the native runner cannot interpret the same artifact differently.
Runtime string values read by the native text parser must also use literal,
unescaped JSON strings. The artifact SHA256 includes the exact manifest bytes,
so even JSON-equivalent serialization changes invalidate a prepared snapshot.
Every FP16/BF16 weight blob is decoded and checked for finite values before use;
this also catches overflow introduced by FP16 conversion or BatchNorm folding.
Auto-export also requires PyTorch's process-global forward/pre-forward hook
registries to remain empty while inference semantics are validated; global
hooks cannot be represented by the native artifact.
For a manually supplied `NativeModel`, the adapter copies only the validated
manifest and required blobs into the private directory for that search. The
copy must retain the validated content hash, and only that private snapshot is
passed to native workers. Later changes in the caller's weight directory cannot
change the model being executed.

For a Q model with one output per generator, also pass `use_child_scores=True`.
Scalar models score child states. Generator order is preserved in both cases.
Native deduplication, scoring, rounding and beam management stay native; identical
frontiers or solution lengths to upstream are not promised.

`max_steps` is a strict bound on the complete returned path, including a
touch-BFS suffix. When `touch_bfs_radius` is larger than the available budget,
the adapter reduces the effective radius to at most `max_steps - 1` and reserves
the remaining steps for the native forward search. It also rejects malformed
native output whose complete path exceeds the requested bound. The configured
and effective budgets are recorded in each run's metadata.

Touch-BFS host expansion is capped at 1,048,576 entries by default. Set a
different finite positive uint64 budget with
`NativeOptions(touch_bfs_max_entries=...)`. Before CUDA inspection, model export
or a build, both normal search and `prepare_native(...)` require the
duplicate-free geometric worst case for the effective radius to fit this budget.
This conservative check may reject a neighborhood that would fit only because
many generated states coincide. The native builder also caps each next-frontier
reservation to the remaining budget. The entry budget and accepted worst-case
size are recorded in run metadata.

## Backend selection

| Selector | Behavior |
| --- | --- |
| `auto` (default) | Prepare a supported native run; otherwise warn with a concrete reason and call the saved original method with the original arguments. |
| `native` | Reject unsupported graph/model/runtime/options with `NativeUnavailable`. |
| `torch` | Call the original method without probing CUDA or native compatibility. |

Already-solved states and zero-step budgets return directly after graph/argument
validation. They do not probe CUDA, inspect a model artifact, compile, or launch a
worker, and they do not create or require a writable cache. Their metadata reports
`run_dir=None` because no runtime artifacts exist. These deterministic boundary
cases therefore also work on a CPU-only host.

In `auto` mode, CUDA/runtime viability is checked before creating a cache run
directory. An unavailable cache is itself a preflight fallback reason, and a
later unsupported model/build removes its fresh private run directory before the
original CayleyPy method is called. Strict `native` failures retain any created
artifacts for diagnosis.

Importing the package does not activate it. `enable_native(...)` registers `auto`
and `native` using CayleyPy's public backend hook, without replacing
`CayleyGraph.beam_search`. `disable_native()` restores the previous default and
unregisters these names. Existing name collisions are rejected. Changes made by
another integration are not silently overwritten. Configure before launching
application threads; the process default is resolved at call time.

Older CayleyPy releases without the hook use a reversible process-wide wrapper;
in that compatibility mode already captured methods retain their session.
The PyPI release and the newer source API both report `0.1.0`, so support is
detected from capabilities rather than the version string. Q-model
`use_child_scores` requires the newer upstream API.

With the public hook, `graph.beam_search(backend="torch", ...)` accepts only
upstream arguments and an upstream predictor. For `NativeModel.fallback`
translation or `native_options` with the Torch selector, use the standalone
adapter function below.

For a single call without changing the class:

```python
from cayleypy_native import beam_search
result = beam_search(graph, native_options=options, backend="native",
                     start_state=start_state, predictor=predictor,
                     beam_width=2**20, max_steps=100, return_path=True)
```

Set `warn_on_fallback=False` in `NativeOptions` only if you intentionally want
silent preflight fallback. Compilation failures, corrupt artifacts, native worker
crashes, OOM, timeouts, malformed output and replay failures raise errors; they
never restart the search with another algorithm. `path_found=False` means no
solution was found within this run's budget, not that the state is unreachable.

## Initial compatibility boundary

| Component | Supported now |
| --- | --- |
| Graph | Permutation generators, one start state, target equal to graph center. |
| State | 1–120 logical positions; integer labels 0–127; matching center/start multiset. Repeated colors are allowed by the graph boundary. |
| Moves | Ordered gather permutations, 1–255 generators; touch-BFS additionally requires at most 32. |
| State specialization | Build uses `STATE_LEN=n`, explicit move count, and storage `ceil((n+4)/16)*16`; small states do not retain a fixed 128-byte stride. |
| Model runtime | Native MLP artifact, scalar or one output per move, FP16/BF16, supported BatchNorm-folded/LayerNorm schema. FP16 requires SM75+, BF16 SM80+. Current Q GEMM requires the move count divisible by 8; scalar output does not. |
| MLP dimensions | `num_classes >= max(state_len, max_label+1)`, 1–1024 residual blocks, hidden widths divisible by 8, hidden1 >= hidden2, and square hidden2 residual blocks. This can rule out an otherwise supported colored graph. |
| Search options | Simple mode, global beam width, strict maximum path length, return path. Native touch-BFS uses `NativeOptions(touch_bfs_radius=..., touch_bfs_max_entries=...)`; its suffix stays inside `max_steps` and host expansion has a finite entry cap. |
| Device placement | Local Linux CUDA devices, one native rank per selected GPU, up to 128. Graph CUDA devices are used by default; explicit `devices=(0,1)` overrides them. |
| Not yet supported natively | Matrix graphs, arbitrary models/tokenizers, PieceTransformer artifacts, custom destination, advanced/history taboo policy, reusing an upstream BfsResult, existing torchrun ranks, multi-node launch. |

There is no puzzle-name whitelist. Representation/model/runtime capabilities
decide support. A compatible graph alone does not imply a compatible model.

## Builds, devices and evidence

Source and CUTLASS paths come from explicit `setup_sources()` or `NativeOptions`.
A build is cached by working-tree source
bytes (including dirty source), CUTLASS bytes, state/move shape, GPU architectures,
backend and compiler identities. Dimensions are passed explicitly to CMake, so an
old inferred shape cannot be reused accidentally. Native build internals are left
unchanged.

Alternatively, set `runner_path` to a trusted existing executable. Its sibling
`native-build.json` must match schema version 1, shape, backend, CUDA architectures
and the binary SHA256, and must retain the exact NCCL library path and SHA256 used
for the build. The wrapper-generated file is the reference format. Do not
manufacture metadata for an unverified binary.

Device indices are relative to PyTorch's current visible devices. An inherited
`CUDA_VISIBLE_DEVICES` mapping is preserved when selecting a subset. Each child
gets a private working directory, inputs, history and rendezvous files. Per-rank
stdout/stderr are redirected separately and merged in a fixed order, avoiding
interleaved result/configuration records. Inherited
native beam/profile/rank/repair/publication settings are filtered. No notebook
kernel or parent environment is modified by the worker launch.

The caller's GPU tensors/model stay resident while the native subprocess runs.
They can reduce the VRAM available to the native beam compared with a standalone
runner. First-call compilation, model export and process startup also have costs;
end-to-end performance must include them. This version does not provide zero-copy
CUDA tensor sharing or a persistent native worker.

Each nontrivial run is stored below `<cache_dir>/runs/<id>/` with model export, build and
native logs as applicable. These files contain states, paths and possibly model
weights; they are local artifacts and are not automatically deleted or uploaded.
The cache can be large. Keep it on a suitable local volume.

Native success returns `NativeBeamSearchResult`, a subclass of CayleyPy's
`BeamSearchResult`, with `backend == "native"` and `native_metadata`. Every path is
replayed with the original ordered generators even when `return_path=False`.
Metadata includes graph/model hashes, requested beam, observed effective beam,
devices, timings and log paths. An unavailable effective width stays `None`;
the adapter does not invent it. `debug_scores` is empty because native layer
scores are not currently exported through CayleyPy's diagnostic interface.

For tiny beams the adapter reduces inference microbatch when the native default
cannot fit its intermediate buffers. It does not set shard top-k or change the
requested global beam. Native may round the effective beam upward: the two-T4
LRX8 check requested7 and observed8192. Always use the recorded effective width
when comparing workloads.

## Verification

From a source checkout, run CPU tests against the pinned development API:

```bash
python -m pip install "git+https://github.com/cayleypy/cayleypy.git@28f3841b34009ea8d51bb36eece3dd0be757a145"
python -m pip install -e './integrations/cayleypy_native[test]'
CUDA_VISIBLE_DEVICES=-1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -B -m pytest integrations/cayleypy_native/tests -q
```

`examples/solve_lrx.py` is an offline smoke example. Without compatible weights it
exercises the explicitly reported torch fallback; providing native weights and a
Linux build runtime enables a strict native attempt. The tests use labelled fake
workers for process protocol checks, not as substitutes for a CUDA performance or
solve-quality result. See [VALIDATION.md](VALIDATION.md) for the measured scope.

Run the reproducible two-GPU smoke from the repository root after installing the
package (do not use an editable install for an installation acceptance run):

```bash
python integrations/cayleypy_native/validation/public_gpu_smoke.py \
  --output-dir ./test_results/cayleypy-native-gpu \
  --cache-dir ./test_results/cayleypy-native-cache --require-public-hook
```

This downloads pinned public sources and generates its own synthetic models;
no competition, pretrained weights, Kaggle account or private dataset is needed.
Omit `--require-public-hook` when checking the legacy CayleyPy adapter path.
Logs and generated weights stay local. The smoke checks both GPUs, path replay,
prepared calls, exhausted budgets, source-cache reuse and two state sizes.

CPU CI separately checks an installed wheel with released CayleyPy 0.1.0 and
the development API. The public-hook integration and the public GPU acceptance
harness, including cache-free host shortcuts, are covered by tests when that API
is installed. Cross-call graph/model identity is compared only for cases that
launch native workers; deterministic host shortcuts are checked separately.
Native CUDA tests require a real compatible machine; CPU protocol fixtures are
not a GPU performance result.

## Prepare explicitly for repeated searches

### Inference-first automatic sizing

Native automatic calibration first selects a numerically verified inference
microbatch using repeated slowest-rank measurements. Cache identity includes the
exact requested beam; a 10M profile does not certify 100M. It then asks the native
planner to admit that beam on every selected GPU, measures Stream3, concurrent
Stream4 jobs, final union and NCCL transport at those exact buffer capacities,
and derives a bounded shortlist from arrival/service and memory constraints.
The shortlist also tests a doubled outer dispatch batch and an alternative
number of concurrent sort lanes, while the inference microbatch stays frozen.
It compares a latency-oriented flush batch with a larger throughput-oriented
batch at the same admitted shard capacity. Selection minimizes summed measured
service work; the maximum of stage estimates is retained only as a diagnostic.
This avoids hiding extra sorting work behind a transport-dominated envelope.
Every proposal must admit the same effective frontier on all ranks. Native
component probes share persistent per-GPU processes and use an allocation
acknowledgement barrier before entering transport collectives.
The five-stream architecture and global selection semantics remain unchanged.

The default reports best found Stream1 throughput and calibration duration.
Inference search is bounded by `calibration_max_batch` (8192 by default); its
winner is the best verified candidate in that search, not an unbounded optimum.
The cache includes this bound as well as the exact requested frontier.
Isolated component timings are a scheduling proxy; they are not a measured
complete depth or a proof of globally optimal performance. To measure the exact
full-frontier gap, use `NativeOptions(calibration_full_frontier=True)`; this
prepares legal unique states and runs five measured complete depths after a
warmup, followed by matched Stream1 inference on those same files. The report
distinguishes throughput loss from relative time overhead. Small finite graphs
or an explicit preparation budget can prevent a full-frontier fixture.
Full-frontier verification is optional and can take minutes on a maximum beam:
its preparation and repeated full steps are included in the reported total
calibration duration. The default component calibration does not generate those
large frontier files. Small beams need not benefit from more GPUs: dispatch,
collectives and final selection can dominate a very fast inference pass.
For effective frontiers up to 1,048,576, this full check also compares a proxy
winner against the exact-frontier baseline and retains the baseline unless a
stable material speedup is measured. Larger frontiers keep the bounded service
selection; a full timing receipt certifies execution, not global optimality.

`beam_width="max"` searches native memory admission across shard counts 1–128
with two staging slots and one active sort slot. This is the largest admitted
beam within that policy, not a universal maximum over every allocation policy.
An unknown maximum starts with a bounded inference bootstrap, then repeats
inference calibration keyed by the actual admitted width. A changed batch or
memory reserve triggers re-admission; an unstable capacity/batch cycle fails
explicitly instead of accepting the small-beam cache as a maximum profile.
Its receipt remains `full_step_verified=False` until the optional full-frontier
check actually completes. GPU topology, graph, model, precision and selected
device cohort remain part of the evidence; 128-GPU planning is not hardware
validation.

```python
import cayleypy
from multigpubeamsearch import beam_search, NativeOptions

result = beam_search(graph, start_state=start, beam_width="max",
                     native_options=NativeOptions(num_gpus=2),
                     backend="native", max_steps=100, return_path=True)
```

```python
from cayleypy_native import prepare_native, enable_native

prepared = prepare_native(graph, predictor, native_options=options)
enable_native(prepared.options)
for start in start_states:
    result = graph.beam_search(
        predictor=prepared.model,
        start_state=start,
        beam_width=2**20,
        max_steps=100,
        return_path=True,
        backend="native",
    )
```

Preparation is optional and strict: it exports the loaded model once, builds or
verifies the runner immediately, and stores owned copies under
`<cache_dir>/prepared/<id>/`. It does not search or patch CayleyPy. The returned
`PreparedNative` exposes `.model`, `.options`, `.preparation_dir`,
`.runner_sha256`, and `.preparation_seconds`. Passing
`native_options=prepared.options` on each search also works without changing
the globally configured options.

Later mutation or retraining of the original model does not update this
snapshot; prepare explicitly again to use new weights. Prepared model bytes,
including the exact `manifest.json` serialization, are pinned by SHA256, and
changing a snapshot artifact raises `NativeBackendError`.
Each search still checks graph/device compatibility, copies the exact prepared
artifact, verified runner and build-pinned NCCL shared library into its private
run directory, and validates every copy against its pinned hash. The private
NCCL bytes are stored as `runtime-libs/libnccl.so.2`, first on the worker's
`LD_LIBRARY_PATH`, so a concurrent package upgrade cannot change the loaded
dependency. The adapter then rehashes every private manifest/blob byte after
runtime preparation and immediately before launching fresh native workers.
Workers execute only the private runner copy. Successful paths are independently
replayed.
It skips exporter subprocesses, compiler/source discovery and source/CUTLASS
tree scans. This is not a persistent worker or a GPU-resident model cache.

No torch fallback is retained implicitly. To choose one explicitly, pass
`fallback=predictor` to `prepare_native`; that fallback is the caller's live
object, while native weights remain the frozen snapshot. Preparation itself
never falls back. A `NativeModel` constructed manually remains an unpinned
source declaration unless its optional `expected_artifact_hash` is supplied;
the per-search execution copy is always content-checked and isolated.
