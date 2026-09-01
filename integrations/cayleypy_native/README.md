# CayleyPy native beam adapter

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

## Backend selection

| Selector | Behavior |
| --- | --- |
| `auto` (default) | Prepare a supported native run; otherwise warn with a concrete reason and call the saved original method with the original arguments. |
| `native` | Reject unsupported graph/model/runtime/options with `NativeUnavailable`. |
| `torch` | Call the original method without probing CUDA or native compatibility. |

Already-solved states and zero-step budgets return directly after graph/argument
validation. They do not probe CUDA, inspect a model artifact, compile, or launch a
worker, so these deterministic boundary cases also work on a CPU-only host.

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
| MLP dimensions | `num_classes >= max(state_len, max_label+1)`, positive residual count, hidden widths divisible by 8 and hidden1 >= hidden2. This can rule out an otherwise supported colored graph. |
| Search options | Simple mode, global beam width, strict maximum path length, return path. Native touch-BFS uses `NativeOptions(touch_bfs_radius=...)`; its suffix stays inside `max_steps`. |
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
and the binary SHA256. The wrapper-generated file is the reference format. Do not
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

Each run is stored below `<cache_dir>/runs/<id>/` with model export, build and
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
the development API. The public-hook integration is covered by tests when that
API is installed. Native CUDA tests require a real compatible machine; CPU
protocol fixtures are not a GPU performance result.

## Prepare explicitly for repeated searches

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
snapshot; prepare explicitly again to use new weights. Prepared model bytes are
pinned by SHA256, and changing a snapshot artifact raises `NativeBackendError`.
Each search still checks graph/device compatibility, hashes weights and the
verified binary, then rehashes every manifest/blob byte after runtime preparation
and immediately before launching fresh native workers. Successful paths are
independently replayed.
It skips exporter subprocesses, compiler/source discovery and source/CUTLASS
tree scans. This is not a persistent worker or a GPU-resident model cache.

No torch fallback is retained implicitly. To choose one explicitly, pass
`fallback=predictor` to `prepare_native`; that fallback is the caller's live
object, while native weights remain the frozen snapshot. Preparation itself
never falls back. A `NativeModel` constructed manually remains an unpinned
declaration unless its optional `expected_artifact_hash` is supplied.
