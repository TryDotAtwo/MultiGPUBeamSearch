# CayleyPy native beam adapter

An optional Python wrapper around this repository's native beam search. Enable
it once, then keep calling `graph.beam_search(...)`. CayleyPy supplies the graph
and public result API; the supported native path runs our existing CUDA/NCCL
algorithm as a local subprocess. Neither CayleyPy nor the native solver is patched
on disk.

This first integration has passed end-to-end validation on **two real Tesla T4**:
LRX8, explicit prepared-model reuse, and a real N88/G24 Tetraminx artifact. Native
paths were independently replayed, including multiple moves and exhausted
budgets. This is integration evidence, not a trained-model Torch speedup claim.

## Installation

Use the CayleyPy checkout inspected with this adapter, currently
`28f3841b34009ea8d51bb36eece3dd0be757a145`. The PyPI package and this checkout both
report version `0.1.0`, despite API differences. Install into your own environment:

```bash
python -m pip install -e /work/100XH100/CayleyPy
python -m pip install -e /work/100XH100/integrations/cayleypy_native
```

Native execution requires Linux, CUDA-capable PyTorch, NCCL, a CUDA/C++ compiler,
CMake and an existing CUTLASS source directory. The adapter does not download
weights or source, submit cluster jobs, install dependencies, or publish results.
Windows/CPU calls in `auto` mode use ordinary CayleyPy and explain why.

## Keep the existing call

```python
from cayleypy_native import NativeOptions, enable_native

enable_native(NativeOptions(
    source_dir="/work/100XH100",
    cutlass_dir="/opt/cutlass",
    cache_dir="/work/native-cache",
))

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

## Backend selection

| Selector | Behavior |
| --- | --- |
| `auto` (default) | Prepare a supported native run; otherwise warn with a concrete reason and call the saved original method with the original arguments. |
| `native` | Reject unsupported graph/model/runtime/options with `NativeUnavailable`. |
| `torch` | Call the original method without probing CUDA or native compatibility. |

Importing the package does not enable interception. `enable_native(...)` installs
a reversible, process-wide method wrapper and is idempotent; `disable_native()`
restores the original method. Configure before launching application threads;
already captured methods keep their session after disabling.

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
| Search options | Simple mode, global beam width, maximum steps, return path. Native touch-BFS uses `NativeOptions(touch_bfs_radius=...)`. |
| Device placement | Local Linux CUDA devices, one native rank per selected GPU, up to 128. Graph CUDA devices are used by default; explicit `devices=(0,1)` overrides them. |
| Not yet supported natively | Matrix graphs, arbitrary models/tokenizers, PieceTransformer artifacts, custom destination, advanced/history taboo policy, reusing an upstream BfsResult, existing torchrun ranks, multi-node launch. |

There is no puzzle-name whitelist. Representation/model/runtime capabilities
decide support. A compatible graph alone does not imply a compatible model.

## Builds, devices and evidence

Source and CUTLASS paths are explicit. A build is cached by working-tree source
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

Run CPU tests against the local upstream source:

```bash
PYTHONPATH=/work/100XH100/CayleyPy:/work/100XH100/integrations/cayleypy_native \
CUDA_VISIBLE_DEVICES=-1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -B -m pytest /work/100XH100/integrations/cayleypy_native/tests -q -p no:cacheprovider
```

`examples/solve_lrx.py` is an offline smoke example. Without compatible weights it
exercises the explicitly reported torch fallback; providing native weights and a
Linux build runtime enables a strict native attempt. The tests use labelled fake
workers for process protocol checks, not as substitutes for a CUDA performance or
solve-quality result. Project evidence is kept under
`test_results/cayleypy_native_adapter_2026-08-31/`.

The real two-T4 run is private Kaggle version4 of
`trydotatwo/cayleypy-native-adapter-2xt4-validation`; evidence is under
`test_results/cayleypy_native_2xt4_2026-08-31/`. It used native
`a1db0e6d9bb5458c8a842b37dfa99572d3025667`, CayleyPy
`28f3841b34009ea8d51bb36eece3dd0be757a145`, and CUTLASS
`afa1772203677c5118fcd82537a9c8fefbcc7008`. The dirty local native checkout was not
exported. Both fetched source trees remained clean.

- 181 CPU contract/validation tests passed on Kaggle Linux.
- 12 synthetic Torch/native records, two prepared public-API calls, and ten real
  Tetraminx runs passed;16 calls actually launched both native ranks.
- State specialization was8 logical/16 storage bytes and88 logical/96 storage
  bytes. Tetraminx requested and observed global beam65536.
- On the one-move real puzzle, median warm full API time was7.14s versus5.61s for
  a direct prepared native call (three alternating pairs). Native worker wall
  was about4.3–4.4s, with about0.09s inside the native search timer. This separates
  setup/verification/worker costs; it is not a Torch comparison or a long-search
  throughput benchmark. The first Tetraminx API call included compilation and
  took161s. Prepared LRX8 calls took5.36s and5.53s without re-export/reconfigure;
  that is a different graph, not another Tetraminx timing row.

Fresh worker creation remains noticeable for short searches. Explicit preparation
removes repeated export/build discovery; it does not remove native initialization
or provide a persistent in-process engine. Installed NCCL is still required at
the path recorded in the build metadata.

This package is local integration work. No upstream submission, external release
or native-code licensing decision is implied.

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
verified binary, starts fresh native workers, and replays successful paths.
It skips exporter subprocesses, compiler/source discovery and source/CUTLASS
tree scans. This is not a persistent worker or a GPU-resident model cache.

No torch fallback is retained implicitly. To choose one explicitly, pass
`fallback=predictor` to `prepare_native`; that fallback is the caller's live
object, while native weights remain the frozen snapshot. Preparation itself
never falls back. A `NativeModel` constructed manually remains an unpinned
declaration unless its optional `expected_artifact_hash` is supplied.
