# Real two-GPU acceptance

`smoke_cases.py` is a source-only validation harness. It does not change CayleyPy
or native CUDA code and does not download dependencies, schedule jobs, or patch
anything at import time. Use a dedicated Python process with the adapter,
CayleyPy, PyTorch/CUDA, native source, CUTLASS, CMake, NVCC and NCCL available.

```bash
PYTHONPATH=/path/to/adapter:/path/to/CayleyPy python -m validation.smoke_cases \
  --source-dir /tmp/native-source \
  --cutlass-dir /tmp/cutlass \
  --output-dir /kaggle/working/cayleypy_native_validation/acceptance \
  --cache-dir /tmp/cayleypy-native-cache \
  --devices 0,1 --timeout-seconds 180 --build-timeout-seconds 1800
```

Or call `run_acceptance(source_dir=..., cutlass_dir=..., output_dir=...,
devices=(0, 1))` from Python. It returns a JSON-compatible report, writes
`acceptance_report.json` after each call, and records full exceptions without
turning native errors into torch fallback. CLI exit status is zero only when
all required checks pass. Reports and the fixture checkpoint go to `output_dir`;
builds, histories, input files, export artifacts and native logs go to
`cache_dir` (default `output_dir/cache`). Preserve the referenced logs before
discarding a temporary cache. Root's Kaggle bootstrap handles artifact copying.
After a native exception, remaining calls are marked `not_run`; the workflow
does not repeatedly retry a failed build, timeout or crash. A completed result
whose checks fail is retained and later cases can still provide evidence.

The workflow creates the public `PermutationGroups.lrx(8)` graph and a seeded,
untrained scalar `Pilgrim` network: one-hot 8×8 input, 32/16 hidden units, one
BatchNorm/ReLU residual block. The existing native exporter writes FP16 weights
from the exact model passed to the public `Predictor`. No pretrained files are
downloaded. A tensor-only checkpoint and content hash identify the fixture.

It calls `enable_native(...)`, then only `graph.beam_search(...)` for searches,
always with an explicit `backend='native'` or `backend='torch'`. The original
method is restored in `finally`. Six scenarios run through both implementations
(12 report records, six with `backend == 'native'`):

| Scenario | Exact distance | Depth budget | Expected |
| --- | ---: | ---: | --- |
| Already at center | 0 | 0 | Empty replay-valid path; adapter host shortcut |
| One move | 1 | 1 | One-move replay-valid path; native workers |
| Multiple moves | 3 | 3 | Three-move replay-valid path; native workers |
| Zero budget | 3 | 0 | Not found; adapter host shortcut |
| Insufficient positive budget | 3 | 1 | Not found; native workers |
| Repeated multiple moves | 3 | 3 | Same correctness and reused native build |

An independent bounded CPU BFS establishes exact distances, and literal Python
tuple gathers replay every returned path against the original generators and
center. The requested global beam is seven: exact walk layers have 1, 3, 7, 15
unique states. All states before a distance-three goal are retained. At depth
two the torch baseline scores seven states and retains all seven, so its
`debug_scores` confirms that this search actually calls the model. Native may
round the effective global beam upward; the report preserves both widths.

For each nontrivial native call, passing requires worker-emitted `WORLD_SIZE=2`,
both `LOCAL_RANK` values 0 and 1, and both `CUDA_DEVICE_LOCAL_RANK` values 0 and 1.
These messages follow CUDA device selection and NCCL initialization in the
native runner. A requested device list alone cannot pass. Garbled/interleaved
startup output or missing evidence fails conservatively. The two host shortcuts
explicitly do not claim worker execution. The report also checks build shape,
graph/model identity across calls, path semantics and search budget behavior.
The original adapter artifact hash is preserved, including its export provenance.
Cross-call numerical model identity uses exported blob hashes and model schema,
excluding the unique per-run source-checkpoint path in the exporter manifest.

Each call records synchronized total wall time. Native metadata separately
contains subprocess `elapsed_seconds`, runner `native_seconds`, build metadata,
and whether CMake configured during that call. The repeated scenario checks
build reuse, but process startup, export, hashing, and runtime setup still occur.
The torch baseline runs on GPU 0, whereas native uses both GPUs. These are single
observations of a tiny graph with an untrained model, not speedup/scaling or
trained-quality measurements. FP32 torch and FP16 native scores need not agree
bitwise. This smoke does not establish large-puzzle, trained-model, OOM or
long-running distributed-history correctness.

Focused CPU tests: `python -m pytest -q validation/test_smoke_cases.py`. They test
the oracle, real upstream baseline, reproducible model, real CPU exporter when
native source is adjacent, strict rank evidence, replay rejection and failure
reporting. They do not simulate or claim a GPU acceptance pass.
