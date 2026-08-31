# Validation scope

The adapter does not promise arbitrary models, arbitrary graphs, all CUDA
architectures, identical Torch/native paths, or an unconditional speedup. Its
capabilities and failure policy are documented in [README.md](README.md).

## Public integration checks

CPU checks cover graph/move encoding, model schema and numerical export,
artifact/build identities, subprocess isolation, rank log collection, result
parsing, independent path replay, safe fallback, explicit source setup and
registration through CayleyPy's optional public hook. Windows omits POSIX
executable tests. CI builds a wheel and exercises actual installed imports with
released CayleyPy 0.1.0 and the pinned development API.

`validation/public_gpu_smoke.py` performs a fresh-source two-GPU check using
synthetic models and LRX graphs of 8 and 88 positions. It checks strict native
dispatch, prepared reuse, bounded search outcomes, original-generator replay,
both rank logs and unchanged cached sources after execution. It needs no
competition or privately supplied artifact. Its JSON reports distinguish Torch
calls, host-only goal/zero-budget shortcuts, and actual two-rank worker launches.

## Earlier native acceptance (2026-08-31)

The integration preceding the public source-setup/registry changes was tested on
two Tesla T4 GPUs (SM75, 15,636,037,632 bytes each), Linux, Python 3.12.13,
PyTorch 2.10.0+cu128, CUDA 12.8. Exact dependencies:

| Dependency | Revision |
| --- | --- |
| MultiGPUBeamSearch | `a1db0e6d9bb5458c8a842b37dfa99572d3025667` |
| CayleyPy | `28f3841b34009ea8d51bb36eece3dd0be757a145` |
| CUTLASS | `afa1772203677c5118fcd82537a9c8fefbcc7008` |

181 Linux CPU contract/validation tests passed. Sixteen actual two-rank native
launches passed, covering LRX8, explicit prepared reuse and a graph-associated
trained Tetraminx N88/G24 scalar FP16 MLP. Every returned path passed independent
replay. Logical/storage sizes were 8/16 and 88/96 bytes. The real Tetraminx case
requested and observed a global beam of 65,536.

For the one-move real puzzle, three alternating warm pairs gave a median 7.14s
through the full wrapper versus 5.61s for a direct prepared native call. The
native worker took approximately 4.3-4.4s; its internal search timer was about
0.09s. The first real wrapper call included compilation and took 161s. These are
setup/worker overhead observations, not a Torch speed comparison or deep-search
throughput benchmark. Prepared LRX8 calls took 5.36s and 5.53s; those are a
different graph, not another Tetraminx timing row.

The trained artifact and competition data are not distributed with this
package. The public synthetic script makes installation/replay checks available
without them. The historical results do not replace validation of later code
changes or another GPU/model configuration.
