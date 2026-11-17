# Connect borrowed CNOT expansion and ownership to Rust search

Status: implemented and source-pinned one/two-GPU verified for the explicit
`MGBS_CNOT_OWNED_LIBRARY` CUDA path. This stage routes root ownership, batched
child generation and child ownership through the existing C ABI on each rank.
It preserves parent score order, generated descriptors, metadata, full-matrix
equality, NCCL exchange and checkpoint scorer. Without the explicit library,
and for CPU execution, the current paths remain selected.

The adapter owns only the native context; PyTorch retains its input, output,
descriptor and workspace allocations. It checks matrix geometry, dtype,
device and contiguous extents, orders descriptor checks and native calls on
the current CUDA stream, and keeps all tensors alive through synchronous
return. Context capacity changes must not silently truncate a batch or make
an otherwise valid `closed + child` ordinal geometry fail. CUDA addresses
remain local to their rank. Errors must pass through the existing rank-wide
initialization or expansion vote before any subsequent collective.

The run used 2xRTX 3060, CUDA 12.8 and Torch 2.9.0. It first confirmed the
instrumented baseline failed because the new `owners` and `children` symbols
were not called, then verified them on one and two ranks. Audited Rust results
matched the previous route for qhat/proposal, n=4/n=144, repeated calls and
mixed rank selection. Existing two-rank unseen, search and checkpoint scorer
tests passed; all 101 Rust workspace tests passed and rustfmt was clean.
Compute Sanitizer memcheck reported zero memory errors with CUDA API-error
reporting disabled, so API-error cleanliness is not established. Full logs,
source hashes and release receipts are in
`test_results/refactor_20261117/cnot-search-expansion/`. This does not establish
multi-node, Slurm, other GPU architecture, performance or full CNOT migration.

