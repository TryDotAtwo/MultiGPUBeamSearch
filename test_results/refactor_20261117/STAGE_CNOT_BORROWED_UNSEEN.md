# Standalone borrowed-device CNOT first-unseen

The optional C ABI and Rust owner now expose the existing full-matrix CNOT
sort/compact operation on borrowed device matrices. The caller provides the
int32 ordinal/count, uint8 flag and aligned CUB scratch buffers, and int64
selected-index output. CUB scratch is queried for the actual batch geometry;
no internal buffer is silently allocated. Mutable regions cannot overlap one
another or the inputs. The call completes the supplied stream; selected count
remains on the device. Empty child batches write zero to that count. Older
libraries without these additive symbols continue to load.

This record covers the standalone ABI. Connecting the stage to the public
distributed search is a separate change and is not established by these tests.

The source-pinned runs used base `0aab5c5111cf0f9f8c7ab823ee8570d5f9dbc8eb`.
The initial archive retains the expected missing-symbol failure against the old
library and the first new-library tests. The latter passed the functional tests
on two RTX 3060s, but memcheck returned 97 for two retained four-byte pinned
host allocations in PyTorch's scalar `.item()` cache. The log shows no invalid
GPU accesses. The test was changed to read the count with `.cpu().tolist()`;
the continuation verified both source manifests and the original native library
hashes before rerunning. No production source changed between these runs.

On two RTX 3060 12 GiB GPUs, CUDA 12.8 and Rust 1.100.0:

- Three standalone tests cover n=2, 4, 64, 65, 144 and 256, equal and distinct
  full matrices, tail masking, original first-child order, reused workspace,
  empty batches, capacity, alignment and overlap errors, wrong thread, device
  and stream, and restoration of the caller's current CUDA device.
- Python and Rust Compute Sanitizer memcheck reported zero errors and zero
  leaked bytes in the corrected runs; racecheck displayed zero hazards for the
  exercised Python workload. The Rust workspace and the old/new Rust ABI
  compatibility test passed. Existing owner and expansion tests passed.
- The standalone library compiled for SM75, SM80, SM90 and SM120. This proves
  compilation only; those GPUs, multi-node operation and Slurm were not tested
  in this stage. Public search, checkpoint scorer and NCCL behavior require a
  separate integration test.

The initial Vast host was unreachable over SSH and was destroyed. The working
host's release and absence receipt is retained with the evidence. Both source
manifests, logs and the exact failed and corrected tests are retained under
`cnot-borrowed-unseen/`; `verify_results.py` checks the evidence and source.
