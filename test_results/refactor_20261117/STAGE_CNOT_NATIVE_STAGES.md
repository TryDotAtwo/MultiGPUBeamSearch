# Rust-owned CNOT rank-local stages

The CNOT Rust search backend retains the rank-local frontier, closed-set,
candidate, metadata and history tensor handles. PyTorch/CUDA continue to own
the actual storage, and existing batch operators, numerical policy, full-word
CNOT equality, physical routes and NCCL exchange order are unchanged. Before a
nontrivial Rust search, ranks agree on availability of the new stage backend;
if any rank lacks it, all use the compatible Python stage adapter.

## Source-pinned evidence

- The base is `46d4780a863c30c77a003850d9bf60194e8a9dd1`; six modified source
  files are SHA-256 pinned in the archive manifest. Source files matched that
  manifest both before and after the tests. The green archive SHA-256 is
  `c23496d1f4b9a138d43c60a4e42e32513454fa321ac68f9a6de9c9f7fcfde0f4`.
- A fresh Cargo target and explicit `run_cnot_native` interface check were used.
  On Linux, CUDA 12.8 and PyTorch 2.9.0 with two RTX 3060 GPUs, the Rust
  workspace and extension built and passed; 45 focused CPU/Gloo tests and 62
  CUDA tests passed. The full Python suite reported 785 passed, 10 skipped.
- Five complete audited CNOT searches matched the Python reference on one and
  two GPUs, including qhat, proposal, full-action and 144-wire cases. The tests
  reject construction of the old Python stage owner on the new path. Both
  ranks also selected that old owner together when its replacement was hidden
  on one rank; the full result still matched. An injected scorer error reached
  both ranks, after which the same process group was reused.
- The checkpoint scorer matched on one and two GPUs. The qhat fixture replayed
  `[5, 7, 0]`; the proposal fixture retained the expected unsupported-feature
  rejection. Two-rank communicator lifecycle, record exchange, radix selection
  and native/legacy goal collectives are included in the green archive.
- The pre-implementation test correctly failed twice when the Python stage
  owner was prohibited. Its archive SHA-256 is
  `a6b4900070eb103b3e01e1fc832a9a9ff2d4dec53f7e92abf4fc8e014ed85b68`.
  A later smoke attempt reused a stale Cargo artifact from an unpacked source
  tree and failed for the same reason; its archive SHA-256 is
  `97f45a7010ac499d8cdd2b7b30d126419b22874a04d3cb48b8c80789c000d652`.
  With a fresh target, an explicit interface check and 11 focused controller
  tests passed; that smoke archive SHA-256 is
  `9397010aada861373a8a0ef5c497dae4276c6d292cd4a3a0a2307be3b50b2858`.
- Compute Sanitizer memcheck ran the full two-rank comparison, including the
  fallback and scorer-error cases, and reported zero errors for both processes.
  CUDA API error reporting was disabled (`--report-api-errors no`); therefore
  this is not an unfiltered memcheck pass. A prior unfiltered Stream 5 check
  reported CUDA API error 209 during NCCL initialization; see
  `test_results/refactor_20261116/STAGE_CNOT_STREAM5_TRANSPORT.md`.
  The filtered search archive SHA-256 is
  `063170681726dae827c068abcf2bbb0a1af93db7e8b4314016dc9018f23436fb`.
- `cnot-native-stages-green/verify_results.py` checks source overlays, archived
  results, rank receipts and the owned Vast lease's provider release receipt.
  Instance `60975486` was deleted and provider absence verified. Its last
  reported duration and rate correspond to approximately $0.071; final
  billing may differ.

This remains an intermediate migration. The Python local-goal coordinator,
selection stage adapter, user scorer callbacks, several collective operations
and the compatibility/reference path still execute Python. PyTorch owns device
storage. These checks cover a single host with one/two RTX 3060 GPUs; they do
not establish multi-node/Slurm, T4, Hopper or Blackwell runtime acceptance.
