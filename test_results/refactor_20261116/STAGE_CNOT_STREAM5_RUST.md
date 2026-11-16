# Rust CNOT Stream 5 communicator ownership

The new Rust owner loads the existing ABI-1 library lazily, validates its ABI,
NCCL identity size and rank geometry, retains the dynamic library while a
communicator is alive, and closes or aborts the communicator once. PyO3 exposes
the thread-bound owner for comparative tests. The caller must keep the same
current CUDA device and thread through initialization and close. It does not
accept or exchange GPU buffers, and the active CNOT search still uses the
existing Python Stream 5 transport and rank collectives. It is not the complete
Rust/CUDA migration and makes no performance claim.

## Source and evidence

- Base: `3880a7d26998fa29acb8efebe3d648bf39485778`. The evidence manifest pins
  the eight implementation/test overlays and all baseline source files. Before
  and after source checks matched.
- Final evidence: `cnot-stream5-retry/cnot-stream5-retry-evidence.tar.gz`,
  SHA-256 `008726c2cfc5d612f5ba76bb348ae0104a7fd8f7a90401f6a5c214a30fd901cf`.
  Run `python test_results/refactor_20261116/cnot-stream5-retry/verify_results.py`
  with an optional committed source revision to check the artifacts and overlays.
- Linux, CUDA 12.8, PyTorch 2.9.0, Rust 1.100.0, two physical RTX 3060 GPUs
  with 12 GiB each. Rust workspace tests and PyO3 build passed; 39 focused
  CPU/Gloo tests and 62 CUDA tests passed. The full Python integration suite
  reported 779 passed and 10 skipped.
- Both physical NCCL ranks completed three new communicator lifecycle rounds,
  alternating graceful close and abort, while the default Torch communicator
  continued to work. Mock native ABI tests cover library lifetime, idempotent
  close, abort on drop, invalid rank/ID, version mismatch and injected errors.
- Existing one/two-GPU CNOT search, radix and goal collective comparisons passed.
  The real checkpoint scorer passed the existing one/two-rank comparison,
  including legal replay of gates `[5, 7, 0]` and the expected unsupported
  feature rejection. These search tests exercise the existing transport, not
  payload exchange through the new Rust owner.
- The prior baseline red test confirmed that the Rust owner was absent. The
  first implementation test failed before GPU tests because archive timestamps
  dated 1970 caused Cargo to reuse an older `beam-native-sys` build. The
  failure and timestamp diagnosis are retained in `cnot-stream5-green/`; the
  successful retry used a fresh Cargo target directory.
- Vast task instance `60961012` was leased at approximately $0.150/hour and
  deleted after collecting evidence; `cnot-stream5-retry/release.json` records
  the provider's verified absence. The listed duration corresponds to about
  $0.095 for this lease before any provider billing adjustments.

Actual CNOT search transport migration, GPU buffer lifetime and rank-wide
backend agreement, multi-node/Slurm operation, other GPU families and complete
public Rust ownership remain separate work. These checks are not a benchmark,
Compute Sanitizer acceptance, or proof of all distributed failure cases.
