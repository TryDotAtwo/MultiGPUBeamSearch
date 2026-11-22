# CNOT local goal evaluation in C++ ATen — 2026-11-16

The tested source is Git base `e940ed360529881fd817999557e47a47c8b9ccfd` plus
seven pinned overlays in `cnot-goal-native/cnot-goal-native-evidence.tar.gz`.
The remote job checked all source bytes before and after execution. Archive
SHA-256: `147def5af2ee76128fae4cf8170f35f6e2e6f1855140be65016f8260fa8cf6ca`.
The adjacent `verify_results.py` checks bounded evidence and source overlays;
pass the published stage commit to verify immutable source bytes.

The Rust CNOT controller invokes a C++ ATen local goal stage on CUDA. The same
full-matrix operations, candidate ordering, physical gate IDs and collective
goal exchange are retained. CPU, the Python reference controller and older
external operator libraries preserve their existing paths. This transfer makes
no performance claim and leaves other CNOT stages in Python.

On Linux with PyTorch 2.9.0+cu128, CUDA 12.8 and one/two RTX 3060 12 GiB:

- Rust workspace, 34 focused CPU/Gloo tests, 36 CUDA cases and the complete
  Python integration suite (748 passed, 10 skipped) passed.
- Goal results matched the existing reference for empty, identity and one-step
  candidate sets, including n=4/65/144 and noncontiguous inputs. A simulated
  older operator interface exercised the fallback.
- Full one/two-GPU searches, two-rank NCCL radix and Python goal comparisons,
  and checkpoint comparisons passed. Injected local-goal errors on each rank
  reached both peers and the same process group was reused. The checkpoint
  Qhat case replayed `[5, 7, 0]`; proposal preserved its feature F02 coverage
  rejection.
- Two preliminary runs stopped due to missing test-harness setup: a scorer
  import path and then a legacy receipt directory. The complete source-pinned
  retry included both corrections; the calculation itself was unchanged.

The checks do not establish multi-node/Slurm behavior, other architectures,
all model/checkpoint domains, performance or completion of the remaining
Python stages.
