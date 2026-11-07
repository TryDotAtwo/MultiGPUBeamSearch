# Rust frontier calibration ownership

Scope: one coarse Rust job coordinates inference calibration, exact-width native
admission, downstream calibration and symbolic maximum/working-width selection.
This is a comparative migration stage, not the public-launch switch or a CUDA
optimization acceptance.

## Implementation and contracts

- Numeric requests retain their exact logical width. A capacity rejection can
  select another measured inference batch; rank, protocol and runtime errors
  cannot be converted into a smaller beam.
- Symbolic `max` separates a finite multiset/VRAM bound, native memory admission,
  an exact-width inference refinement and full-step working-width evidence. The
  multiset calculation is an upper bound, not proof of the reachable orbit.
  Changed inference batch or reserved memory requires readmission, with at most
  three refinement rounds. Maximum mode currently requires the natural-growth
  calibration protocol.
- The working-width comparison uses a fresh complete cohort at half width under
  the same profile, with at least one microbatch per rank and at least 80% of the
  smaller frontier's throughput. At most three comparisons are made. The result
  is the largest suitable width actually measured, not an exhaustive optimum.
- The existing `calibration_frontier_max_states` is a fixture-generation budget;
  it does not cap the search width. A separate optional low-level descriptor
  ceiling exists for explicitly bounded diagnostic requests and is not supplied
  by the public adapter.
- Rust owns the job deadline, cancellation and rank cleanup. Python exposes a
  thin adapter; the existing public search launcher remains in place until its
  own comparison and switch gate. Specialized inference and precision are intact.

## Regressions covered

- A two-rank request could accept a one-rank admission result when that result
  declared its own cohort as one. A failing test is retained in `red-cohort.log`;
  admission now checks count and ordered rank IDs against the requested cohort.
- The first implementation accidentally treated the fixture state budget as a
  `max` search limit. The old fixture-generation path was checked, and a separate
  explicit descriptor ceiling and regression test now preserve its semantics.
  `red-fixture-budget.log` records the new test before its interface existed;
  its compilation failure is not presented as a runtime reproduction.

## Verification

All builds and GPU runs used the owned Vast 2x RTX 3060 instance. The final source
manifest and hashes of the extension, Rust runner and both native executors are
in `stage3-frontier-3060/`. The final Rust workspace test run passed 64 tests,
with one ignored; the release extension and CLI built successfully. The Python
regression suite passed 528 tests with 9 skipped before the final Rust-only
fixture-budget correction.

The retrieved archive contains 2,068 files / 20,444,796 uncompressed bytes;
its SHA-256 is `04e85b33b17e1299110f61f42421112625f387c6bd9a70b6aa900e67cd003943`.
The archive and each extracted file were checked before staging.

The final GPU series ran both actual `native_cuda_graph` and `libtorch_eager`
executors on one and two GPUs. Each exercised exact width 4097 on first and
cached calls, an explicitly diagnostic-bounded `max` of 8192, complete-step
and working comparisons, the real thin model/options adapter and cancellation
with both rank processes reaped. Per-rank logs assert the selected executor.
Raw first failures and final receipts are retained together rather than replaced.

This Hamming-model fixture does not establish trained-model quality, a full
application migration or a performance improvement. CNOT orchestration, public
launcher switching, CUDA bulk operations, paired production performance and
T4/Hopper/Blackwell hardware acceptance remain outstanding. The earlier observed
small-capacity Stream 4 overflow remains distinct from VRAM admission; this
fixture explicitly reserves sufficient arrival capacity without shrinking width.
