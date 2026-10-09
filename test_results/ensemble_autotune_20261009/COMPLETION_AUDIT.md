# Ensemble and inference-first calibration acceptance, 2026-10-09

Implementation is available on the separate development branch
`codex/ensemble-autotune-source-20261009`. Runtime Python source tested in the
installed wheel is commit `5d2ff3b5f216c50e2dde0f280c9d08fea7476512`;
branch head `4c3f5f35293257e4c2ad9098cc7cbdb57fce3b5d` adds its tests/docs.
Native source snapshot is `b24037316dc62f9c71a171ead4fba5af24d981ed`.
The default branch has not been merged or represented as this release.

## Requirement-by-requirement evidence

| Requirement | Current implementation and inspected evidence | Status / scope |
|---|---|---|
| Arbitrary model count, preserving ordered coefficients | `models.NativeEnsemble`, manifest exporter and native `stream1_ensemble_libtorch.hpp` use dynamic member containers; no two-head cap. CPU tests cover private snapshots, signed coefficients, ordering and tampering. GPU readout counts 1/2/3/8/17, output dimensions 1/3/24. Installed public search with 3 and 17 heads passes both calibration stages and solution replay. | Complete for registered families and available memory. Public artifact adapter currently accepts MLPs; native C++ has additional registered families. Arbitrary Python architectures are not automatically converted. |
| CUTLASS weighted blend | `cuda/stream1_ensemble_readout.cu` accumulates weighted final-GEMM+bias outputs in FP32, preserves signed intermediates, clamps/quantizes only on the final head, and propagates nonfinite inputs. Independent CPU dot/quantizer oracle passes both 3060 GPUs for random signed FP16 inputs/weights, 1/3/17 heads and three output shapes. | Complete on SM80+; older devices use the explicit LibTorch readout. Backbones remain LibTorch. |
| Benchmark Stream1 first using the actual executor | Native MLP CUDA Graph helper, ordinary LibTorch MLP eager helper and ensemble helper measure their execution paths on every selected GPU, not inferred FLOPs. Seven repeated timings and numeric gates; source/binary hashes retained. | Complete for MLP/ensemble. Other registered families retain explicit conservative profiles until executor-specific probes exist. |
| Statistical batch selection and non-power batches | `autotune.py`, schema4, coarse cap inclusion and local refinement, bootstrap confidence intervals, MAD rejection, slowest-rank throughput, all-rank evidence and hardware telemetry. Nine new regressions pass, including selection of batch384 over coarse512. Installed GPU run measures32/64/128/160/192/224/256 on both ranks. | Complete bounded search. Best verified statistically accepted candidate within configured time budget, not a proof of a global optimum. |
| Freeze inference while reducing complete-depth overhead | `pipeline_profiles.py` fixes inference microbatch; ordinary scalar MLP also fixes the child-row budget. Candidate outer/ring/shard/sort settings require exact per-rank memory admission, unchanged effective beam and identical global workload. Actual native full depths are timed; each failing candidate has a bounded process-group timeout. | Complete. Installed3-head test selects sort16384;17-head keeps baseline after statistical comparison. |
| Preserve beam architecture and semantics | `ARCHITECTURE_NEED.md`, dynamic state-size specialization, zero persistent padding, original Stream3 candidate ids, Stream4 threshold/compact/CUB fixed-temp sort/reduce/compact. No tuning keys for shard top-k or semantic shard cap. Existing capacity policy is preserved. Current native source closure is byte-verified; full1M legal frontier completes and downstream sweep is archived. | Complete within tested algorithm/shape scope. Does not claim an exhaustive mathematical correctness proof for every graph. |
| Prepare128-GPU planning without hardware tests | `cluster.py`, `test_cluster.py`, `validate_rank_plans` and its tests cover16×8 layout, communicator rank versus node-local GPU ordinal, beam alignment, capacity arithmetic and all128-rank memory/cohort rejection. These tests are included in the285-pass installed-wheel suite. | Complete requested planning scope. No multi-host launcher or128-GPU performance/acceptance claim. |
| Cheap2-GPU validation,USD20 total ceiling | Only owned lease54987544,2×RTX3060,USD0.1342222222/hour including40GB. Existing real Cube444 transformer/MLP/MLP gate and1M frontier sweep; fresh installed-wheel3/17-head search, native1GPU and LibTorch2GPU cases. Instance stopped and DELETE succeeded; subsequent status returned absent. | Complete. No new expensive hardware rental. Quoted rates and start_date retained in stopped billing snapshot; final invoice is not invented. |
| Usable installation, GitHub/HF transport, teardown | Compact source archive585041B,176 CMake/include closure files verified. Wheel681868B,SHA25675af2e80d390a2634f3b396cee6ec77b0e288bf7a25000de6bd03d42fa786710; fresh installed package automatically selected bundled native sources and downloaded pinned CUTLASS. Installed suite285passed/5skipped. Complete raw evidence archived directly from Vast to GitHub and independently downloaded/verified on Vast before destruction. | Complete. No result archive downloaded to Windows; HF write access was not needed. |

## Immutable evidence

- Fresh3/17-head receipts,285-pass CPU log and full raw archive:
  [11236901774c07827c41997380eea0a8e69d3777](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/11236901774c07827c41997380eea0a8e69d3777).
- GitHub round-trip verification:
  [f1ddddfb8e5846a6605ad8f310145c472c8c9a9c](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/f1ddddfb8e5846a6605ad8f310145c472c8c9a9c).
- Random readout oracle, signed accumulation and nonfinite gates:
  [e4fc06ee537de8f8f8e8e4dfba645057f69fb08a](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/e4fc06ee537de8f8f8e8e4dfba645057f69fb08a).
  Readout source at this receipt has an extra trailing CRLF compared with the
  compact source; inspected contents are otherwise identical. Old hash42c58c...
  and current93a6e5... are not presented as byte-identical binaries/source.
- Real Cube444 three-model gate:
  [f87e076be1ab8d4150549a775d209cf2993f1a84](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/f87e076be1ab8d4150549a775d209cf2993f1a84).
- Full1M downstream sweep:
  [c1d67b9bed564c00530f58f723a6dd5282eb0054](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/c1d67b9bed564c00530f58f723a6dd5282eb0054).
- Serial singleGPU-native / twoGPU-LibTorch public searches:
  [2baf516fd3fb15bbc6272edc6f66f2a66504b6ad](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/2baf516fd3fb15bbc6272edc6f66f2a66504b6ad).
  Forced LibTorch on3060 is not T4 hardware acceptance of these bytes.

Fresh installed-wheel solve paths have length4 and were independently replayed
again with CPU CayleyPy from persisted terminal artifacts. Two harness assertions
were wrong: prepared options intentionally clear source/CUTLASS paths, and
ensemble build metadata says LibTorch for the backbones. These assertions were
corrected; raw failed-harness logs are retained. Successful searches/calibrations
were not repeated to hide those harness errors.

The raw archive is349879B, SHA256
`310bba33bdf926c1a49ca2938428e517ac402d5fb613cb7bbbea360b6448406d`.
It stores624 unique content blobs plus a manifest for all1256 logical files.
To restore a file, read its manifest row and copy `blobs/<sha256>` to that
manifest path; verify its byte count and SHA256. The independent round-trip
verifier checked every logical file this way before lease destruction.

## Timing scope

On the installed lrx8/Hamming test at beam8192, both runs selected parent batch256.
Three-head complete-depth median is0.0152106s;17-head is0.0633793s. These are
bounded legal-frontier full-depth measurements, not trained Cube444 timings or
complete solution wall times. The earlier Cube4441M median is36.7595s/depth
with inference micro1024. Different models/workloads are kept separate.

No guarantee is made that short calibration maximizes every GPU workload.
Calibration uses a bounded legal frontier for larger beams, inference cache
does not re-benchmark on every hit, and128-GPU planning has no hardware tests.
