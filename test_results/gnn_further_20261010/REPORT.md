# Further native GNN optimization with Astra consultation

Accepted: fuse FP16 residual addition, LayerNorm affine transform and exact
erf GELU. Original modules, learned parameter names/shapes, epsilon and both
GNN levels are retained. CUTLASS compact GAT uses this operator; LibTorch and
unsupported channel widths retain their existing path. Five beam streams and
frontier/deduplication contracts are unchanged.

Baseline commit: 31126ac91b78e82d4acc6cba66153fefe1944783.
Accepted GPU candidate: b74b6acf599e9a134aa866bbb7bcf61fe01376c0.
Same RTX3060; n=100/d_model=256, two GAT layers, two neighbor hops 75/38, state batch3,
seed 42 synthetic parameters, seed 901 states, FP16 body/common FP32 scalar head.
Compilation, loading and transfers are excluded. Five warmups then21 CUDA
event repetitions per block, with variant order reversed in the second block.

| Block | Before | Fused normalization |
|---|---:|---:|
| First | 178.373 ms | 164.289 ms |
| Reversed | 180.840 ms | 164.440 ms |
| Median of42 samples | 179.027 ms | 164.364 ms |

Throughput improves 1.0892x (8.92%); latency drops 8.19%. This is a complete
single-model GNN forward for three states, not a full beam-search depth.
No cumulative cross-rental speed ratio is asserted.

The isolated855000x128 residual/norm/GELU tensor takes8.562ms through ATen
versus1.977ms fused (4.33x). At4096x128:0.08694 vs0.04608ms. The kernel uses
one warp per row, compile-time register chunks, centered two-pass variance,
FP16 rounding after residual and after normalization, and the erf GELU formula.
Its reduction order can differ from ATen; bitwise norm equality is not claimed.

Correctness:120 CUDA normalization cases (widths 4..1024 including 132, odd row
counts, random/constant/nearly-constant/large/cancellation inputs) pass with
rtol 0.003/atol 0.002; maximum absolute error 0.00048828125. Complete scalar scorer
differs from the previous native implementation by at most0.0000349805 for
the matched states. The existing full-scorer acceptance tolerance was not
broadened. CPU reference/export tests: 9 passed. Compute Sanitizer:zero errors.
Final normalization-only production binary is rebuilt incrementally, with its
source digest verified equal to the accepted candidate checkout and binary
sidecar hashes refreshed. Single GNN and three-GNN blend both find length 3
paths and pass adapter replay. This is synthetic correctness, not trained quality.

Rejected experiments, preserved as evidence:

* Parallel edge exponentials (fe13889d246e99a1c77ddc2b3ab36f483ef7d0ff):
  complete scorer outputs match normalization-only after warmup. Initial
  end-to-end timing looked encouraging but varied strongly between blocks.
  Controlled isolated GAT CUDA-graph replay on855000 nodes is slower:
  16.539ms original versus17.325ms candidate. Outputs are bitwise identical.
  This variant is not enabled in the release. The graph timing isolates this
  operator; it does not imply CUDA-graph support for the whole production beam.
* Wider CUTLASS output tile128 (d5f39c39d9d8e2d310ceb669491769f739588443):
  in its own AB/BA comparison, reference156.812ms versus wider tile160.489ms;
  scalar outputs match. Rejected. The established128x64x32 tile is retained.

Astra recommended normalization fusion first, preserving rounding and original
edge order, then isolated softmax/tile measurements. Its follow-up recommended
closing this round after acceptance: direct GAT-tail integration has uncertain
occupancy/synchronization cost, and incoming-table reuse is not justified by
the measured sub-millisecond construction cost. No universal optimum is claimed.

Raw results, build logs, fingerprints and SHA256 manifest are under evidence/;
executed drivers are included alongside this report. Old exported TorchScript
GNNs must be re-exported to use the fused normalization. No checkpoint/reference
tensors were copied to Windows. Own rental is archived before deletion; see
teardown.json. Total new-test authorization is$10.
