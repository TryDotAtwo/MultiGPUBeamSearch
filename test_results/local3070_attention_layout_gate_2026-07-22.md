# RTX 3070 QKV-to-FMHA layout gate — 2026-07-22

## Decision

Reject the opt-in vectorized packed Q/K/V layout and keep the production strided token-major QKV path unchanged.

The packed layout preserved byte-exact output but regressed the full Stream1 median by 10.79%. Nsight Systems showed that the packed CUTLASS FMHA kernel itself was not faster, while the additional vectorized pack cost about 219.5 us per layer. Therefore folding this permutation into the QKV GEMM epilogue has no measured upside for the current CUTLASS FMHA implementation on SM86.

## Measurement validity and exactness

Before the A/B and profiling runs, the long-lived queue reported no active compute processes and three consecutive `SM=0%`, `MEM=0%` samples. All GPU work ran serially through the queue.

The packed kernel copied Q/K/V with aligned 16-byte `uint4` loads and stores. Both smoke dumps and all 40 alternating A/B dumps matched the canonical complete score SHA-256:

`a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`

No fallback, `CUDA_LAUNCH_BLOCKING`, production MLP change, or mathematical reordering was introduced.

## Clean alternating A/B20

| Layout | Median | Mean including one system outlier | Range | Paired wins |
|---|---:|---:|---:|---:|
| Production strided | 8.51100 ms | 9.76428 ms | 8.0720–35.2019 ms | 19/20 |
| Vectorized packed | 9.42925 ms | 11.20970 ms | 8.9332–46.9878 ms | 1/20 |

Each layout had one large system outlier, so the decision uses the robust median and paired ordering. Packed regressed the median by 10.79% and lost 19 of 20 pairs.

## Nsight Systems decomposition

Clean CUDA Graph node profiles used Nsight Systems 2025.6.3 with `--cuda-graph-trace=node`.

| Kernel | Strided median | Packed median | Observation |
|---|---:|---:|---|
| CUTLASS FMHA q64k64 | 224.1455 us | 224.1910 us | no packed-layout benefit |
| Vectorized QKV pack | absent | 219.5185 us | pure added traffic |

The profile captured 28 attention invocations. The pack consumed 6.144 ms aggregate, or about 0.878 ms for the four-layer inference represented by one benchmark iteration. That closely matches the 0.918 ms full-path median regression.

This closes the layout-only path for the current CUTLASS example-41 kernel. A future custom fixed-shape attention kernel may choose a different layout, but it must demonstrate faster K/V reuse inside the kernel; merely materializing head-major Q/K/V is not sufficient.

## Cleanup and production revalidation

The temporary layout enum/parser/tests, vectorized pack kernel, and packed dispatch were removed. After rollback:

- Docker CTest passed 18/18;
- production measured 8.0635 ms;
- the complete score dump matched the canonical SHA;
- `git diff --check` passed.

## Evidence

- idle gates: `.gpu_queue/logs/5a18806ef0ae.log`, `.gpu_queue/logs/e082d47294e8.log`, `.gpu_queue/logs/bfd3d4a186f2.log`, `.gpu_queue/logs/31ac1f86aa3c.log`;
- exact smoke: `.gpu_queue/logs/6bd0e08d7323.log`;
- alternating A/B20: `.gpu_queue/logs/929c7ca6827a.log` and `test_results/packed_layout_exact_ab20/`;
- profiles: `test_results/local3070_attention_layout_strided_nodes_2026-07-22.nsys-rep`, `test_results/local3070_attention_layout_packed_nodes_2026-07-22.nsys-rep`, and their `.kernels.csv` summaries;
- post-rollback validation: `.gpu_queue/logs/f5bdad9d4f35.log` and `test_results/production_after_packed_reject_2026-07-22.bin`.
