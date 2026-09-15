# Cube4 H200: PyTorch, LibTorch and native comparison

## Result

At the same full-four-block workload, the tested best configurations are effectively tied: Python PyTorch CUDA Graph 390,177 parents/s, C++ LibTorch CUDA Graph 389,806, native CUDA Graph 393,538. The optimized native production profile reaches 603,486 parents/s, but has final-CLS work elimination, Hopper GEMMs and eight parallel lanes; this is not a pure language/backend comparison.

One parent is one Cube4 state producing **24 candidate scores**. All numbers below are parents/s, not candidates/s, and refer to one GPU.

## Physical run and contract

- Vast instance 51111383, NVIDIA H200 SXM, 143771 MiB, CUDA toolkit 12.8.93, driver 580.159.03. Same device as the earlier single-GPU pipeline run.
- Python PyTorch **2.11.0+cu128**. LibTorch links to the exact same installed Torch libraries, compiled with GCC 13.3 in a separate `/workspace/build-libtorch` directory. Native binary remains unchanged from the preceding compact57 experiment.
- Git source worktree `codex/hopper-stream1-fusion`, parent commit `a6483877`; remote original Git HEAD remains `3b862de1` with the matching previously uploaded compact57 source. No production code was changed for this comparison.
- FP16 Cube4 piece transformer: state96, S57, d256, 8 heads of width32, 4 blocks, FF1024, ReLU, CLS pooling, 24 outputs. Identical exported weights under `/workspace`.
- Sweep batches 128,256,384,512,768,1024,2048,4096; one CUDA stream for the full-block comparison. Three measurements of 100 iterations/replays per configuration. These are best **within this sweep**, not a certified global optimum. Torch/LibTorch best batch is at the upper tested boundary.
- Inputs already reside on GPU; synthetic `(arange * 17 + 23) % 6`. Since state length96 is divisible by6, every parent repeats the same state. This is throughput evidence, not a reachable-frontier quality certificate.
- Torch/LibTorch time uint8 state -> FP16 logits, excluding key quantization. Native additionally includes its score-key output kernel. Weight loading, input upload, warmup, graph capture and output inspection are outside timing. Torch uses synchronized wall-clock timing; native uses its existing graph benchmark timing. Do not call the boundaries bit-for-bit identical.
- Python legacy eager has per-slot GPU `.item()` and boolean indexing. The benchmark-only static-token adapter mirrors the existing LibTorch token plan and permits graph capture. No `torch.compile`, TensorRT, FP8 or alternate model was tested.
- The full-block native control uses compact57 but disables final-CLS, split-QKV and Hopper QKV/FF1 modes. It is the existing full-block path, not a separately tuned full-block Hopper implementation.

## Measured medians

| Implementation | Batch / real CUDA lanes | Parents/s | Range over 3 measurements |
|---|---:|---:|---:|
| Python legacy eager | 4096 / 1 | 368234 | 367697–368495 |
| Python static-token eager | 4096 / 1 | 387882 | 387843–387916 |
| Python static-token CUDA Graph | 4096 / 1 | 390177 | 390156–390201 |
| C++ LibTorch eager | 4096 / 1 | 387529 | 387513–387794 |
| C++ LibTorch CUDA Graph | 4096 / 1 | 389806 | 389658–389901 |
| Native full 4 blocks, CUDA Graph | 768 / 1 | 393538 | 393031–393581 |
| Native optimized CLS/Hopper, CUDA Graph | 384 / 1 | 498787 | 497682–499001 |
| Native optimized CLS/Hopper, CUDA Graph | 384 / 8 | 603486 | 603130–605625 |

At identical batch384/lane1, full-block Python graph =333703, LibTorch graph =327768, native full =387144. At batch4096/lane1, native full falls to352164 while Torch reaches390k. Increasing batch is not uniformly beneficial across implementations.

Native optimized / best Python full =1.547x, but that includes work reduction and concurrency. Native full / best Python full =1.009x. Python and C++ share ATen kernels; there is no evidence here that switching front-end alone solves throughput.

Earlier same-session production evidence remains separate: optimized native 100,007,936-parent frontier, depth7/8 =179.381/179.310s, about558k parents/s. That pipeline was not rerun for each backend in this comparison.

## Output checks and limitations

- Static Python token construction and full logits are bitexact against the original Python implementation on97 varied random states (fixed seed20260915, not necessarily reachable cube states).
- Python eager/graph equality is asserted at each measured batch. All saved Python and LibTorch first-parent key vectors agree on the repeated benchmark input.
- All30 native saved first-parent vectors agree across full/optimized modes, batches and repetitions. Compared with Torch, maximum key difference is23, equivalent to23/1024 =0.0224609375 score units. Native and Torch FP16/fusion rounding are **not bitexact**. This test does not qualify beam-decision parity over real frontiers.
- Earlier compact57 correctness and sanitizer evidence is in `h200_single_2026-09-15.md`; do not replace it with this one-state timing check.
- Existing native printed TFLOPS estimator is not reliable for generic final-CLS Cube4. It was not used to infer hardware utilization or speedup.

## Historical hardware evidence

These are recovered saved runs, not fresh T4/P100/3070 measurements. Raw relevant depth logs and historical reports are copied to `h200_backend_history/`.

| Hardware and workload | Scope | Parents/s |
|---|---|---:|
| 1xT4, Cube4 S57/d256/4 blocks/FF1024, LibTorch eager, batch384 | Full pipeline, beam4194304, depth6–9 | 24824 / 23773 / 23269 / 23137 |
| 1xP100, same Cube4 shape and LibTorch pipeline | Full pipeline, beam4194304, depth6–9 | 13964 / 13937 / 13907 / 13890 |
| 1xT4 equivalent, Megaminx S51 transformer, LibTorch eager | Isolated per-GPU mean, July5 stability report | 30519 / 30624 on the two measured T4s |
| 2xT4, same Megaminx transformer, LibTorch eager | Sum of both GPUs' isolated mean rates | 61143 |
| RTX3070 Laptop SM86, Megaminx S51 transformer, native final-CLS, micro512/c2 | Isolated July8 graph run | 39794 |

T4/P100 Cube4 production depth times:

- T4 depth6–9:168.960,176.428,180.254,181.282 seconds.
- P100 depth6–9:300.376,300.951,301.602,301.955 seconds.
- Rate is input frontier4194304 divided by whole-depth seconds, **not** next-frontier size divided by an unrelated interval. Both runs use `--nproc-per-node=1`; the Kaggle host showing two T4s does not imply both were used.
- All57 exported files listed in current `weights_sha256.txt` match the saved T4 Cube4 export by SHA256. This was checked locally against `D:/100XH100/test_results/paper_benchmarks/cube4_ours_t4_b4m_depth10/stream1_transformer_weights_fp16/`. The P100 comparison confirms shape/backend, not an independently verified full weight hash set.
- Megaminx rates were reported as candidates/s and converted by24: 732462/24=30519.25,734979.7/24=30624.15,1467441.7/24=61143.40. It is a similar transformer, not the same Cube4 model (S51 and SiLU versus S57 and ReLU).
- RTX3070 figure955065.5 candidates/s /24 =39794.40 parents/s; it is an older implementation and different model.

Original local historical sources:

- `D:/100XH100/test_results/paper_benchmarks/cube4_ours_t4_b4m_depth10/`
- `D:/100XH100/test_results/paper_benchmarks/cube4_ours_p100_b4m_depth10/`
- `D:/100XH100/test_results/molab_cube4_sm120_src/test_results/stream1_libtorch_transformer_stability_v11_2026-07-05.md`
- `D:/100XH100/test_results/molab_cube4_sm120_src/test_results/stream1_transformer_sm80_tensorcore_dispatch_2026-07-08.md`

Current H200 optimized pipeline / old T4 pipeline is roughly22–24x, but backend, software version, beam and workload distribution differ. It is a historical system comparison, not a controlled hardware-only acceleration factor. There is no evidence that T4 was faster on this transformer.

## Hardware context

NVIDIA lists H200 SXM FP16 Tensor Core1979TFLOPS **with sparsity**; this corresponds to about989.5 dense TFLOPS, versus T4's65 FP16 TFLOPS. Bandwidth is4.8TB/s versus320+GB/s. These imply roughly15x peak compute/bandwidth ratios, not guaranteed end-to-end speed ratios. Small matrix dimensions, normalization, attention, layouts and scheduling remain relevant.

- https://www.nvidia.com/en-us/data-center/h200/
- https://www.nvidia.com/en-gb/data-center/tesla-t4/

## Reproduction and preservation

Benchmark-only helpers: `hpc/h200_compare_torch.py`, `hpc/h200_compare_native.sh`, `hpc/h200_compare_summary.py`.
LibTorch target: `stream1_transformer_libtorch_benchmark`, CMake `BEAM_ENABLE_LIBTORCH_STREAM1=ON`, `CMAKE_PREFIX_PATH=/venv/main/lib/python3.12/site-packages/torch/share/cmake`, built separately with Cube4 state96 and CUDA90a.

```bash
/workspace/build-libtorch/stream1_transformer_libtorch_benchmark --weight-dir /workspace --batches 128,256,384,512,768,1024,2048,4096 --concurrency 1 --warmup 10 --iters 100 --passes 3 --csv /workspace/results/compare_libtorch_eager.csv
# Repeat with --cuda-graph and a different CSV/log name.
/venv/main/bin/python /workspace/MGBFS/hpc/h200_compare_torch.py --output /workspace/results/compare_python.csv
bash /workspace/MGBFS/hpc/h200_compare_native.sh
/venv/main/bin/python /workspace/MGBFS/hpc/h200_compare_summary.py
```

`h200_backend_comparison_evidence.tar.gz` contains all current raw CSV/logs, native score dumps, per-run reports, summary and weight hashes. Historical subset is under `h200_backend_history/`.
Archive SHA256: `6bc107dc0ae884876ba94e0bb061cece4c3945fc21f6af4dc4607b9147d3e11e`.

After tests, instance51111383 was stopped via its scoped Vast CLI and browser status verified **Inactive**, $0.112/hr retained disk. Credit changed from836.99 to836.53 (about$0.46 during this comparison window; not an invoice-exact isolated compute bill). No GPU job remains active. Instance was not destroyed.
