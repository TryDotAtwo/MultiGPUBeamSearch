# Stream1 LibTorch Transformer Stability v11

Date: 2026-07-05

Scope:

- Validate the explicit C++ LibTorch Stream1 `piece_transformer` benchmark with repeated passes on Kaggle 2xT4.
- Use the same exported-weight contract as the PyTorch and native CUTLASS backend comparisons.
- Preserve explicit backend selection only: no fallback and no distillation.

Kaggle kernel:

```text
trydotatwo/cayley-beam-transformer-libtorch-2xt4-benchmark
status=COMPLETE
checked_out_commit=ee141bc
```

Downloaded artifacts:

```text
test_results/kaggle_libtorch_transformer_stability_v11_2026-07-05/
```

Configuration:

```text
modes=eager,cuda_graph
batches=80,256,320,384,448,512,640
passes=3
reference_2xt4_pytorch_batch_process=1421505.5 candidates/s
```

Aggregate results:

```text
eager_mean_2xt4=1467441.7 candidates/s
eager_best_2xt4=1497577.0 candidates/s
cuda_graph_mean_2xt4=1411365.0 candidates/s
cuda_graph_best_2xt4=1442113.0 candidates/s
eager_mean_over_pytorch_reference=1.0323x
cuda_graph_mean_over_pytorch_reference=0.9929x
cuda_graph_best_over_eager_best=0.9630x
```

Best mean rows:

```text
eager gpu0 batch=384 passes=3 mean=732462.0 best=744394.0 min=721839.0
eager gpu1 batch=384 passes=3 mean=734979.7 best=752571.0 min=720075.0
cuda_graph gpu0 batch=384 passes=3 mean=716360.7 best=726257.0 min=706804.0
cuda_graph gpu1 batch=80  passes=3 mean=695004.3 best=709916.0 min=681718.0
```

Interpretation:

- Repeated passes reduce the v10 single-row optimism but keep the same production direction: explicit LibTorch eager is the current fastest stable exported-weight transformer path on 2xT4.
- CUDA Graph capture remains functional, but it is not the current default candidate because it is slower than eager under the same repeated-pass contract.
- Production integration should use a named non-graph `libtorch:eager` Stream1 path and fail closed if the opt-in LibTorch build/runtime is unavailable.