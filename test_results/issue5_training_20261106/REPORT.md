# Pancake-10 trained model comparison

The dual-stream GATv2 encoder found an optimal path in **268 of 300** searches. Pilgrim found 219 and the Level-1 Transformer found 107. Each model was trained from three initializations and evaluated on the same 100 published Pancake-10 starts each time. Every returned path replayed to the identity. [Machine-readable summary](summary.json)

| Model | Trainable parameters¹ | Solved | Optimal | Mean flips on successes | Mean flips over optimal on successes |
|---|---:|---:|---:|---:|---:|
| Pilgrim MLP | 501,249 | 300/300 | 219 | 8.870 | 0.300 |
| GATv2 → shared neighbor GNN | 48,170 | 300/300 | 268 | 8.677 | 0.107 |
| Transformer → shared neighbor GNN | 48,490 | 294/300 | 107 | 9.680 | 1.136 |

¹ Parameter counts include the two-level models' policy heads. The common scalar objective supplies gradients through the value path.

The Transformer missed six starts at seed 42. On the 294 matched starts/seeds solved by all three, mean lengths were 8.837 for Pilgrim, 8.650 for GATv2 and 9.680 for the Transformer. The exact mean optimal distance of the 100 unique starts was 8.57. Repeating the starts measures sensitivity to weight initialization. [Paired results and per-seed metrics](summary.json)

| Seed | Pilgrim solved / optimal | GATv2 solved / optimal | Transformer solved / optimal |
|---|---:|---:|---:|
| 42 | 100 / 74 | 100 / 87 | 94 / 31 |
| 43 | 100 / 73 | 100 / 89 | 100 / 48 |
| 44 | 100 / 72 | 100 / 92 | 100 / 28 |

## Models and source compatibility

MultiGPUBeamSearch publishes inference adapters. For training we used the original trainer of Pilgrim, one of its explicitly supported baseline models. The submitter's pinned implementation is a Python file. It contains both encoders and the shared neighbor network. Its entrypoint selects only the value-graph encoder and names a private resume checkpoint. The public tree supplies neither those weights nor a fixed RL evaluation split. This experiment trains all nine checkpoints from scratch. [Host model interface](https://github.com/TryDotAtwo/MultiGPUBeamSearch/blob/de59281ca5fe6f32b04685a9c999118ca1c6c6c2/integrations/cayleypy_native/README.md#L211-L225) · [Author's model definitions](https://github.com/Alimbj/Two_level_GNN_for_pancacke_graph/blob/ef0157276dd94330e36fc4668d15fabb9973b13f/pancake_RL.py#L948-L1438) · [Author's training entrypoint](https://github.com/Alimbj/Two_level_GNN_for_pancacke_graph/blob/ef0157276dd94330e36fc4668d15fabb9973b13f/pancake_RL.py#L1856-L1948)

* **GATv2.** Nodes are pancake values. Two directed graphs connect neighboring values and neighboring positions in the current stack. Forward, reverse and self edges have different types. The readout combines mean, maximum, top and bottom stack values, and the smallest and largest values. Training applies dropout to attention probabilities when configured.
* **Transformer.** Each token concatenates a learned value embedding and a learned position embedding. Two pre-norm four-head layers use GELU and feedforward width 128. Mean pooling and LayerNorm yield the state embedding.
* **Shared Level 2.** Both encode prefix-flip neighbors through their selected Level 1. Each hop aggregates child, action and hop features back to the original root. A learned query attends over the center and both hop updates. We used width 32 and deterministic stratified sampling with maximum 3 moves at hop one and 2 at hop two. The full sampled expansion fits the configured cap. The two variants started with identical Level-2 and head weights within each seed and scalar bias `9n/11`.

We exported both trained seed-42 checkpoints using the public native-model preparation path and loaded them strictly into the author's corresponding models. For 17 fixed permutations the Level-1 vectors and complete scalar values matched the author's FP32 outputs exactly on CPU for both variants. Reimported FP16 TorchScript on CUDA differed from the author's FP32 scalar by at most 0.00760 for GATv2 and 0.00706 for the Transformer. [Parity receipt](seed42/reference-parity.json) · [Verification code](verify_reference.py)

We also compiled the production C++ GNN loader and CUDA projection kernels for SM86 and loaded both trained artifacts through LibTorch and CUTLASS. On 17 permutations, the largest absolute difference from eager FP16 features was 0.00391/0.00586 for GATv2 and 0.00000/0.00195 for the Transformer (LibTorch/CUTLASS). This measures native feature extraction. The path-quality table uses Pilgrim's full search for all three models. The adapter test suite at the documented CayleyPy pin passed 465 tests with 8 skips. [C++/CUDA receipt](seed42/native-features.json) · [Native probe](verify_native.py) · [Test record](checks.json)

## Common training and evaluation protocol

* **Data.** We used upstream group `p044`, its identity target and its 100 fixed random test permutations. We retained all 18 ordered generator names in the upstream trainer and searcher, including both aliases of each of the nine physical flips. Level 2 uses the nine distinct physical flips. At each epoch all three arms received the same tensors, draw order and labels from `Trainer.generate_random_walks`. Labels are walk lengths 1–15. A row-by-row comparison found zero test starts in the training tensors. [Public starts](https://github.com/khoruzhii/cayleypy-cube/blob/f02604fa7b665b82e5fbe5692b336a4fe4a01bdc/datasets/p044-t000-rnd.pt) · [Generator order](https://github.com/khoruzhii/cayleypy-cube/blob/f02604fa7b665b82e5fbe5692b336a4fe4a01bdc/generators/p044.json)
* **Optimization.** We called the unmodified Pilgrim `Trainer._train_epoch` for each arm. Each saw 8 epochs × 256 walkers × 15 depths = 30,720 examples, 480 Adam updates with batch 64, learning rate 0.001, FP32, MSE and dropout 0. Pilgrim retained the published 1024/256 architecture with one residual block. The value-only objective gives the two-level models the same labels as Pilgrim. The submitter's separate Bellman/replay trainer also optimizes a policy head. [Upstream trainer](https://github.com/khoruzhii/cayleypy-cube/blob/f02604fa7b665b82e5fbe5692b336a4fe4a01bdc/pilgrim/trainer.py) · [Three matched runs](compare.py)
* **Search and validation.** All nine trained models used upstream Pilgrim `Searcher`, FP16, width 32, one attempt, 30 steps and the same hash seed. All training and search ran on one RTX 3070 Laptop GPU with PyTorch 2.8.0+cu128. We replayed physical prefix lengths on plain tuples. Exact distances came from meeting a breadth-first ball of radius five around the identity. The separate audit checks every saved path, optimal length, checkpoint digest and train/test overlap receipt. [Upstream searcher](https://github.com/khoruzhii/cayleypy-cube/blob/f02604fa7b665b82e5fbe5692b336a4fe4a01bdc/pilgrim/searcher.py) · [Audit](audit.py)
* **Scale.** The published Pilgrim Table 4 recipe trains up to 128 epochs with about one million examples per epoch, batch 10,000, and evaluates the first 10 starts at beam 1,048,576. This comparison uses a smaller common training and beam budget to fit the two-level models on one 8 GB GPU. We evaluated all 100 starts and retained the first-ten metrics separately. Use the numbers above for this Pancake-10 protocol. The author's Pancake-100 RL run, larger beams and multi-GPU throughput are separate experiments. [Published reference schedule](https://github.com/khoruzhii/cayleypy-cube/blob/f02604fa7b665b82e5fbe5692b336a4fe4a01bdc/traintest-tab4-rnd.sh#L14-L77)

## Reproduction

Use a CUDA-enabled PyTorch environment with pandas, numpy, tqdm and the project's Python dependencies. The author parity check additionally uses torch-geometric. Pin the two external checkouts shown below. The scripts take local paths and run from a checkout of this repository.

```sh
git clone https://github.com/khoruzhii/cayleypy-cube.git pilgrim-source
git -C pilgrim-source checkout f02604fa7b665b82e5fbe5692b336a4fe4a01bdc
git clone https://github.com/Alimbj/Two_level_GNN_for_pancacke_graph.git author-source
git -C author-source checkout ef0157276dd94330e36fc4668d15fabb9973b13f
work=test_results/issue5_training_20261106
for seed in 42 43 44; do
    python "$work/compare.py" --pilgrim-root pilgrim-source --seed "$seed" --out "reproduction/seed$seed"
done
python "$work/verify_reference.py" --author-source author-source/pancake_RL.py --out reproduction/seed42
python "$work/audit.py" --dataset pilgrim-source/datasets/p044-t000-rnd.pt --results reproduction
```

To repeat the C++ feature comparison, provide a CUDA compiler, Ninja, a compatible host compiler and CUTLASS at commit `7107b05535f8977f5ecb9d01ee203205b1fd9bc4`. Run `python "$work/verify_native.py" --cutlass /path/to/cutlass --out reproduction/seed42`. The recorded run used the RTX 3070, NVCC 12.5, MSVC 14.44 and the Windows PyTorch 2.8.0 CUDA 12.8 package.

Each seed directory contains three state dictionaries, per-epoch training losses, hashes of the ordered training tensors and checkpoints, and the returned flip sequence or failure for every test start. GPU floating-point reduction order may vary across toolchains. [Seed 42](seed42) · [Seed 43](seed43) · [Seed 44](seed44)
