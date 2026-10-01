# Cube555 on Molab

Open `molab_notebook.py` through Molab's GitHub importer. The first cell holds
model paths, puzzle range, beam, depth, blend, reflection and collection settings.
Defaults match Kaggle v16: beam 3,100,000, depth 140, Transformer weight 0.8,
collect 2000 solutions with one extra depth, BFS radius 5.

This notebook runs one native rank on one actual CUDA GPU. The Python launcher
chooses the observed compute capability for CMake. It never simulates two ranks
on one GPU. Model microbatch128/concurrency1 are inherited T4 settings, not a
Molab tuning result. Native CUDA memory preflight still applies; history budgets
can reduce depth and report that reduction, while beam stays unchanged.

## Launch

1. Attach one GPU in Molab. The environment needs CUDA-enabled PyTorch, NVCC
   supporting that GPU, Git, CMake, Ninja and the PyTorch NCCL headers/library.
2. Put the Kaggle credential in Molab Secrets as `KAGGLE_API_TOKEN`, and accept
   the competition rules on Kaggle. Credentials never enter notebook source.
   Alternatively upload the four public model bundle files and competition
   `test.csv`/`sample_submission.csv` to the first-cell paths.
3. Click Prepare, then Check GPU. Both stages are explicitly gated.
4. Select Short replay smoke first. It solves/replays four legal scrambles.
   Select Configured search afterwards and press Run search deliberately.
5. Keep the foreground computation attached to the notebook. Download the ZIP
   in the Molab file browser after each run; automatic persistence of generated
   sandbox files is not assumed.

Automatic Cloudflare publication is disabled: the current public ingestion
format requires Kaggle execution provenance. This notebook saves replayed
solutions without falsely claiming Kaggle execution. The launcher rejects
`--publish` on this target until Molab provenance has separate support.

## Evidence

127 local Python tests pass, covering default two-rank behavior and the new
single-rank beam/history/torchrun/log contract. No Molab GPU search or native
SM120 build has been verified yet. The Kaggle runtime forecasts do not apply
to Molab. The runtime source is pinned in the notebook independently of its
notebook-document commit.
