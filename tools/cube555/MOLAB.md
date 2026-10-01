# Cube555 on Molab

Select one GPU, edit the first cell if needed and choose **Run All**.
There are no Prepare/Check/Run search widgets and no Kaggle credentials.
Model and competition files download from the public GitHub Release;
the ZIP and every member are SHA-256 verified.

Defaults: beam 3,100,000; depth 140; Transformer weight 0.8; BFS radius 5;
collect up to 2000 solutions, then one additional depth after the first hit.
Set AUTHOR_NAME to your public name. Every saved path is replayed before sending.

The notebook builds the pinned native solver for the observed architecture,
runs one real GPU rank in a foreground cell and attempts Cloudflare publication
with explicit Molab provenance. Publishing errors are saved, never fatal to
local results. Download the results ZIP before ending the sandbox session.

CUDA PyTorch, NVCC supporting the GPU, Git, CMake, Ninja and PyTorch NCCL are
checked automatically. Microbatch128/concurrency1 are T4 seed settings.
Molab GPU build and speed are not verified yet. Kaggle timing forecasts do not
apply to Molab. History preflight may cap depth with a warning, never beam.

The explicit Molab server schema must be deployed before publication succeeds.
This remains a separate validation gate from the source checks.
