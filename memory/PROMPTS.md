2026-10-10 user requirement: Update existing public Kaggle notebook and leave only one GitHub branch in MultiGPUBeamSearch. Preserve checked source and historical reproducibility evidence. No additional GPU rentals.

## 2026-10-10 GNN requirements
Implement our own support for the two-level GNN architecture in issue #5, without using the submitted implementation as runtime. Provide both LibTorch and CUTLASS backends and preserve the five-stream search architecture.


GNN acceptance rental authorized: up to $5 total including disk/traffic. Implement our own GNN from issue5 with both LibTorch and CUTLASS; preserve beam architecture and do not use subagents.


2026-10-10: User requested running a large GNN comparison after the tiny fixture; approved a new RTX3060 rental up to $5 total. Compare matched Python/PyTorch, native LibTorch and native CUTLASS inference, archive results via GitHub and remove own rental.


## 2026-10-10: native GNN CUTLASS fusion
Requirement: improve large GNN native CUTLASS by at least 2x, preserving beam architecture and numerical correctness; profiling rental authorized up to $5.


## 2026-10-10: GNN fused rounded bias
User requested separate bias removal, RTX3060 speed/precision verification, an active goal and authorized $5 total for new tests.


## 2026-10-10: further GNN optimization with Astra
Continue improving GNN while measured gains remain, consult Astra as expert, preserve architecture; new GPU-test budget $10 total.

2026-10-11: Fix slow results publication and misleading published status. Keep the checked Cube555 solver and GitHub-only inputs; no new GPU execution.

2026-10-25: User authorized publishing the verified Cube555/API/autotuner changes to GitHub and updating the existing public Kaggle notebook. Use GitHub-pinned source and inputs; preserve requested beam and explicit blend.

2026-10-27: Run two Cube555 Kaggle searches at beam 2,500,000 on 2×T4 using the reference blend notebook settings: a legal scramble around 10–12 moves and a puzzle around 100 moves. Do not silently reduce requested beam. For unstable component-calibration baseline timings, permit one bounded retry and report failure explicitly if still unstable.

2026-10-31: Ivan asked whether the Cube555 helper can be omitted so the notebook is “through CayleyPy” rather than exposing unfamiliar MultiGPUBeamSearch modules. Keep graph search on CayleyPy's graph.beam_search API; puzzle/model loading and result handling may be optional convenience helpers, while native multi-GPU activation/session still requires the current adapter. Do not claim the public interface is CayleyPy-only until CayleyPy or its official optional backend exposes that native integration.


2026-10-31 clarification: Keep `import multigpubeamsearch`. Remove only the puzzle-specific `multigpubeamsearch.cube555` helper; configure other problems with a CayleyPy graph, supported predictor/model, and start state, then use the common beam-search API.
