# CayleyPy native automatic Kaggle example

Public notebook: https://www.kaggle.com/code/trydotatwo/cayleypy-multigpubeamsearch
Public runtime dataset: https://www.kaggle.com/datasets/trydotatwo/cayleypy-native-runtime
CayleyPy (Private Group) has Can edit on the public notebook.

Open the public notebook and choose Copy & Edit (team editors can edit directly). GPU and Internet are configured, and the runtime dataset is attached. Set the graph/start, BEAM_WIDTH and NUM_GPUS, then Run All. BACKEND="native" selects this implementation; BACKEND="torch" selects ordinary CayleyPy. Default scoring is untrained Hamming. The first native run builds and caches the runner; Linux CUDA toolkit/nvcc, driver and compiler are platform requirements, checked before execution.

Version 1 completed on two physical Tesla T4 GPUs with automatic LibTorch selection, CayleyPy 0.2.0 and Torch 2.11.0+cu128. Its LRX4 example found and replayed a four-move path. This proves this example on the current Kaggle image; it does not prove compatibility with old Torch versions or performance on large graphs. Full graph matrix evidence is tracked separately.

Accepted wheel SHA256: 9d91f4e0c66ee927d5343726f6c34574529656a0a73b6a8eb093cfd535ebbd13.
Evidence: ../../test_results/kaggle_public_auto_20261007/REPORT.md and output_manifest.json.
Local .ipynb and kernel-metadata.json mirror the published configuration. The refreshed zip includes this README, the notebook and the accepted wheel for offline handoff.

Canonical distribution/import: multigpubeamsearch 0.1.0.dev1. The notebook uses real import multigpubeamsearch, without a Python alias. This package is distributed as the attached wheel; no PyPI publication is implied. Legacy cayleypy_native imports forward to the same implementation.
New wheel SHA256: d67910a1dc8bbee31cd11af75d39b2fa3062e1cae9bc5c1cf3d380f59ad9e7b5
Version 3 acceptance is tracked under ../../test_results/multigpubeamsearch_rename_20261007.
