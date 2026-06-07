# Artgor Stream1 Model Cluster Prep - 2026-06-07

Prepared the Artgor `ResMLPDistance` value-only checkpoint for the MEPhI
8xA100 runner.

Downloaded from Kaggle dataset:

- Dataset: `artgor/megaminx-tpu-artifacts`
- File: `m_az_v4_v_only.pt`
- Local path: `test_results/artgor_model/m_az_v4_v_only.pt`
- Kaggle-reported license: `CC0-1.0`

Exported Stream1 weights:

- Command:
  `python tools/export_stream1_mlp.py --weights test_results/artgor_model/m_az_v4_v_only.pt --out stream1_weights_artgor_bf16 --format resmlp-layernorm --dtype bf16`
- Output directory: `stream1_weights_artgor_bf16`
- Manifest:
  - `hd1=2048`
  - `hd2=512`
  - `nrd=2`
  - `output_dim=1`
  - `dtype=bf16`
  - `normalization=layernorm`
- Exported payload size: about 60 MiB.

Cluster transfer artifact:

- `test_results/artgor_stream1_weights_bf16_2026-06-07.tar.gz`
- Archive size: about 48 MiB.

Code/runtime preparation:

- `hpc/mephi_8xa100_common.sh` now preserves a caller-provided
  `BEAM_WEIGHT_DIR`; default remains `${REPO_DIR}/stream1_weights`.
- Docker `bash -n` passed for:
  - `hpc/mephi_8xa100_common.sh`
  - `hpc/start_8xa100_best.sh`
  - `hpc/solve_then_reflect.sh`

GPU runtime was not executed locally because Docker does not have an NVIDIA
driver attached in this environment.
