# MEPhI Stream1 Row Budget For Artgor - 2026-06-07

Context:

- Old model: `stream1_weights`, `output_dim=24`.
- New Artgor model: `stream1_weights_artgor_bf16`, `output_dim=1`.
- Requested MEPhI setting: `BEAM_B_MICRO=8192`, `BEAM_STREAM3_RING_SLOTS=8`.

Code finding:

- C++ already treats `BEAM_B_MICRO` as a Stream1 row budget in
  `build_runtime_config_from_budget()`:
  `config.b_micro = stream1_parent_batch_from_row_budget(row_budget, model)`.
- MEPhI shell derivation still used `BEAM_B_MICRO` as a parent count when
  computing `STREAM3_BATCH_CANDIDATES`.

Fix:

- `hpc/mephi_8xa100_common.sh` now reads the selected
  `${BEAM_WEIGHT_DIR}/manifest.json`.
- For `output_dim=1`, it sets:
  - `BEAM_PARENT_BATCH_EFFECTIVE = BEAM_B_MICRO / 24`
  - `STREAM1_ROWS_PER_JOB_EFFECTIVE = BEAM_PARENT_BATCH_EFFECTIVE * 24`
  - `STREAM3_BATCH_CANDIDATES = BEAM_STREAM3_RING_SLOTS * BEAM_PARENT_BATCH_EFFECTIVE * 24`
- For `output_dim=24`, it keeps:
  - `BEAM_PARENT_BATCH_EFFECTIVE = BEAM_B_MICRO`

Verification calculation:

| Model | output_dim | row budget | effective parent batch | rows/job | stream3 batch |
|---|---:|---:|---:|---:|---:|
| `stream1_weights` | 24 | 8192 | 8192 | 8192 | 1572864 |
| `stream1_weights_artgor_bf16` | 1 | 8192 | 341 | 8184 | 65472 |

Checks:

- Docker `bash -n` passed for:
  - `hpc/mephi_8xa100_common.sh`
  - `hpc/start_8xa100_best.sh`
  - `hpc/solve_then_reflect.sh`
- Local manifest calculation produced the table above.
