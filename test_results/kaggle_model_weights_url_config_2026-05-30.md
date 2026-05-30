# Kaggle model weights URL config 2026-05-30

task_id=kaggle_model_weights_url_config
environment=local PowerShell, notebook JSON validation

## Change

- Added `STREAM1_MODEL_WEIGHTS_URL` as the first config line in `kaggle/cayley-beam-gpu-runner.ipynb`.
- Mirrored the same change to `kaggle/beam_kernel.ipynb`.
- Empty `STREAM1_MODEL_WEIGHTS_URL` keeps the repository `stream1_weights` directory.
- Non-empty `STREAM1_MODEL_WEIGHTS_URL` downloads a direct `.pth` URL or copies an absolute local/Kaggle path to `/tmp/stream1_model_weights.pth`.
- The notebook then runs `tools/export_stream1_mlp.py --weights /tmp/stream1_model_weights.pth --out /tmp/beam_solver/stream1_weights`.
- `output_dim` remains inferred by the exporter and runtime manifest, so the notebook supports both one-output and twenty-four-output Stream1 models.

## Verification

```text
command=$nb = Get-Content -Path kaggle\cayley-beam-gpu-runner.ipynb -Raw | ConvertFrom-Json; $nb.cells[0].source[0]
result=pass
evidence=STREAM1_MODEL_WEIGHTS_URL first line
```

```text
command=$nb = Get-Content -Path kaggle\beam_kernel.ipynb -Raw | ConvertFrom-Json; $nb.cells[0].source[0]
result=pass
evidence=STREAM1_MODEL_WEIGHTS_URL first line
```

```text
command=python read-only notebook code compile check for both Kaggle notebooks
result=pass
evidence=notebook_compile=pass for cayley-beam-gpu-runner.ipynb and beam_kernel.ipynb
```

## Remaining

- Kaggle push required after local validation.
