# CayleyPy public checkpoint CLI gate - 2026-07-29

## Scope

- Added `tools/run_cayleypy_public.py` as the only notebook execution entry point.
- Public config is closed-world and checkpoint-only; `MODEL_SOURCE`, `MODEL_DTYPE`, and `CHECKPOINT_FORMAT` are rejected.
- Automatic checkpoint detection supports `batchnorm-folded` and `resmlp-layernorm` only.
- Export dtype is fixed automatically to FP16.
- Model heads are limited to `output_dim=1` or `output_dim=move_count`.
- Hardware preflight requires exactly two Tesla/NVIDIA T4 GPUs.
- User beam is retained after required alignment; the nearest measured p16..p25 profile supplies runtime knobs.
- Standard CayleyPy puzzle data, strict inclusive IDs, reflection, first/collect, and touch-BFS controls are wired through existing helpers.
- Release SM75 `production_runner` is built without source edits against pinned CUTLASS `afa1772203677c5118fcd82537a9c8fefbcc7008`.
- Solve, partial-failure, submission, profile, preflight, solution, log, and best-effort publication artifacts are materialized.

## Security and failure semantics

- Checkpoints are loaded with `torch.load(..., weights_only=True)` in detection and both exporters.
- Unsupported and unknown public config fields fail closed.
- Publication provenance requires exact Kaggle and solver hashes when enabled.
- Build logs and public errors redact configured paths, home/repository paths, URL credentials, and common token forms.
- GitHub/ingest failure cannot turn a successful solve into a failed solve; `publish_status.json` records a bounded safe reason.

## Verification

- `py -m pytest tests/cayleypy_public -q`: 176 passed.
- `py -m pytest -q`: 209 passed.
- `py -m compileall -q tools/cayleypy_public tools/run_cayleypy_public.py tools/export_stream1_mlp.py`: passed.
- `git diff --check`: passed.
