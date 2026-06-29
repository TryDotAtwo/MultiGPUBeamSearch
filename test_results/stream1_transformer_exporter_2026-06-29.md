# Stream1 Piece Transformer Exporter Verification - 2026-06-29

## Scope
- Added `tools/export_stream1.py` auto-detect wrapper.
- Added `tools/export_stream1_transformer.py` for Kaggle p900 `piece_transformer` checkpoints.
- Added Python unit coverage in `tests/test_stream1_transformer_exporter.py`.
- Tightened exporter validation to reject internally consistent non-target variants: non-24 move/output checkpoints, non-4-layer checkpoints, and non-1024-FF checkpoints.

## Commands

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; py -B -m unittest discover -s tests -p "test_stream1_transformer_exporter.py" -v
```

Result: PASS, 6 tests. Coverage includes happy-path inference, prefix stripping/detection, fast input table generation, and negative rejection tests for 23-output, 3-layer, and 2048-FF synthetic checkpoints.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; py -B tools\export_stream1.py --weights 'C:\tmp\megaminx_qtransformer\megaminx-transformer\weights\p900-t000-q-rw-sym_1782210824_best.pth' --out test_results\stream1_transformer_reference\weights_fp16 --dtype fp16 --format auto --num-classes 120 --reference-out test_results\stream1_transformer_reference\reference.json --reference-count 8 --reference-seed 12345
```

Result:

```text
stream1_export_done out_dir=test_results\stream1_transformer_reference\weights_fp16 format=piece_transformer dtype=fp16 seq_len=51 d_model=256 layers=4 output_dim=24
```

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; py -B tools\export_stream1_mlp.py --help
```

Result: PASS, help printed for the existing MLP exporter CLI.

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; py -B -c "from pathlib import Path; [compile(Path(p).read_text(encoding='utf-8-sig'), p, 'exec') for p in ['tools/export_stream1.py','tools/export_stream1_transformer.py','tools/export_stream1_mlp.py','tests/test_stream1_transformer_exporter.py']]; print('syntax_ok')"
```

Result: PASS, `syntax_ok`.

## Export Checks

Manifest fields from `test_results/stream1_transformer_reference/weights_fp16/manifest.json`:

```text
backend=piece_transformer
output_dim=24
seq_len=51
num_layers=4
ff_dim=1024
dtype=fp16
```

Reference JSON checks:

```text
states=8x120
scores_fp32=8x24
```

Exported file count and selected sizes:

```text
file_count=61
manifest.json=1276 bytes
fast_slot_projected.fp16=184320 bytes
fast_piece_static.fp16=25600 bytes
cls_token.fp16=512 bytes
output_weight_hxk.fp16=12288 bytes
piece_positions.u16=300 bytes
piece_mask.u8=150 bytes
piece_types.u8=50 bytes
```

## Notes
- `py_compile` was not used because the sandbox denied writes to `tools/__pycache__`; the no-bytecode `compile(...)` syntax check was used instead.
- The local Kaggle checkpoint existed at the requested path and was used for the real export/reference check.
