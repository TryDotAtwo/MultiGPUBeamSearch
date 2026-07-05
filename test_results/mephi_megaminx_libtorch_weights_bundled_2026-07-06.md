# Bundled Vlad Megaminx Transformer Weights

Date: 2026-07-06

Scope:

- Add the exported fp16 `piece_transformer` weights used by the LibTorch Stream1 path to the repository.
- Update the MEPhI 8xA100 launcher so the cluster run uses the repo-bundled weights by default.

Bundled path:

```text
weights/megaminx_vlad_transformer_fp16/manifest.json
```

Inventory:

```text
files=61
bytes=6544840
mib=6.24
largest_file_bytes=524288
```

The files are small enough for ordinary GitHub storage; no Git LFS is required.

Launcher behavior:

- If `BEAM_WEIGHT_DIR` is unset and `repo/weights/megaminx_vlad_transformer_fp16/manifest.json` exists, use that path.
- If `BEAM_WEIGHT_DIR` is set, use the override unchanged.
- If neither exported weights nor a full `.pth` model dump exists, fail before the compute run.