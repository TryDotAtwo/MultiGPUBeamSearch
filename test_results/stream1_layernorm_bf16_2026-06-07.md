# Stream1 LayerNorm/BF16 Verification - 2026-06-07

Changes verified:

- `tools/export_stream1_mlp.py` parses successfully with Python AST.
- Synthetic Artgor-style `ResMLPDistance` checkpoint exported with:
  - `--format resmlp-layernorm`
  - `--dtype bf16`
  - output directory `test_results/fake_resmlp_stream1_bf16`
- Export manifest contains:
  - `state_len=120`
  - `num_classes=120`
  - `hd1=2048`
  - `hd2=512`
  - `nrd=2`
  - `output_dim=1`
  - `dtype=bf16`
  - `normalization=layernorm`
- CUDA build check passed for target `production_runner` using:
  - Docker image `gpu-dev-cutlass-nsight:2026-05-24`
  - `CUTLASS_DIR=/opt/cutlass`
  - build directory `build-gpu-dev-cutlass`
- The LayerNorm input path was checked so folded embedding+linear writes the
  pre-bias activation; `input_bias` is added exactly once inside LayerNorm.
- The existing fp16 no-LayerNorm folded-input hot path keeps its `half2` bias
  and weight reads; dtype-aware scalar reads are used only outside that path.

Notes:

- `gpu-dev:latest` did not contain `/opt/cutlass`; CMake stopped at
  `CUTLASS_DIR is required`.
- No fallback inference kernel is used. LayerNorm is inserted into the primary
  Stream1 CUTLASS pipeline when `normalization=layernorm`; models without
  LayerNorm keep the previous path.
