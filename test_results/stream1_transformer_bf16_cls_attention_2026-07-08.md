# Stream1 Transformer BF16 and CLS-Attention Smoke

Date: 2026-07-08

Scope:

- Add explicit final-layer CLS-attention switch for the block51 native transformer path.
- Keep production default behavior selectable: `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=0/1`.
- Add BF16 benchmark/export wiring for the Megaminx transformer HPC launcher.
- Keep MLP and Stream 4 paths unchanged.

Implementation notes:

- Added `stream1_transformer_fmha_cls_attention_cuda`, a CUTLASS FMHA call with `num_queries=1` and `num_keys=51`.
- `BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION=1` uses CLS-only FMHA in the final transformer block after QKV projection.
- `hpc/bench_8xa100_megaminx_transformer.sh` now supports `BEAM_STREAM1_EXPORT_DTYPE=fp16|bf16` and isolated native sweeps over `ISOLATED_FINAL_CLS_ATTENTION_SWEEP`.
- Python Torch benchmark now accepts both `fp16` and `bf16` manifests.

Local 3070 native benchmark, real exported fp16 weights, graph mode, `b_micro=512`, `concurrency=2`:

```text
old full final FMHA: candidates_per_sec=978307.4 checksum=1242545152 digest=3860563098260702083
new CLS-only FMHA:  candidates_per_sec=948191.8 checksum=1242545152 digest=3860563098260702083
```

Interpretation: the CLS-only CUTLASS FMHA path is numerically identical on this smoke, but slower on 3070 for this batch shape. It is therefore exposed for A100 sweep instead of being forced globally.

Local 3070 BF16 smoke using temporary fp16-export-to-bf16 converted weights under `test_results/stream1_transformer_weights_bf16_from_fp16_smoke`:

```text
native bf16 graph b512 c2: candidates_per_sec=1035658.3 checksum=1242650624 digest=11726401608200844163
python torch bf16 b512 c2: candidates_per_s=630037.0 checksum=620363776 digest=4543884025843041155
libtorch bf16 CPU loader smoke: pass, dtype=bf16, batch=2
```

The BF16 smoke conversion is only an infrastructure test. Production BF16 weights should be exported from the original checkpoint with `tools/export_stream1.py --dtype bf16 --format piece-transformer` or via the HPC launcher with `BEAM_STREAM1_EXPORT_DTYPE=bf16`.

Verification:

```text
docker gpu-dev-cutlass-nsight:2026-05-24
cmake --build build-cls-attn-local -j2
ctest --test-dir build-cls-attn-local --output-on-failure
100% tests passed, 0 tests failed out of 13

bash -n hpc/bench_8xa100_megaminx_transformer.sh
python3 -m py_compile tools/stream1_transformer_backends.py tools/stream1_transformer_torch_benchmark.py tools/export_stream1_transformer.py
```

LibTorch BF16 loader smoke:

```text
stream1_transformer_libtorch_backend=1 mode=eager device=cpu dtype=bf16 seq_len=51 d_model=256 nhead=8 layers=4 output_dim=24
stream1_transformer_libtorch_micro mode=eager batch=2 concurrency=1 ... candidates_per_sec=1621.75
```
