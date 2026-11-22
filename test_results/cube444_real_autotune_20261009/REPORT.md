# Real Cube444 FP16 Transformer + MLP on 2xRTX3060

Trained s3 PieceTransformer and mlp_x16 PairQMLP, coefficients 0.6/0.4. FP16 weights/activations, FP32 reductions/accumulation. LibTorch backbones and CUTLASS weighted final epilogues; serial heads share parent inputs. No Hamming predictor.

| Exact global parents | Matched Stream1 seconds | Complete depth seconds | Throughput loss |
| ---: | ---: | ---: | ---: |
| 1,048,576 | 34.0035546875 | 34.4901 | 1.410681% |
| 8,388,608 | 272.16246875 | 273.469 | 0.477762% |

Both rows use the same legal full frontier, trained weights, GPU UUID cohort and inference microbatch8192 within each comparison. Complete depth has one discarded warmup plus five measured depths; inference seven repeats. This is a fixed-frontier benchmark from index1000 followed by80 legal random moves, not a solved index1000 search. Host power limits135W per card. Different eight-GPU host results must not be treated as a matched-power strong-scaling proof.

Fixes: clear only unused CUDA allocator cache before candidate warmup; use each measured batch's own reserve when admitting the exact requested width; do not charge native weights/scratch twice when the runner already subtracts an explicit LibTorch reserve. Maximum-frontier bootstrap starts from the minimum measured inference footprint, then recalibrates/admit inference on the exact resulting width. Five-stream architecture and Stream4 CUB sort/reduce remain unchanged.

Verification: real all-rank FP32-readout oracle gates <=1 integer key unit, no numerical errors; independent CPU replay of16 frontier samples/rank, all-row color counts and padding, input hashes; three-move trained-blend solve and path replay; runtime accounting tests; final Linux Python suite314 passed,5 skipped. Public CubeTransformer model adapter is not covered: these native internal Cube444 family tests validate the native runner and admission/component machinery, not arbitrary model integration.

Source component2ee4d4e1602b91502406f2c2f6333df4c70b3d68; nativec1917b6c99c0048bd1f309915f7e3cac677be3dd. Raw evidence archive and part manifests are adjacent. Binary archive: immutable prepared-runtime commitbcd6e160ce32ba0d86d1eda80423a682469aad56 under test_results/cube444_prepared_runtime_20261009, verified by commitd0f43e2b62cb21910a7752cd1563809e009339ca. Large public weights, deterministic frontier payloads and derived histories are excluded; hashes, byte inventory and regeneration scripts are retained in the evidence manifest.

Eight-GPU real-neural acceptance and largest-frontier full-depth verification remain in progress. This branch is evidence/development, not a default-branch release.
