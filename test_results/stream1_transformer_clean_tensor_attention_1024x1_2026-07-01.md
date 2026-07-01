# Per Stream Benchmark 2026-05-22

- puzzle_id=0
- gpu_total_bytes=8589410304
- gpu_free_before_bytes=7463763968
- generator_path=FullBeamNice/generators/p900.json
- puzzle_info_path=data/puzzle_info.json
- test_csv_path=data/test.csv
- weight_dir=test_results/stream1_transformer_reference/weights_fp16
- stream1_model_hidden1=1536
- stream1_model_hidden2=512
- stream1_model_residual_count=2
- stream1_model_output_dim=24
- stream1_model_weight_bytes=6543396
- cuda_architectures=75,86
- stream1_gemm=TensorOp_Sm75_common_for_T4_and_RTX3070
- stream1_backend=piece_transformer

## Stream1 Piece Transformer

- filter_b_micro=1024
- filter_concurrency=1

| b_micro | concurrency | rows_per_launch_group | ms_per_launch_group | parents_per_sec | candidates_per_sec | scratch_bytes |
|---:|---:|---:|---:|---:|---:|---:|
|1024|1|1024|41.4473|24706.1|592946.4|327729152|

## Status

- status=pass
