# Per Stream Benchmark 2026-05-22

- puzzle_id=991
- gpu_total_bytes=8589410304
- gpu_free_before_bytes=7463763968
- generator_path=FullBeamNice/generators/p900.json
- puzzle_info_path=data/puzzle_info.json
- test_csv_path=data/test.csv
- weight_dir=/workspace/test_results/kaggle_libtorch_transformer_benchmark_v1_2026-07-04/stream1_transformer_weights_fp16
- stream1_model_hidden1=1536
- stream1_model_hidden2=512
- stream1_model_residual_count=2
- stream1_model_output_dim=24
- stream1_model_weight_bytes=6543396
- cuda_architectures=75,86
- stream1_gemm=TensorOp_Sm75_common_for_T4_and_RTX3070
- stream1_backend=piece_transformer

## Stream1 Piece Transformer

- graph_bench=0
- filter_b_micro=512
- filter_concurrency=2

| b_micro | concurrency | rows_per_launch_group | ms_per_launch_group | parents_per_sec | candidates_per_sec | scratch_bytes |
|---:|---:|---:|---:|---:|---:|---:|
|512|2|1024|25.6715|39888.6|957325.9|327729152|

## Status

- status=pass
