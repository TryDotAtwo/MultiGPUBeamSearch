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

| b_micro | concurrency | rows_per_launch_group | ms_per_launch_group | parents_per_sec | candidates_per_sec | scratch_bytes |
|---:|---:|---:|---:|---:|---:|---:|
|512|1|512|32.1490|15925.8|382220.4|120348672|
|512|2|1024|63.7914|16052.3|385255.4|240697344|
|512|4|2048|129.8843|15767.9|378429.0|481394688|
|1024|1|1024|62.1694|16471.1|395306.8|240697344|
|1024|2|2048|130.3844|15707.4|376977.7|481394688|
|1024|4|4096|252.6517|16212.0|389089.0|962789376|
|2048|1|2048|120.1118|17050.8|409218.8|481394688|
|2048|2|4096|248.7781|16464.5|395147.4|962789376|
|2048|4|8192|524.5324|15617.7|374825.3|1925578752|
|4096|1|4096|286.9448|14274.5|342588.6|962789376|
|4096|2|8192|578.1363|14169.7|340072.1|1925578752|
|4096|4|16384|1537.1440|10658.7|255809.5|3851157504|
|8192|1|8192|775.0871|10569.1|253659.2|1925578752|
|8192|2|16384|1583.3726|10347.5|248340.8|3851157504|
|8192|4|32768|skip|skip|skip|7702315008: estimated allocation exceeds available GPU memory free_bytes=7449083904 io_bytes=3538944|

## Status

- status=pass
