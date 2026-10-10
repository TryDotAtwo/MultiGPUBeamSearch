# Real trained Cube444 neural blend: progress

Transformer s3 and MLP mlp_x16, FP16 weights/activations, coefficients 0.6/0.4. LibTorch backbones and CUTLASS weighted output epilogue; serial heads. This is not Hamming inference.

| GPUs | Global frontier | Matched Stream1 seconds | Full depth seconds | Throughput loss |
|---|---:|---:|---:|---:|
|2 RTX3060|1,048,576|34.00355|34.4901|1.4107%|
|2 RTX3060|8,388,608|272.16247|273.469|0.4778%|
|8 RTX3060|1,048,576|8.18268|9.62429|14.9789%|
|8 RTX3060|8,388,608|67.17878|69.667|3.5716%|

Two-GPU evidence972d523b and228-file SHA verification retained in adjacent cube444_real_autotune_20261009 directory; owned pair55065930 destroyed and absence verified. Eight-GPU numeric readout and exact frontier-count gates pass for these two beams. 32M, timeline diagnostics, candidate-isolation A/B and maximum neural frontier remain pending; eight-GPU final archive not yet complete. Pair135W/card versus eight170W/card and different rank-seeded input corpora: this table is not matched strong-scaling evidence.

Fixtures are legal80-move random walks from index1000, reset between benchmark depths. No complete index1000 solution is claimed. Public CubeTransformer Python adapter and128GPU execution are not validated by these internal native benchmarks.
