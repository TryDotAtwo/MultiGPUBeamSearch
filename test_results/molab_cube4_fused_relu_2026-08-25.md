# Molab Cube4 fused ReLU validation — 2026-08-25

## Scope

- Target: RTX PRO 6000 Blackwell Server Edition (SM120), CUDA 13.0.
- Workload: Cube4 piece Transformer, ReLU, output_dim=24, beam `2**25`, depth limit 9.
- Correctness oracle: current ReLU path (`depth_done=8`, threshold 10294, frontier 33554432) plus transformer CUDA reference test.
- Performance oracle: `depth_done=8` wall time; historical SiLU result is not a correctness-valid baseline.

## Root cause

The SiLU FF1 path used a CUTLASS broadcast epilogue that fused GEMM, bias, and activation. The ReLU FF1 path used the same GEMM+bias kernel followed by a separate full-tensor ReLU kernel. This extra read/write pass occurs in every Transformer FFN layer.

## TDD evidence

- RED in Molab: `tests/test_stream1_transformer_relu_fusion.py` failed because the ReLU dispatcher contained no fused ReLU entry point.
- GREEN in Molab: source contract passed (`1 passed`).
- Native CUDA compilation: passed for `sm_120a`; shared library linked with
  `DT_SYMBOLIC`/`-Bsymbolic`, so the already loaded historical SiLU library did
  not interpose the new implementation.
- The standalone CUDA reference binary was built and launched, but its optional
  reference fixture was not present in the recovered sandbox, so it reported
  `skip missing_reference_fixture`.

## In-process validation

- Short smoke: ReLU, beam `2**16`, `rc=0`, depth 3 in 0.108245 s.
- Full run: ReLU, beam `2**25`, `rc=0`, no OOM/CUDA error.
- Saturated depths: 83.1811 s, 83.3797 s, 83.4427 s for depths 6, 7, 8.
- Depth 8: frontier 33,554,432; Stream3 jobs 391; threshold 10,294.
- Correctness gate matched the previous unfused correct-ReLU run exactly on
  activation, frontier, Stream3 jobs, and depth-8 threshold.
- Performance improved from 103.374 s to 83.4427 s at depth 8: 19.28% lower
  wall time (1.239x throughput).

Molab log: `/tmp/cube4_relu_fused_c4a7ba36/build/cube4_fused_relu_p0_b2p25_depth8.log`.
