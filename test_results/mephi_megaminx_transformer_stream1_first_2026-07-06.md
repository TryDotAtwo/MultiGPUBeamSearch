# MEPhI Megaminx Transformer Stream1-First Benchmark Update

Date: 2026-07-06

Scope:

- Make `hpc/bench_8xa100_megaminx_transformer.sh` run Stream1-only benchmarking by default.
- Select production backend plus `BEAM_B_MICRO` / `BEAM_STREAM1_CONCURRENCY` from isolated Stream1 throughput.
- Keep the known 24-output 900M pipeline parameters unchanged for the selected 900M pass: 32 shards, Stream3 ring 8, Stream4 batch 262144, Stream4 trigger 1048576, final chunk 98304, final exchange scale 2x, shard capacity scale 1x.
- Add opt-in `RUN_SELECTED_900M_AFTER_STREAM1=1` for a single selected 900M depth-limited target pass after Stream1 microbench.
- Expand native Stream1 transformer microbench sweep so production-sized `B_MICRO=8192/12288/16384` and concurrency `8` can be measured instead of silently skipping those points.

Verification:

```text
python tests\test_stream1_transformer_backends.py
.....
Ran 5 tests in 0.001s
OK

python tests\test_stream1_transformer_parity.py
.......
Ran 7 tests in 0.000s
OK

git diff --check
warning: in the working copy of 'hpc/README_8xa100.md', CRLF will be replaced by LF the next time Git touches it

docker run --rm -v D:\100XH100\.worktrees\stream1-piece-transformer:/work -w /work cmz-native-dev:2026-05-26 bash -lc "bash -n hpc/mephi_8xa100_common.sh hpc/bench_8xa100_megaminx_transformer.sh hpc/start_8xa100_libtorch_megaminx.sh"
exit=0
```

Cluster usage:

```bash
sbatch -p kaf12 bench_8xa100_megaminx_transformer.sh
```

For immediate Stream1 benchmark plus one selected 900M target pass:

```bash
sbatch -p kaf12 \
  --export=ALL,RUN_SELECTED_900M_AFTER_STREAM1=1,TARGET_DEPTH_LIMIT=8 \
  bench_8xa100_megaminx_transformer.sh
```