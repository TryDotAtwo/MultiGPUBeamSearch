# MEPhI 8xA100 Megaminx Transformer Benchmark Launcher

Date: 2026-07-06

Scope:

- Add a cluster benchmark launcher before running a large 900M Megaminx transformer beam search.
- Compare explicit Stream1 backends and production full-loop parameters on 8xA100 40GB.
- Keep all compute under SLURM `sbatch -p kaf12`.

Files:

- `hpc/bench_8xa100_megaminx_transformer.sh`
- `hpc/start_8xa100_libtorch_megaminx.sh`
- `hpc/README_8xa100.md`

Default benchmark:

```text
isolated Stream1: pytorch:eager, libtorch:eager, libtorch:cuda_graph, native_cutlass:graph
full smoke: 64M beam, depth 12, native_cuda_graph and libtorch_eager
target sweep: 900M beam, depth 8, native_cuda_graph and libtorch_eager, B_MICRO=8192 12288, concurrency=8
```

Outputs:

```text
logs/tuning_<job_id>/megaminx_transformer_bench_<job_id>.tsv
logs/tuning_<job_id>/megaminx_transformer_stream1_isolated_<job_id>.tsv
logs/best_megaminx_transformer.env
```

The production launcher now accepts `MEGAMINX_STREAM1_BACKEND=libtorch_eager` or `MEGAMINX_STREAM1_BACKEND=native_cuda_graph`, so `source logs/best_megaminx_transformer.env` can feed the later solve job.