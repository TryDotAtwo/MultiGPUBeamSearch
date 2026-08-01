# MEPhI Megaminx Transformer Benchmark Export Fix

Date: 2026-07-06

Issue from cluster job 31729:

- Native CUTLASS checked the requested `B_MICRO x concurrency` grid, but `16384x8` was skipped by the benchmark memory guard.
- PyTorch/LibTorch only received batch `512` because Slurm `--export` treats commas as variable separators.
- `best_megaminx_transformer_stream1.env` was not written because AWK used `log` as a variable name.

Fix:

- Add `ISOLATED_BATCH_LIST` as the Slurm-friendly space-separated input and derive `ISOLATED_BATCHES` CSV inside the script.
- Parse both `candidates_per_sec=` and PyTorch `candidates_per_s=` output keys.
- Rename AWK `log` variable to `source_log` in both best-env writers.

Verification commands:

```text
python tests\test_stream1_transformer_backends.py
python tests\test_stream1_transformer_parity.py
bash -n hpc/mephi_8xa100_common.sh hpc/bench_8xa100_megaminx_transformer.sh hpc/start_8xa100_libtorch_megaminx.sh
```