# Megaminx autotune bundled runtime path fix — 2026-08-02

## Cluster evidence

Clean SM80 archive checksum passed. Job 33318 failed in 5 seconds because every rank returned rc=127 loading `libnccl.so.2`. On the login node, the bundled `lib/libnccl.so.2` was a valid ELF file and `LD_LIBRARY_PATH=<archive>/lib ldd bin/production_runner` resolved it. The packaged autotune entrypoint, unlike the normal solve entrypoint, did not export the archive library path.

## Fix

`autotune_job.sh` now exports archive-root `PYTHONPATH` and `<archive>/lib` in `LD_LIBRARY_PATH` before preflight/controller launch. Contract assertions prevent this from regressing.

## Verification

`py -m pytest tests/portable/test_megaminx_autotune_job_sh.py tests/portable/test_megaminx_autotune_e2e.py tests/portable/test_megaminx_runtime_closure.py -q`

Result: 10 passed, 1 skipped.
