# Public ready-to-run Megaminx puzzle 10 example — 2026-07-30

## Public notebook

- URL: https://www.kaggle.com/code/trydotatwo/cayleypy-2xt4-megaminx-puzzle-10-example
- Kernel version: 3
- Terminal status: `KernelWorkerStatus.COMPLETE`
- Competition source: `cayley-py-megaminx`
- Model source: `arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1`
- Solver commit: `f312265d63cfc563b90c79a2fd70d1307d28d4bb`

## Exact run

- GPUs: `Tesla T4`, `Tesla T4`; world size 2.
- Puzzle: 10 only.
- Mode: first, reflection off, max depth 40.
- Requested beam: 1024; aligned effective beam: 8192; measured p16 output1 profile.
- Model: batchnorm-folded, fp16, output_dim 1.
- Checkpoint SHA-256: `7f5071e6155c4eb7718539bf990a4234404f06c2979307d8e3cdcd37a539b759`.
- Solve result: success, found depth 10, length 10.
- Path: `-B.-BL.-U.-BL.BR.-R.FL.-DL.-BL.F`.
- Rank-0 reported search time: 1.83026 seconds.
- Aggregated solve stage: 8.528350792 seconds; notebook wall time: 94.923515929 seconds including clone/export/build.
- Independent local replay against standard data: valid.

## Publication

- Public ingest POST `/v1/results`: HTTP 202, `publish_status.state=published`.
- Client submission id: `019fb409-1cf0-733a-b1fb-2293cf6c7be7`.
- GitHub staging file: `results/v1/cayley-py-megaminx/cayleypy-120-24-cfa02f58c6b66053/10/2026-07-30/019fb409-1fda-7934-8028-cf71acfbc503.json`.
- Verified staging branch head: `77567bee187471095830f3405d8b8c102ed91775`.

## Fixes from live iterations

- v1 proved solve/model/data but exposed the wrong endpoint root.
- v2 used `/v1/results` but Cloudflare returned 403 to the anonymous urllib client.
- v3 adds explicit `CayleyPy-Kaggle-Publisher/1.0`, keeps DNS/redirect safety, and publishes successfully.
- Repository checkout/scratch moved to `/tmp`; `/kaggle/working` now retains only final artifacts.
- Beam-search and CUDA architecture were not changed.