# A100x8 autotune bootstrap shard-capacity correction

Date: 2026-08-02

## Live evidence

Clean-cluster job 33343 reached model execution on all eight A100 GPUs but failed at 30M beam in `stream3_remote_recv_collect`, flag/code 3002. The seed profile had `shard_capacity_candidates=586752`; observed occupancy reached about 581k while another remote batch arrived and the active spill capacity was zero.

## Fix

- Increase only the conservative bootstrap `shard_capacity_scale_ppm` from 1,250,000 to 2,500,000.
- At 30M beam on 8 ranks and 8 shards this reserves about 1.17M candidates per shard.
- Preserve shard-capacity scale as a successive-halving tuning dimension.
- Include at most 256 KiB from each nested solver log in failure classification.

## Verification

- Focused controller/probe suite: 21 passed.
- Full portable suite: 192 passed, 3 skipped.
- `git diff --check`: passed.

Runtime acceptance requires a rebuilt SM80 archive and another clean-cluster run.