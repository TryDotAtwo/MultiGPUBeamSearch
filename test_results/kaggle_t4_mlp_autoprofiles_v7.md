# Kaggle 2xT4 MLP autoprofiles v7 acceptance

- Date: 2026-07-29
- Private kernel: `trydotatwo/cayley-beam-2xt4-mlp-autoprofiles`, version 7, `COMPLETE`.
- Hardware: two `Tesla T4, 15360`; torchrun world size `2`.

## Accepted registry safety gate

The downloaded terminal artifacts in
`D:\100XH100\test_results\kaggle_t4_mlp_autoprofiles_v7` report:

- `attempt_count=0`, `missing_anchor_count=0`, and
  `sweep_status=no_missing_anchors`;
- header-only `autoprofile_attempts.csv` (no fresh runs);
- exact canonical source SHA256
  `3785713329f0a8dac13e5455f8e5acc58a760f21dbd447e544b08f3338948a50`;
- `selected_profiles.json` contains exactly 20 measured depth-8 records:
  output1 and output_move_count for every p16 through p25.

v6 is explicitly rejected as a safety-gate acceptance: its stale notebook
reran 14 old focused attempts (9 successful, 5 failed). v7 embeds and checks
the canonical v4/v5 registry, so it does not require that file to exist in the
GitHub-main clone. No v6 timing replaces canonical v4/v5 evidence.

Scope is generator/configuration and notebook packaging only. CUDA, C++, and
beam-search algorithms are unchanged.
