# Initial 3090 diagnostics

These are the preserved results from the retired initial 2xRTX3090 lease.
The state-120 build passed 55 CTests, skipped one and failed the Cube4-specific
production-score fixture, which requires state size 96. Python passed 504 tests
and skipped 25. This is diagnostic evidence, not final acceptance.
The matching state-96 baseline and current runtime were subsequently validated
on 2xRTX3060; see STAGE2.md and STAGE3_HOST.md. The archive manifest covers the
original downloaded files, before this explanatory note.
