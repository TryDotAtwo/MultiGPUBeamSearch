# Portable Cube4 readiness, 2026-09-23

Evidence level: local bundle check plus mock GPU/compiler/runner. No physical GPU compilation, exact score comparison or throughput measurement.

- `py -3 test_results/check_cube4_bundle.py`: 2 tests passed. The archived Cube4 model (all 60 required non-empty tensor/metadata files) and generator/test bundle for puzzle 1000 passed; reversed output order was rejected. The repository default `data/puzzle_info.json` has 120 state elements and must not be used with the Cube4 state96 model.
- `test_results/check_cube4_portable.sh`: mock test passed. It exercised SM103 detection, compiler architecture listing, state96 CMake arguments, isolated Stream1, explicit 2-rank 256M beam alignment, absent-profile rejection and host-RAM overcommit rejection.
- `test_results/check_b300_build.sh`: mock test passed after pinning `BEAM_STATE_LOGICAL_BYTES=96` in the B300 build command.
- The saved H200 weight hash list was checked against the existing local T4-exported FP16 directory: 57 entries checked, zero mismatches. This verifies local identity only; transfer and remote checksum remain pending.

Next physical gate on any available supported card: inspect offer and spend, use a compatible CUDA image, transfer/verify assets, `probe`, `build`, CUDA tests and exact score parity, isolated Stream1 sweep, then a bounded depth-8 beam profile with static GPU memory plan and measured disk/headroom. Do not interpret a mock run as a solved puzzle.
