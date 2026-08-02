# Zero-touch cluster profile provisioning

The production SLURM job now performs hardware-only preflight, exact profile lookup, automatic depth-8 tuning for unknown tuples, measured-only atomic cache installation, and final profile selection before torchrun.

Safety properties:
- no approximate or cross-hardware fallback;
- signed archive registry remains immutable;
- cached profile requires matching GPU family, VRAM, SM, world size, driver, solver commit, and model digest;
- a hardware lock serializes first-time tuning;
- incomplete tuning aborts before solve;
- final materialization remains constrained to the 32K-128K search range.

Verification commands and results are appended after the full suite.

Verification:
- `python -m pytest -q`: 199 passed, 3 skipped.
- `python -m compileall -q portable/megaminx_cluster`: passed.
- `git diff --check`: passed.
- No cluster command was executed; live A100x8 validation remains user-operated.
