# Native runtime SONAME closure fix

- Date: 2026-08-02
- Cluster evidence: job 33303 reached the bundled launcher, then every rank exited 127 because `libnccl.so.2` was absent.
- Root cause: closure collection resolved the SONAME symlink and copied only the fully-versioned target basename.
- RED: a synthetic `libnccl.so.2 -> libnccl.so.2.28.3` dependency test failed because the SONAME file was missing.
- GREEN: collector preserves the requested basename; 13 closure/policy tests passed.
- Defense in depth: release workflow runs `ldd` against the staged `lib/` directory and rejects any `not found`; 28 closure/policy/workflow tests passed.
- Required live gate: GitHub native matrix plus a fresh 8xA100 SLURM run.
