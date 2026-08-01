# Stream1 transformer SM75 GEMM policy build gate — 2026-07-22

## Decision

Expose opt-in SM75 CUTLASS policy candidates for QKV, FF1, attention-output, and FF2 so the private 2xT4 benchmark can tune GEMM tiles instead of inheriting SM86 choices. Keep existing defaults unchanged and reject unsupported combinations explicitly.

## Architecture contract

- SM75 QKV: baseline plus `m64n128`, `m128n128`, and `m256n128`; QKV swizzles remain explicit candidates.
- SM75 FF1: baseline plus `m64n128`, `m128n128w64n32`, and `m128n128`.
- SM75 FF1 is compiled only with CUTLASS stages 2 and identity swizzle 1. The three-stage fused-broadcast kernel is not supported by the SM75 CUTLASS specialization and now fails closed before launch.
- SM75 attention-output and FF2: separate residual policies expose `m64n64` and `m128n128`; fused exact residual+bias-round candidates use SM75 tensor-op instruction shape and stages 2.
- Existing SM80/SM86 stages-3 and swizzle paths are unchanged.

## Verification

- Expected-red CPU contract test failed before the architecture-stage helper existed.
- Updated policy contract test passed.
- SM75 architecture build passed for `stream_benchmark`.
- Local SM86 Docker CTest passed 18/18.
- The local RTX 3070 was idle before the regression run (`0 MiB`, `5%` sampled utilization).
- Selected SM86 production path remained byte-exact: full score dump SHA-256 `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.
- Local regression row: `8.1789 ms`, `1,502,410.1 candidates/s`; this run is a correctness regression gate, not a new speed comparison.
- Shared Docker GPU queue job: `8cc3232659ba`.
- No `CUDA_LAUNCH_BLOCKING`, fallback, or MLP production change was introduced.