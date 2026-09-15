# Native Hopper E4M3 implementation boundary

## Offline artifact checkpoint

`tools/hopper_fp8_export.py` now emits an explicit experimental bundle. Real
Cube4 export:57 tensors,3 full-token QKV weights in E4M3FN, all others in FP16,
one physical representation per original file. Bundle manifest SHA256
`b38ee1490a870dc16254cbfc15f3d3292e0e0f08d48ab9c836074b9dbd57a3a1`.
CPU tests5/5 pass, plus100000 encoded values byte-exact against PyTorch FP8.
This is offline encoding evidence, not native loader/inference validation.
Source manifest SHA256 `03f55e5b617874cbbfaace5796f7e348588eb2c0b5ef079e45c06b05d84d0cb0`.
Do not equate manifest hashes across exports with different source metadata.

Old51111383 again unavailable; cancelled pending start. Replacement51120383
on machine51172 is H200 SXM,300GBdisk,$5.789/hr. SSH key setup pending.

User ordering: native performance first; distillation and quality recovery later.
Keep Cube4 architecture, ReLU, token/move order and all four blocks unchanged.

First boundary to implement and measure:

```
FP16 persistent residual/token input
  -> FP32 LayerNorm reduction and affine
  -> direct E4M3 output using explicit artifact scale
  -> SM90 TMA/WGMMA QKV with offline E4M3 weights
  -> FP16 QKV output including original bias
```

Lifetimes: weights and scales immutable across replay; LN registers die after
one-byte-per-element packed output; FP16 normalized temporary is absent;
QKV remains alive through attention. No per-forward weight conversion/cache.
Fixed activation scales are a declared experimental inference contract, not
an implicit calibration claim. Compare native FP16 LN+QKV at identical rows,
shape and output lifetime. Quantized CPU oracle proves implementation, not
teacher quality. Report both independently.

Next only after first physical gate: extend FF1 epilogue to ReLU/direct packed
hidden, native FP8 FF2, then full model and caller integration. Do not promote
the boundary measurement as whole-Stream1 speedup. Existing best FP16 native
profile and experimental TE measurements remain controls.

Prepared `tests/hopper_native_fp8_tests.cu` with independent CPU LN and dense
dequantized-product checks at M1/31/128/131. Added the experimental header
`cuda/stream1_hopper_native_fp8.cuh`; it is not connected to production or the
default build. RED compilation failed on the missing implementation header.
GREEN not established: Windows CUDA12.5/CUTLASS3.8 and newer local CUTLASS
headers fail on SM100 constexpr-dim3 declarations; logs retained. Local compiler
also needs allow-unsupported-compiler for installed MSVC14.44. These checks
are not target CUDA12.8/CUTLASS3.9.2 evidence. No device run or speed claim.

Vast instance51111383 remained Scheduling across repeated checks; direct SSH
refused connection. Instance and extra-debug logs only showed old container
events, not a new launch failure. Do not confuse the stale card's lower
`running` text with the current top-level Scheduling state. Stop the pending
start before handing off so it cannot later begin billing an idle GPU.
