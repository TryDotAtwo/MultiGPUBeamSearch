# RTX 3070 LayerNorm row-scheduling gate (2026-07-22)

Target: the fused bias + model-dtype round + LayerNorm + copy kernel, which accounts for 11.7% of the current Nsight Systems kernel time.

| Candidate | Correctness | Baseline median | Candidate median | Relative result | Wins |
|---|---|---:|---:|---:|---:|
| one warp per row, four rows per CTA | rejected: stable different SHA `ebea7e...8663` | 8.7059 ms smoke | 8.2668 ms smoke | faster but invalid | n/a |
| two rows per CTA, original four warps per row | 40/40 exact | 8.47655 ms | 8.4806 ms | -0.05% | 8/20 |
| occupancy-sized persistent grid, loads inside loop | 40/40 exact | 8.50405 ms | 8.57335 ms | -0.81% | 7/20 |

Hoisting bias/gamma/beta into persistent-kernel registers recovered about 2% in a smoke run but produced the same invalid SHA as the one-warp implementation, so it was blocked immediately. Keeping the old expression/load structure restored byte exactness but lost the gain due to the extra loop barrier.

Production remains one row per 128-thread CTA. Exact `block2` and `persistent` policies remain opt-in candidates for independent hardware autotuning; the known-invalid one-warp policy was removed.

Reference SHA-256: `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.

Final production verification after removing the invalid policy: Docker CTest 18/18 passed; fresh production launch measured `8.5315 ms` and retained SHA-256 `a9495016409c4d43a4b592da613ceba6b32ea398ec353dd22b9fc019a8569d94`.
