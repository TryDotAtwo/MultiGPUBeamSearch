# Verified hardware profiles

The library first checks its local cache, then this source-pinned GitHub registry, then runs automatic calibration. A registry record is a measured proposal: it does not bypass current GPU memory admission, numerical validation, or a short all-GPU timing cohort. A successful validation creates a normal local cache receipt. An unavailable or incompatible record falls back to calibration.

Matching is exact for graph, model artifacts (including blend weights), native source digest, shape, backend, precision, PyTorch/CUDA version, GPU model/capability/memory/SM count, GPU count and beam width. A 10M profile is not selected for a 100M frontier. UUIDs are deliberately not portable. GPU topology and current contention can differ, so a transferred profile is not a promise of globally optimal speed on every host.

`index.json` references SHA-256-addressed `records/<sha256>.json`. Records carry measured inference batch, a whitelisted pipeline environment, compatibility and evidence provenance. Remote settings cannot change the graph, models, paths, requested beam, or correctness policy. Full-frontier verification is separate from short component validation.

Only completed, source-bound GPU results are published. An empty index means no compatible hardware profile has been accepted for this source revision yet; it does not disable automatic tuning. Historical T4 or RTX3060 results from a different native source are not silently relabeled compatible.


## Accepted records

| Workload | Hardware | Global beam | Inference batch | Full step measurement |
|---|---|---:|---:|---:|
| Cube555 Transformer 0.8 + MLP 0.2, FP16 / FP32 accumulation | 2 x RTX3060 12GB, CUTLASS, PyTorch 2.10.0+cu128 | 825,344 | 8,192 parents | 53.1106 s (median, 5 runs) |

The [record](records/2c24c6c1711cc27f8d8c54e4ae87282f6b7589445a84dc907d8467eda6247c20.json)
contains the exact compatibility key and raw selected-cohort timings. Fresh-cache validation on the measured
host took 19.88 s for the original export and 20.34 s for independently exported notebook models. Two short CayleyPy searches with persistent worker reuse completed in 27.46 s total in the latter check.
The full width retained 99.96% of the half-width reference throughput. These measurements
apply to this workload and machine; the width is the largest suitable width found in the bounded search,
not a universal maximum.

Cube555 profile keys retain executable TorchScript code, tensors and constants while excluding source debug maps, serialization IDs and the order of independent literal Final declarations. Loading and local artifact integrity still use exact byte hashes. A different executable payload does not match the shared profile.

| Cube555 Transformer 0.8 + MLP 0.2, FP16 / FP32 accumulation | 2 x Tesla T4, LibTorch, PyTorch 2.11.0+cu128 | 1,048,576 | 256 parents | 101.642 s (one full step) |

That legacy T4 record is `21d3e9953857ba60c9ef8669267712f1c48ad612ff6cb3f31bb931fdbff01015`; its source key and PyTorch version differ from the grid below. It remains a separate exact-match entry.

## 2 x Tesla T4 profile grid: current pinned LibTorch source

All rows use the same Cube555 Transformer 0.8 + MLP 0.2 workload, FP16 execution, FP32 accumulation, LibTorch eager, PyTorch 2.10.0+cu128, and CUDA 12.8. Each row is one completed saturated full-frontier step per rank. “Stream1 estimate” multiplies the selected 8,192-parent inference median by the number of parents per rank; the slowdown is a derived comparison, not a second full-step benchmark.

| Global beam | Model inference batch | Outer `B_MICRO` | Full step (s) | Stream1 estimate (s) | Derived slowdown | Stream3 ring slots | Shards |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 65,536 | 256 | 256 | 7.4853 | 5.6757 | 31.88% | 2 | 8 |
| 131,072 | 512 | 512 | 15.0906 | 14.3137 | 5.43% | 2 | 8 |
| 262,144 | 512 | 512 | 31.9735 | 28.4235 | 12.49% | 8 | 4 |
| 524,288 | 256 | 256 | 60.3059 | 58.0102 | 3.96% | 2 | 4 |
| 1,048,576 | 256 | 256 | 122.0200 | 120.4569 | 1.30% | 2 | 4 |
| 2,097,152 | 256 | 256 | 246.0280 | 235.5007 | 4.47% | 2 | 4 |
| 4,194,304 | 256 | 512 | 494.8480 | 467.2405 | 5.91% | 4 | 4 |

The records are individually content-addressed in `index.json`. Their exact native source digest is `c00b842d2baedd7028f2c736af7a07c297b2cad7b652dea000ed0c625a59cad1`, reproduced from solver commit `804efdc28d84f1d547d5f0e7fe47b2c81ec3b928` using the same CRLF normalization performed by `BuildWithNativeSource` and the pinned `build.py` source-file allowlist. The prior 1,048,576 T4 row uses a different source digest and is not substituted or relabeled.

These are one-step timing profiles, not solutions and not proof that each configuration is a global optimum. The 65,536 row incurs visibly higher derived pipeline overhead; the table retains the measured config rather than claiming ideal scaling. Every reuse still requires fresh all-rank memory admission, correctness validation, and a short current-GPU timing check. Full evidence summary and limitations: [`evidence/cube555-t4-grid-20261031.md`](evidence/cube555-t4-grid-20261031.md).

