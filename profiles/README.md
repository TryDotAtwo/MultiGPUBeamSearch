# Verified hardware profiles

The library first checks its local cache, then this source-pinned GitHub registry, then runs automatic calibration. A registry record is a measured proposal: it does not bypass current GPU memory admission, numerical validation, or a short all-GPU timing cohort. A successful validation creates a normal local cache receipt. An unavailable or incompatible record falls back to calibration.

Matching is exact for graph, model artifacts (including blend weights), native source digest, shape, backend, precision, PyTorch/CUDA version, GPU model/capability/memory/SM count, GPU count and beam width. A 10M profile is not selected for a 100M frontier. UUIDs are deliberately not portable. GPU topology and current contention can differ, so a transferred profile is not a promise of globally optimal speed on every host.

`index.json` references SHA-256-addressed `records/<sha256>.json`. Records carry measured inference batch, a whitelisted pipeline environment, compatibility and evidence provenance. Remote settings cannot change the graph, models, paths, requested beam, or correctness policy. Full-frontier verification is separate from short component validation.

Only completed, source-bound GPU results are published. An empty index means no compatible hardware profile has been accepted for this source revision yet; it does not disable automatic tuning. Historical T4 or RTX3060 results from a different native source are not silently relabeled compatible.


## Accepted records

| Workload | Hardware | Global beam | Inference batch | Full step median |
|---|---|---:|---:|---:|
| Cube555 Transformer 0.8 + MLP 0.2, FP16 / FP32 accumulation | 2 x RTX3060 12GB, CUTLASS, PyTorch 2.10.0+cu128 | 825,344 | 8,192 parents | 53.1106 s |

The [record](records/2c24c6c1711cc27f8d8c54e4ae87282f6b7589445a84dc907d8467eda6247c20.json)
contains the exact compatibility key and raw selected-cohort timings. Fresh-cache validation on the measured
host took 19.88 s for the original export and 20.34 s for independently exported notebook models. Two short CayleyPy searches with persistent worker reuse completed in 27.46 s total in the latter check.
The full width retained 99.96% of the half-width reference throughput. These measurements
apply to this workload and machine; the width is the largest suitable width found in the bounded search,
not a universal maximum. No T4 profile is claimed for this native source revision yet.

Cube555 profile keys retain executable TorchScript code, tensors and constants while excluding source debug maps, serialization IDs and the order of independent literal Final declarations. Loading and local artifact integrity still use exact byte hashes. A different executable payload does not match the shared profile.
