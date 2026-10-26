# Verified hardware profiles

The library first checks its local cache, then this source-pinned GitHub registry, then runs automatic calibration. A registry record is a measured proposal: it does not bypass current GPU memory admission, numerical validation, or a short all-GPU timing cohort. A successful validation creates a normal local cache receipt. An unavailable or incompatible record falls back to calibration.

Matching is exact for graph, model artifacts (including blend weights), native source digest, shape, backend, precision, PyTorch/CUDA version, GPU model/capability/memory/SM count, GPU count and beam width. A 10M profile is not selected for a 100M frontier. UUIDs are deliberately not portable. GPU topology and current contention can differ, so a transferred profile is not a promise of globally optimal speed on every host.

`index.json` references SHA-256-addressed `records/<sha256>.json`. Records carry measured inference batch, a whitelisted pipeline environment, compatibility and evidence provenance. Remote settings cannot change the graph, models, paths, requested beam, or correctness policy. Full-frontier verification is separate from short component validation.

Only completed, source-bound GPU results are published. An empty index means no compatible hardware profile has been accepted for this source revision yet; it does not disable automatic tuning. Historical T4 or RTX3060 results from a different native source are not silently relabeled compatible.
