# Portable Cube555 blend and GitHub profiles — 2026-10-26

The portable notebook uses CayleyPy `graph.beam_search`, automatic visible GPU count/backend selection,
and `NativeSession.activate()` for repeated calls. Source, weights and inputs are pinned GitHub artifacts.
A private Python environment handles PEP 668 without modifying system packages. Source builds require
an NVIDIA driver, CUDA-enabled PyTorch, CUDA Toolkit and a C++ compiler on one allocated GPU node.

## Verified execution

Hardware: 2 x RTX3060 12GB; PyTorch 2.10.0+cu128; CUTLASS; actual pretrained Cube555
Transformer 0.8 + MLP 0.2, FP16 execution with FP32 accumulation.

| Check | Result |
|---|---|
| Native arithmetic | `failures=0` |
| Global frontier 825,344, inference batch 8,192 | Full-step median 53.1106 s |
| Half-width reference 412,672 | Full-step median 26.5443 s |
| Normalized throughput retained at full width | 99.96% |
| Initial bounded maximum/profile verification | 759.16 s total |
| Ready-profile validation after independent notebook export | 20.34 s |
| Two short CayleyPy calls, including ready-profile validation | 27.46 s total; paths `[6,1]` and `[10,6,1]` independently replayed |
| Persistent reuse | Same worker PIDs on the second call; workers absent after session close |
| Portable notebook code cells | All six code cells completed on Linux, default beam 1,048,576, automatic 2 GPUs, four-depth test budget |

Each full-width/reference cohort uses six saturated natural-growth depths, including warmup, and
five measured repetitions. The width is the largest suitable width found by the bounded search with
the frozen inference batch, not an exhaustive universal maximum. The failed pre-fix run is not a
performance baseline. Isolated inference timings are not a matched full-step slowdown measurement.

The notebook test injected the independently verified native binary instead of recompiling it on the
paid GPU host. Native compilation separately passed in Kaggle CPU builder v5. This is not a claim of
one unmodified cold source-build execution on 2xT4. The four-depth real-puzzle run correctly returned
`not_found_within_budget`; no complete solution for that puzzle is claimed.

## Profile reuse and export identity

Lookup order is local cache, exact-compatible GitHub proposal, then ordinary calibration. A transferred
batch receives fresh numerical/timing checks on every GPU and downstream settings receive fresh exact
memory/component checks. These short checks are not reported as current-host full-step verification.
A different beam does not reuse this record. The [published record](../../profiles/records/2c24c6c1711cc27f8d8c54e4ae87282f6b7589445a84dc907d8467eda6247c20.json)
contains the compatibility key and raw selected-cohort timings.

The notebook test found that independent TorchScript exports differ in debug source maps, serialization
IDs and literal Final declaration order. Executable code, tensor storage and other constants matched.
A separate `cube555-ts-v1` profile identity normalizes only those non-execution differences. Original
byte hashes remain mandatory for loading and local artifact integrity. Tests reject changed tensor
storage, methods and reordered duplicate/nonliteral declarations. Fresh-cache GPU reuse was then
verified using the notebook's independently exported models.

The executed notebook SHA256 was `03524cc0755e57fecf83d816ee463ef8c67129ded24d55551c44bfc21dcf6647`.
The final notebook SHA256 is `a9eb677ea7b6e139f59ddcba103a95689a93618256cb726bb2de7d2438e31b00`.
The only difference is the source pin advancing from `7aa4412` to separately validated `869e146`.
The initial raw-artifact-equality postcheck failure remains in the archive; it is resolved by the
execution-content comparison and subsequent GPU profile-reuse test, not by rerunning unchanged cells.

Focused CPU suites passed: shared profile/cache/fallback and pipeline integration; 12 profile identity
and shared-profile tests in the final focused run. Packaging includes hash-bound profile records in
the installed native-source bundle. Windows pyreadline emitted a shutdown warning after successful
pytest completion; it did not change test outcomes.

## Stream3 reservation and limits

Native candidate `ad344234322fcebf9519a9a24dcc93ef86d9a48c` reserves local B plus receive R before
launching the local graph. The existing per-round threshold collective is moved, not duplicated.
Ready-queue ownership cannot hand the only receive-capable bank to Stream4. No receive-side capacity
wait is added, and explicit requested beams are not reduced.

The guarantee applies to admitted rounds. Arbitrary score ties/hash skew are not bounded by average
beam/shard sizing; admission fails before unsafe launch if filtering cannot restore credits. No shard
top-k or tie dropping was added. No 128-GPU acceptance or new T4-ready profile is claimed. T4 uses the
normal calibration path when no exact compatible registry entry exists.

## Evidence and cleanup

- Native prebuild SHA256: `1e6073e1d6f179afd2bcde143a57d9c0f7ae764996b12f455a3648c92b28df3b`.
- Full GPU run Python source: `de53e6e`; portable shared-profile implementation: `869e146`.
- Evidence archive: 318 files, each manifest entry verified; SHA256
  `5fe93ed8ac9bffd676d5c5732c038c4ec74aead454e283227711d007a303d8ad`.
- Only the task-owned Vast instance was deleted; its absence was confirmed. The $5 authorization
  and six-hour protective bound were retained. The local watchdog was stopped after deletion.
