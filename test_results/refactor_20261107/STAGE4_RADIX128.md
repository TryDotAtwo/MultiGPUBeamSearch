# Experimental full-width CUB radix sort

This experiment preserves the existing comparison sort as the default. On the
measured 2x RTX 3060 fixture, the radix variant was slower and required more
planned device memory. It is not promoted to a production profile.

## Implementation

- `BEAM_EXPERIMENTAL_RADIX128=ON` selects `cub::DeviceRadixSort::SortPairs`
  for Stream 3 hash deduplication, Stream 4 shard deduplication and the final
  logical A/B union. The tuple decomposer orders `{hi, lo}`; all 128 bits remain
  part of identity. CUB 2.7 / CUDA 12.8 or newer is required for this opt-in.
- Sorting writes from buffer A to B, and the existing deterministic reduction
  writes from B back to A. The default comparison-sort path keeps its previous
  A-to-B reduction. Both paths use disjoint buffers and the same fixed arena;
  the arena queries the selected CUB operation for its actual scratch size.
- The ordinary Python build passes `BEAM_EXPERIMENTAL_RADIX128=OFF` explicitly,
  including when a CMake build directory is reused. Inference kernels, precision,
  original Stream 3 candidate IDs, survivor semantics and the public launcher
  are unchanged.

## Correctness evidence

The same source was built with the experimental option off and on. Both builds
passed Stream 3, Stream 4, logical-union, captured/bounded graph, static-memory,
stitched-pipeline and final single-rank partial/full-count tests on RTX 3060.
The standalone radix test also checked two physical GPUs, empty and partial
batches, original payloads, equal keys, both independent hash halves, extreme
keys and graph replay against a CPU stable-sort oracle. Stream 3 additionally
compares both threshold entry points with a full-hash CPU reduction, including
ties and a valid all-ones hash. The logical-union oracle checks cross-buffer
deduplication and histograms, including an all-ones hash.

CUDA Compute Sanitizer memcheck with full leak checking reported zero errors and
zero leaked bytes for the standalone radix, Stream 3 and captured logical-union
tests. The separate radix test compiled for SM 75, 80, 90 and 120 with CUDA
12.8; that is not a full-engine build or physical acceptance on those GPUs.
The Python build suite passed 25 tests, including the explicit default-off guard.

The first paired benchmark attempt failed before measurement because its test
harness used an older incomplete one-shot planner response. It was changed to
the existing all-rank planning session. The failure and subsequent runs remain
in the evidence directory.

## Full-step comparison

The final run used two RTX 3060 GPUs, a deterministic Hamming96 scalar MLP,
FP16, logical width 4096 and the same frozen native CUDA Graph profile. It
alternated control/radix ordering across five independent pairs. Each process
performed one warmup and five complete saturated steps; the per-pair statistic
is the median of the slower-rank step times. Model files, input and executables
were hashed before and after the series; the rank geometry and sampled GPU
telemetry were checked. Telemetry samples cannot exclude every transient event.

| Measurement | Control | Radix experiment |
| --- | ---: | ---: |
| Median of pair medians, complete step | 20.376 ms | 23.670 ms |
| Planned required bytes per GPU | 64,007,106 | 64,372,674 |

The median paired throughput change was **-13.99%**; the five pair changes were
-12.77%, -14.34%, -11.91%, -13.99% and -14.43%. The plan estimates are not
physical peak-VRAM or memory-traffic measurements. This fixture does not measure
resident-session latency, trained-model quality, other hardware or every beam
profile. It does not satisfy the approved default-promotion gate.

The source snapshots, binary hashes, complete rank logs, telemetry and failed
attempt are in `stage4-radix128-3060/`. The retrieved archive contains 232 files
and 1,279,312 uncompressed bytes, SHA-256
`48dbd8c1509956378a34467e0ebdad4b6b9b68d3c34838eaedff0ca68b04075b`.
Build products are retained in the ignored local cache and on the owned lease;
the Git evidence archive excludes executables, object files and ephemeral binary
transport files.
