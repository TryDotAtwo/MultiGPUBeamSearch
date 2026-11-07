# Vast 59551108: recovery inventory

Before releasing the task-owned RTX 3060 instance, its project working trees,
CUTLASS repository and test results were inventoried and saved to disk D.
The first archive is a content-addressed snapshot of `beam-work`,
`beam-native-source`, `beam-baseline`, `beam-results` and `cutlass`. Its 17,202
unique blobs and 36,726 file/symlink records were checked against their SHA-256
digests on Vast, and the complete archive SHA-256 was checked after transfer:

`67e742f5ac4c651bf96d719ceabd428b5a1dd3d9163b0bf87766f1e20d96fecc`

The archive is `.local/cache/vast-59551108/owned-work-rescue-20261107.tar.gz`.
The checker found no missing `.pt`, `.pth`, `.ckpt`, `.safetensors`, `.fp16`,
`.onnx`, `.npy` or `.npz` files in those five trees. Two sampled files, including
a checkpoint and an architecture file, were restored on Vast and compared
byte-for-byte with their originals. This is a sampled restore plus a full blob
integrity check, not a full materialization of all 36,726 paths.

Models and test results discovered outside those trees were saved separately in
`.local/cache/vast-59551108/owned-work-supplement-20261107-v2.tar.gz`. Its
1,095 records refer to 130 verified content-addressed blobs; SHA-256:

`086709f251d63f562e4e441d30c423e02a043869ff4562db0eb4d3cc6a168c47`

The supplement includes model-named `.bin` files and the test-results folders
of the two build trees. A sample of the build-tree weights was restored and
compared with its original. The restoring program is retained at
`.local/consolidation/restore-vast-rescue.py`; it reconstructs files into a new
directory and checks each restored blob.

Rebuildable object files, shared libraries, ELF binaries and interpreter caches
were excluded from the first inventory; the native libraries, runners, wheels
and synthetic fixtures used by the published acceptance runs are preserved in
the separate verified stage evidence archives. The old baseline's Git database
was excluded because its clean checked-out commit `0eafb19369262a7d254058a0b4839bea7dfa3334`
is present in the local Git history. Its non-Git working files are in the first
archive. This inventory does not claim that unrelated Vast resources were read
or changed.
