# Rust search-session and result ownership — 2026-11-07

This increment follows `ab7b60bf`. Rust owns compatible resource lifetime,
rank launch, cancellation, request deadlines, typed permutation replay and
incremental terminal/collection validation. Public Python orchestration has not
yet switched. Existing native numerical kernels are unchanged.

All compilation and execution used owned Vast 59551108, two RTX3060 GPUs,
USD 0.1342222222/hour including disk. Local source and artifacts remain on D.

`stage3-session-3060/manifest.json` hashes 986 downloaded files, 1,223,891 bytes.
Archive SHA-256:
`abd244df9537c988c70410ed320533ad1f5047787c2e2c845165d3ba38cfd773`.
`source-final.json` binds the final executed source. Later documentation edits
are not claimed as executed code.

| Check | Observed result |
| --- | --- |
| Rust workspace | 41 CPU tests passed; one real-FFI test explicitly ignored in this command and covered previously |
| PyO3 session adapter | trivial tasks, process ownership and active thread cancellation passed |
| Native CUDA Graph / actual LibTorch eager | 24 one/two-GPU cases passed: first, repeated, unsolved, changed model, changed beam 64→65, collection |
| Retained resources | compatible calls retained PIDs; model/geometry changes reaped old groups |
| Two-GPU failure handling | active cancellation and exactly one missing-library rank reaped all owned workers |
| Original engine / Rust LibTorch eager | three repeated solved/unsolved jobs on one/two GPUs matched legal paths and exact width |
| Output compatibility | legacy Python parser and Rust metadata/path results matched; atomic stored outcomes matched returned values |

`gpu-final/` explicitly sets and checks `BEAM_STREAM1_EXECUTOR` in every rank's
bootstrap log. Earlier `gpu/` results loaded both library variants but selected
the native default. This also exposed an overstatement in STAGE2/STAGE3_HOST:
those historical "LibTorch" session checks exercised a LibTorch-linked library
with `native_cuda_graph`. Their reports now state the narrower scope; raw logs
and hashes are unchanged. The standalone stage-3 LibTorch inference benchmark
already exercised its eager scorer.

Reproduced defects: `deadline-red.log` shows protocol preparation previously
resetting the execution timeout; one request deadline now covers both stages.
`path-spaces-red.log` shows an unquoted diagnostic `result_tsv` path with spaces
being parsed as a numeric collection field; parsing now stops at that diagnostic
suffix. Missing-interface RED logs are retained alongside these reproductions.

The parser checks finite timings, puzzle identity, legal complete paths,
terminal multiplicity, collection count/uniqueness and owner-rank bounds. It
rejects incomplete collection receipts and negative reconstructed touch depths.
It stores one terminal path, iterates bounded log lines, and sorts collection
indices for duplicate detection rather than copying every path again.

These are correctness/lifetime fixtures, not production speed or trained-model
quality acceptance. No C++ source changed in this increment; no new native
sanitizer/architecture claim follows. Configuration/cache orchestration, CNOT,
CUDA bulk improvements and final public switching remain outstanding.
