# Generic aligned Stream1 Transformer sequence design

## Goal

Make native CUDA/CUTLASS `piece_transformer` accept every supported puzzle shape without puzzle-specific sequence kernels. The manifest defines the logical shape; runtime selects an efficient aligned physical shape.

Immediate acceptance is Cube4 `seq_len=57`, `d_model=256`, `nhead=8`, `head_dim=32`, four layers, 24 outputs on A100 SM80. Existing Megaminx `seq_len=51` remains correct.

## Shape contract

- `logical_seq_len` comes from the model manifest.
- The active native attention backend selects `sequence_alignment`.
- `padded_seq_len = align_up(logical_seq_len, sequence_alignment)`.
- Allocations, scratch planning, CUDA Graph capture, and tensor-core launches use `padded_seq_len`.
- Model semantics, piece layout, positional data, and output interpretation use `logical_seq_len`.
- No fixed maximum such as 51, 57, or 64 is introduced. Existing State128, manifest, overflow, and device-capacity guards remain fail closed.

## Data flow

Input construction writes logical CLS and piece-token rows, then zero-fills `[logical_seq_len, padded_seq_len)`. QKV projection and attention operate on aligned storage.

Zero filling alone is insufficient because padded keys change the softmax denominator. Padded key logits are treated as negative infinity before softmax. Padded query rows are skipped where possible; otherwise their results are discarded and the tail is zeroed before the next layer. CLS remains row zero.

All layer strides use aligned physical storage. Output projection and scoring consume only logical rows and the logical CLS result.

## Native implementation

Replace exact `seq_len=51` validation in shared tensor attention with the general aligned-stride contract. Pass logical and padded lengths through FMHA. Preserve dtype, tensor-core, head-layout, alignment, and preallocated-scratch requirements. Do not add runtime allocation, CPU readback, LibTorch dispatch, or fallback inference.

Backend selection owns alignment. Unsupported GPU, dtype, or head configurations fail before graph capture with an explicit diagnostic.

## Compatibility

- MLP is unchanged.
- The manifest retains logical `seq_len`; padding is runtime storage, not architecture.
- Existing 51-token score-key digests remain within established native FP16 tolerance.
- Cube4 57-token results match a real-checkpoint PyTorch oracle within the same tolerance.
- Streams 2-5, State128 padding, and distributed payload contracts are unchanged.

## Test and performance gates

Implementation follows red-green-refactor:

1. Add a planning test that fails on the current 57-token rejection and asserts the nearest backend-supported aligned length.
2. Add a real-weight Cube4 CUDA correctness test and observe the current rejection.
3. Implement aligned planning, zero-tail handling, and masked FMHA.
4. Verify existing 51-token correctness and the full CUDA suite.
5. Benchmark graph replay for logical lengths 51 and 57, recording alignment, throughput, scratch bytes, and score digest.
6. Run an A100 SLURM smoke before resubmitting the 400M search.

Acceptance requires numerical correctness, no steady-state allocation, CUDA Graph compatibility, and no material 51-token regression. Alignment may be tuned per backend and hardware; puzzle-specific inference kernels must not return.

## Operational handoff

After verification, push a scoped `codex/` branch. Cluster commands pin the full commit, verify checkout, generate a complete `.sbatch`, submit with `sbatch --parsable -p kaf12`, print job/output identifiers, wait through `squeue`, and report `sacct` plus the log.
