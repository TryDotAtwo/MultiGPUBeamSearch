# Native ownership and transfer register

Status: implementation in progress; changes require the acceptance results in
`test_results/refactor_20261107/`. The reference is main `0eafb193`.

## Process and ABI boundary

The Python module must not load the CUDA shared object. Rust launches one owned
worker process per rank. That worker loads the versioned native library lazily,
owns one native context, and keeps the library loaded until context destruction.
The initial ABI permits one context per process because the existing engine uses
the process environment, working directory, CUDA device selection and stdout.
It is an explicit restriction, not a claim of thread-safe independent sessions.

`prepare` copies argv and starts the engine thread. `submit` copies only a small
host job description (initial state, sequence, absolute output path). Preparation
loads the model and reserves GPU resources once; compatible jobs reuse them.
Rust does not dispatch individual GPU kernels or candidates. `beam-native-sys`
is the only Rust crate allowed to contain unsafe code or convert native pointers.
Device addresses must never become Rust references or cross a process boundary.

The native context owns the lifetime of the engine thread. Scope owners inside
that thread release graphs, events, streams, arenas, inference scratch, weights,
generator data and lookup tables in reverse dependency order. Cancellation is
cooperative at a depth boundary. The rank supervisor must terminate all its
workers after a bounded grace period if an external collective cannot finish.

Rust-owned history is installed through the optional ABI-1 host-history table.
The table and its boxed owner outlive every native writer callback. Native code
waits for D2H, dispatches a complete pinned batch to Rust, then releases the host
slot after the callback returns. Rust reads the aligned host view directly and
converts it into 16-byte history entries; it never retains the borrowed pointer.
Rust owns the RAM arena, reserved file, depth files, parent remapping and exact
capacity checks. Native pinned-slot futures still provide the transport-side
completion condition. RAM/disk history is reset per job; CUDA resources remain
resident across compatible jobs.

## Buffer contracts

| Buffer | Capacity / valid range | Writer and ready condition | Reuse condition |
|---|---|---|---|
| Current permutation frontier | physical rank capacity / GPU frontier count | final materialization; frontier-ready event | all Stream1/2 readers complete |
| Stream1 score rings | configured ring geometry / original candidate range | inference lane; score-ready event | Stream3 consumer releases ring slot |
| Stream2 hash rings | same original candidate range | expansion/hash lane; hash-ready event | Stream3 consumer releases ring slot |
| Stream3 sort/reduce planes | planned accumulator capacity / GPU compact count | Stream3 job on its stream | reduction and owner partition complete |
| Stream3 partition key/value A/B | max of batch and spill capacities / operation count | owner radix or shard partition on Stream3 | its sort and output gather complete before the next same-stream partition |
| Stream3 fatal flag and trace | one flag and 26 trace words / first failure | Stream3 operations in stream order | dispatcher reads the trace after the Stream3 completion event before exchange |
| Stream4 physical A/B | planned shard capacity / valid count | owner Stream4 job | all readers of old survivor version complete |
| Published histogram A/B | full score-bin count / active bank | Stream4 publishes inactive bank after writes | Stream5 finished reading selected bank |
| CUB temporary storage | exact queried byte capacity / operation-specific extent | stream that launches CUB | every previous reader/writer completes; graph overlap must not alias |
| Stream5 send/receive slots | physical slot capacities / rank count ranges | pack/NCCL receive | exchange and downstream consumer events complete |
| Final state slots | configured materialization chunk / actual selected count | final gather, with temporary local index | index consumed and persistent padding zeroed |
| GPU history source | selected metadata range | final materialization event | D2H copy completion event; independent of host file completion |
| Pinned history host slot | slot capacity / committed count | D2H stream | D2H complete **and** file/RAM writer finished reading the slot |
| CPU packed history | planned RAM/disk entries / committed depth ranges | history writer | path reconstruction and adjacent-depth pruning readers finish |

Capacity errors must report both required and available sizes. Physical alignment
does not change logical beam width. Stream4 has no semantic shard cap or shard
top-k. Permutation padding is zero in persistent storage. CNOT matrices use a
separate packed-word domain and full matrix equality, not hash equality alone.

## Transfer register

N is a valid element count, S the specialized state storage size, W the model's
encoded bytes, R the number of ranks, and C the configured slot capacity.
These are source-derived volumes; measured bus traffic must come from profiling.

| Source → destination | Logical volume | Frequency / purpose | Action |
|---|---:|---|---|
| Exported model → host encoded weights → device weights | W plus conversion buffers | preparation / model change | retain resident weights; numerical format policy unchanged |
| Host generators → device generator table | moves × S | preparation / graph change | retain; identity padding belongs to generator format |
| Host Zobrist → device Zobrist | S × alphabet pad × 16 | preparation / graph change | retain; padded hash contributions zero |
| Host solved-neighborhood/suffix tables → device tables | exact table capacities | preparation / repair target change | retain compatible tables; record target changes separately |
| Initial/target state host → device | S each | job start or target change | necessary control input, not a full frontier copy |
| Score/hash rings → compact/sort/reduce planes | operation-dependent N | each candidate batch | retain original candidate IDs; materialize full states only after selection |
| Stream4 sort metadata payload → ping-pong payload | 32 × N per logical pass, plus 16 × N keys; experimental index variant uses 4 × N indices | shard dedup | account for the immutable source and final gather; physical traffic remains unmeasured |
| Stream3 unique hash/value → owner key/value A/B → local or send ranges | input 24 × N; logical radix pair 36 × C per pass; gather reads and writes 32 × N | experimental multi-rank owner routing | reuses existing A/B planes; C includes sentinel tail and actual CUB traffic is unmeasured |
| Shard A/B → final logical union | at most combined valid entries | final exact dedup | necessary for cross-bank identity; full Hash128 and deterministic winner |
| Selected parent states → generated final states | S × selected states plus parent reads | each depth | necessary state transformation; preserve late materialization |
| Owner send slots → remote receive slots | valid packet bytes per rank | NCCL exchange | stable owner ranges avoid repeated full-array classification |
| GPU selected metadata → pinned history | 32 × N | each depth | batched D2H; event releases GPU source independently |
| Pinned metadata → packed RAM/disk history | 16 × N output; reads 32 × N | each depth | representation conversion; omit hash/score, zero entry padding |
| RAM history pruning → remapped adjacent depths | old/current valid entries | RAM pruning only | semantic path-index rewrite; cannot remove without replacement |
| GPU service counters / flags → host | mostly 4/8-byte fields and rank arrays | scheduling, errors and final selection | batch where dependencies permit; do not remove required NCCL count reads |
| Host rank offsets/counts → GPU | O(R) | exchange setup | small control transfer; separate from payload cost |
| Artifact files → hash accumulator | 64 KiB reusable host buffer per file | prepared model/helper verification, including cache hits | content binding; no model-sized host copy or device transfer |
| GPU error/solution records → host | valid diagnostic/solution count | error or solution | retain complete records and path replay checks |
| History entry routing host ↔ GPU/NCCL ↔ host | requested 16-byte entries plus routing fields | distributed path reconstruction | necessary cross-rank dependency; preserve rank/local-index identity |

Future measurements must separate these logical volumes, representation
conversions, device allocations, pinned allocations, and actual memory/bus
traffic. Replacing a sort payload is not by itself an end-to-end speedup claim.
