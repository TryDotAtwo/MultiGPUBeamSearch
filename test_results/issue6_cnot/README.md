# Issue 6 evidence

## Two physical GPU acceptance

The private Kaggle run on two Tesla T4s used engine
`98bd0d1ed44db8906744d4f21e6bfeff60a8d2b0`, CUDA PyTorch 2.9.0 and the
original GL(4,2) checkpoint with its 102 input features. A separate
enumerator expanded Python integer rows, checked every eligible child,
re-evaluated its scalar value and compared both the membership and order
of the global frontier. It also replayed the returned construction and
computed the exact distance by BFS.

| Input rows | Head | K / global B | Measured layer | Returned construction |
| --- | --- | --- | --- | --- |
| `[1,2,7,8]` | Qhat | 11 / 8 | 11 distinct, 8 retained. Ranks scored 7+4 and retained 4+4. | `1→2, 0→2`. Length and exact distance 2. |
| `[1,2,7,8]` | Proposal | 11 / 8 | 11 distinct, 8 retained. Ranks scored 8+3 and retained 4+4. | `1→2, 0→2`. Length and exact distance 2. |
| `[1,7,4,10]` | Qhat | 3 / 2 | At depth two each GPU expanded a parent and proposed three moves. Hash owners scored 1+3 unique children. Each GPU received one of the two global winners. | `1→3, 2→1, 0→1` (physical IDs `[5,7,0]`). Length and exact distance 3. |

The deep fixture exercised both GPUs as sources, value owners and recipients
on the same saturated layer. See the [two-GPU trained results](kaggle_trained/)
and the [reproduction commands](../../integrations/cayleypy_native/CNOT.md).

## Consumer scorer

On an RTX 3070 Laptop GPU the adapter restored the original one-update GL(4,2)
checkpoint with all 102 input features. Both raw action heads and the scalar
value head were exercised. Scalar values agreed with the donor CLI on the
two held-out matrices. With input rows `[1,2,7,8]`, K=11 and global B=8,
both head modes returned the two physical gates `1 → 2, 0 → 2`. The measured
ordered frontiers differed between the heads. See the [scorer and build
receipt](checkpoint_scorer_rtx3070.md), [Qhat](trained_single_qhat_rtx3070.json)
and [proposal](trained_single_proposal_rtx3070.json).

## Physical gate routing at 144 wires

The input is the identity followed by gates `130 → 2` and `2 → 143`.
The directed alphabet has 20,592 entries. The diagnostic profile uses width
1 and a three-action shortlist. It forces the reduction to undo physical gate
428 before the full-alphabet goal check. Backtracking returns `[18592, 428]`
in construction order. An independent Python enumerator checked the full
eligible set, the ordered winner and every bit after replay.

| Runtime | Measured result |
| --- | --- |
| Two distinct physical Tesla T4 GPUs on one Kaggle host, NCCL | Both qhat and proposal modes passed. Rank 0 generated three children. Hash owners 0 and 1 evaluated one and two respectively. The selected edge for move 428 went from source 0 to hash owner 1 and back to frontier rank 0. |
| Sixteen actual Linux CPU processes on one host, Gloo | The same search, winner digest, physical moves and bitwise replay passed. The selected edge went through hash owner 3. |
| One RTX 3070 Laptop GPU | The same gates replayed. The domain suite checked every physical n=144 row transition in batches, all 5,592,320 ordered gate encodings across n=2..256, exact deduplication and high-ID route preservation. |

These runs use `WireFixture` to isolate routing. The two-T4 run used engine
`e2f75771ed79de59cf87ff16b801980d5e1461fe`, PyTorch 2.11.0+cu130 and
NCCL 2.28.9. Exact GPU identities, per-rank counts, mode outputs and the
two-process Linux regression result are in [kaggle_wide](kaggle_wide/).
Local inputs and results are in [domain_checks.json](domain_checks.json)
and [wide_single_rtx3070.json](wide_single_rtx3070.json).

## Scheduler and inter-node status

Two torchrun agents started a 2×8 process layout on one physical host.
Slurm placement was also tested with an 8+6 task distribution and one
visible GPU per task. In a separate 16-process Gloo run, a six-wire search
with K=25 and B=17 generated 425 candidates on its second layer. There were
313 distinct eligible matrices and exactly 17 winners. All 16 ranks expanded
parents, evaluated children and received winners. The same ordered frontier
was obtained on two physical GPUs. The [cluster report](cluster_20261106.md) records
the rank, device, process-failure and native runner checks. A physical
multi-host run and throughput measurements above eight GPUs remain open.

The complete Python adapter suite ran in fresh batches on Windows with CUDA
hidden. It passed 501 tests and skipped 28. The Linux Gloo and physical CUDA
checks above ran separately. [Adapter regression counts](adapter_regression.json)
