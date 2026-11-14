# Single-read native permutation generators

The native permutation runner now parses generator bytes and their ordered names
from one read, owned by one `NativeMoves` value throughout the runner. Previously
the two vectors came from two independent reads and could disagree if the file
changed between them. Existing separate loader functions remain available.
The runner borrows both vectors without making another host copy; the necessary
host-to-device generator copy remains. Bi-uniqueness validation uses an indexed
stack array instead of a tree node per index, and the result vectors are reserved
before insertion. Generator padding stays the identity permutation; persistent
state padding remains zero.

One read supplies a consistent parsed byte sequence to both outputs; it does not
make an in-place external write atomic. The central state is still loaded
separately, and this change does not bind the resulting file contents to the
Rust session's validated graph hash. CNOT and inference kernels are unchanged.
This improves rank startup, not each compatible repeated call to a resident rank.

## Measured startup work

On the same remote CPU, five alternating pairs each performed 100 loads after a
warmup, using an uninstrumented loader and the same synthetic ordered JSON.
The operating-system file cache was warm. These figures are loader timings,
not cold end-to-end session timings:

| State positions / moves | Baseline median / 100 | New median / 100 | Median paired time change |
| --- | ---: | ---: | ---: |
| 96 / 24 | 158.314 ms | 68.6768 ms | -56.61% |
| 150 / 255 | 2.07678 s | 0.876924 s | -58.12% |

A separate instrumented executable counted ordinary C++ `new`/`delete` requests
for 20 loads. For 96 / 24, the count fell from 109,880 to 8,520, and the
largest simultaneously live requested total fell from 78,761 to 74,665 bytes.
For 150 / 255, the count fell from 1,696,320 to 82,680 and the corresponding
peak fell from 1,175,793 to 1,110,257 bytes. These counters exclude allocator
overhead, over-aligned allocations, stack use, RSS and GPU memory; the
instrumented timings should not be used as production timing estimates.

## Full-step comparison and checks

The two builds used the same RTX 3060 pair, FP16 synthetic Hamming96 MLP, 24
moves, logical beam width 4096, CUTLASS and frozen native CUDA Graph profile.
Each of five alternating process pairs included one warmup and five saturated
complete steps; the slower-rank time was used. The median of pair medians was
19.2869 ms for the baseline and 18.8792 ms for the new build. Individual
throughput changes were +3.808%, +2.160%, -2.311%, -0.682% and +4.832%, with
median +2.160%. This does not meet the 5% GPU speedup threshold; it supports no
claim of GPU acceleration. The measured reduction concerns requested host heap
bytes, not VRAM. The paired script retains its historical `radix` label for the
candidate; both tested builds have the experimental radix options OFF.

- The new loader consumed a FIFO that supplied exactly one generator document
  for 96 / 24, 120 / 18 and 150 / 255. A shim reproducing the old two-read
  behavior timed out on the second read, and passed on an ordinary file.
- Old and new loaders both accepted nine supported object/array and name-field
  formats and rejected inconsistent name order. The Windows C++ test passed 45
  checks; the Linux C++ test did the same. Linux AddressSanitizer and UBSan
  reported no errors on the 120 / 18 test and FIFO case.
- Linux Rust workspace tests passed with one ignored ABI fixture; the full
  Python integration suite reported **643 passed, 9 skipped**. Four bounded
  public Rust-session searches on one and two RTX 3060s solved their synthetic
  inputs, their paths were independently replayed, and repeated calls reused the
  same rank PIDs. These are not trained-model, CNOT, multi-node, CUDA Compute
  Sanitizer or T4/Hopper/Blackwell acceptance tests.

The tested base is Git commit `31240256672aa5b2412294e1e84436599b15dc8f`, with
497 base files and seven source/test overlays. The format test and two
uninstrumented timing sources were supplied separately and are bound to the
archive. `native-generator-snapshot/verify_results.py` compares the base files
with Git, verifies the overlays and separately supplied test sources, replays
the four GPU paths and checks five paired full-step receipts. The full archive
is `.local/cache/native-generator-20261115/evidence-chunks.tar.gz`, SHA-256
`4fd4ba74b0d2c6399580ad796777f254d1678a0d7f4f5486fed08168426aaa1f`;
all 207 blobs and 298 records were verified. The 938 omitted files are explicitly
listed as rebuildable intermediates. Compact, content-addressed rank and timing
receipts are in `native-generator-snapshot/trace-receipts.tar.gz`.

Vast lease `60705806` used two RTX 3060s at $0.1688889/hour. Its archive was
verified before destruction and a fresh provider inventory confirmed absence.
The last duration snapshot and the time of verified absence bound the elapsed
GPU/disk estimate to approximately **$0.0779–$0.0787**, excluding network and
without asserting an invoice. The task-specific stop guard remained alive
through the run and was stopped after verified instance removal.
