# Experimental bounded score-histogram radix bits

Status: opt-in only. The measured profile did not reach the 5% full-step
promotion threshold and had unchanged planned memory. The ordinary Python
build explicitly selects `BEAM_EXPERIMENTAL_SCORE_BITS=OFF`.

For Stream 3 and Stream 4 score histograms, valid scores are in
`[0, SCORE_BIN_COUNT)` and the invalid/tail sentinel is `SCORE_BIN_COUNT`.
The experimental CUB sort uses the number of low bits needed to represent the
sentinel: 19 bits for this configuration instead of 32. The same shared
constant controls launch and scratch-size queries. Hash128 sorting and identity
still use all 128 bits. No model precision or inference kernel is changed.

## Verification and limitation

The source manifest in `green2/`, `default1/` and `combined2/` has SHA-256
`0e8c50092b7fbe5ff680395e2caffebeca5e3dd02ed185d3fc2a4fd1eb47f216`.
The Stream 3 and Stream 4 score-histogram test compared every histogram bin
with a CPU oracle on empty, partial and full 513-item inputs, repeated values,
zero, the maximum valid score, the high score bit, invalid scores and tail
sentinels. Eager execution and repeated CUDA Graph launches passed on both
physical RTX 3060 GPUs under memcheck with zero errors and leaked bytes.
The ordinary CUDA regression, static memory, final-count and 25 Python build
tests passed. Repeated native sessions matched the reference on one and two
GPUs. With score bits, owner radix, index sorting and scan all enabled, their
scoped regression and four memcheck tests passed.

An uninitialized-scratch initcheck run reported 9,912 reads inside the CUB
temporary allocation during Stream 4 hash `ReduceByKey`. The same count and
scratch offsets reproduced with score-bit optimization OFF. With the test's
scratch explicitly zeroed before first use, both builds passed initcheck and
the CPU oracle. The installed CUB header initializes tile status padding and
then its lookback path can load the associated value storage; the address
evidence is consistent with that path, but does not by itself prove all such
reads harmless. The production arena's scratch initialization has not yet been
established by this experiment. Failed uninitialized runs, allocation ranges,
CUB header SHA-256 and the successful preinitialized controls are retained in
`sanitizer1/`, `diagnostic1/` through `diagnostic4/`, and `cub-source/`.
`sanitizer2/` initcheck explicitly uses preinitialized scratch. These results
are not a clean uninitialized-production-scratch initcheck or multi-node/CNOT
acceptance.

Subsequent production-arena work initialized the union of CUB scratch ranges
and repeated the production-pointer reduction and final-union initchecks; see
`STAGE4_CUB_INITIALIZATION.md`. The historical failing logs above are unchanged.

## Paired full-step measurement

Five alternating process pairs used two RTX 3060 GPUs, the same Hamming96
scalar MLP FP16 model and input, width 4096, frozen profile, one warmup and five
complete steps. The two binaries were built from the same source snapshot and
differed only in the score-bit option. Times use the slower rank.

| Pair throughput changes | Median change | Median pair-median step, control / 19-bit | Planned bytes per GPU, control / 19-bit |
| --- | ---: | ---: | ---: |
| +0.85%, +3.32%, -1.73%, +1.69%, -0.69% | +0.85% | 20.192 / 20.022 ms | 64,007,106 / 64,007,106 |

The measurement is not cold-start, repeated-session performance, physical
peak VRAM or physical memory traffic. The 334-file raw evidence archive has
SHA-256 `c2969b8c78bcc2895c31da327ff54997aa98c1183a13ecc1908cb5db93079104`;
per-file hashes and source/build identities are under `stage4-score-bits-3060/`.
