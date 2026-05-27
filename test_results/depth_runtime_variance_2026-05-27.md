# Depth Runtime Variance Diagnostic 2026-05-27

## Input

- User supplied depth logs for a T4x2 `beam=2**26` style run.
- Question: why some depths run around `45s` while adjacent depths run around `17s`.

## Observed Pattern

- Depths `12` and `14` start with `frontier_size=33554432`, run `ring_slot_jobs=4096`, run `stream3_jobs=4096`, and finish around `43-45s`.
- Depths `13` and `15` start with `frontier_size~3.2M`, run `ring_slot_jobs~388-392`, run `stream3_jobs~388-392`, and finish around `17s`.
- `4096 / 392 ~= 10.45`, matching `33554432 / 3210053 ~= 10.45`.
- Full-frontier depths generate about `33554432 * 24 = 805306368` move candidates per rank.
- Small-frontier depths generate about `3210053 * 24 ~= 77041272` move candidates per rank.

## Interpretation

- Wall time is dominated by current-depth input frontier expansion and Stream3 batch count, not by the previous depth number itself.
- Depths `12` and `14` are full-input depths: Stream1/2 and Stream3 must scan a full local beam before final selection shrinks the output to about `3.2M`.
- Depths `13` and `15` are small-input depths: Stream1/2 and Stream3 process about one tenth as many parents, then expansion refills the beam.
- The speedup is less than `10x` because small-input depths still do full final materialization/history for `33554432` selected candidates and launch more Stream4 jobs (`87` vs `12`).
- `final_threshold=4294967295` on depths `12` and `14` means the final survivor set is below the beam target, so no finite score cut is needed and the next frontier remains about `3.2M`.

## Conclusion

- `45s` depths are full-frontier input plus small output.
- `17s` depths are small-frontier input plus full output.
- The alternating runtime pattern is expected from alternating `next_frontier_size` values unless scoring/dedup/threshold behavior is changed.
