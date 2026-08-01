# Kaggle 2xT4 maximum beam validation (2026-08-01)

## Fully saturated limits

| Backend/model class | Maximum validated global beam | Alignment | Kaggle evidence | Depth 8 | Local frontier/rank |
|---|---:|---:|---|---:|---:|
| MLP output1 | 67,239,936 | 131,072 | `trydotatwo/cayleypy-public-acceptance-limit-p27-output1`, v4 | 1723.45 s | 33,619,968 |
| MLP output_move_count | 67,633,152 | 65,536 | `trydotatwo/cayleypy-public-acceptance-limit-p27-output24`, v5 | 54.3726 s | 33,816,576 |
| piece Transformer output_move_count | 67,108,864 (`2**26`) | measured p26 layout | existing real 2xT4 p26 evidence | 2765.04 s | 33,554,432 |

Both MLP maxima completed BFS-off saturation through depth 8 on two Tesla T4 GPUs with the standard 768 MiB device headroom. Both rank logs are free of fatal, overflow, and termination records.

The next aligned MLP steps are over the guarded device budget:

- output1: 67,371,008 requires an estimated 14,659,469,186 bytes versus a 14,638,972,928-byte budget.
- output_move_count: 67,698,688 requires an estimated 14,640,504,155 bytes versus the same budget.

The transformer `2**27` private probe ended in `DeadKernelError` without runner/preflight artifacts, so it is not accepted. Transformer selection remains fail-closed at the fully measured `2**26` profile.

The registry exposes per-MLP `maximum_validated_beam` values and rejects `max + 1`. All five public notebooks retain `TOUCH_BFS_RADIUS = 4` and pin solver commit `cec74003f262e9d93d2fc0cc56a5e2d49a5010e0`.

Final local gate: 318 tests passed.
