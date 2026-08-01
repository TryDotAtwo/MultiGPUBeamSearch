# Kaggle 2xT4 maximum beam validation (2026-08-01)

## Fully saturated limits

| Backend/model class | Maximum validated global beam | Alignment | Kaggle evidence | Depth 8 | Local frontier/rank |
|---|---:|---:|---|---:|---:|
| MLP output1 | 67,239,936 | 131,072 | `trydotatwo/cayleypy-public-acceptance-limit-p27-output1`, v4 | 1723.45 s | 33,619,968 |
| MLP output_move_count | 67,633,152 | 65,536 | `trydotatwo/cayleypy-public-acceptance-limit-p27-output24`, v5 | 54.3726 s | 33,816,576 |
| piece Transformer output_move_count | 74,203,136 | 16,384 | trydotatwo/cayleypy-transformer-p27-limit-probe, v8 | 1562.95 s | 37,101,568 |

All three maxima completed BFS-off saturation through depth 8 on two Tesla T4 GPUs with the standard 768 MiB device headroom. Both rank logs are free of fatal, overflow, and capacity-error records.

The next aligned MLP steps are over the guarded device budget:

- output1: 67,371,008 requires an estimated 14,659,469,186 bytes versus a 14,638,972,928-byte budget.
- output_move_count: 67,698,688 requires an estimated 14,640,504,155 bytes versus the same budget.

The transformer maximum was derived from adjacent failing memory probes at a measured 192 bytes/global state. Beam 74,203,136 completed the depth-8 workload on both ranks; the next aligned beam 74,219,520 requires 14,639,246,672 bytes, exceeding the 14,638,972,928-byte guarded budget by 273,744 bytes. Transformer selection fails closed above 74,203,136.

Both registries expose per-model-class maximum_validated_beam values and reject max + 1. All five public notebooks retain TOUCH_BFS_RADIUS = 4 and pin solver commit f2261a81873ea162a0c36ba8874b9173db8d4083.

Final local gate: 318 tests passed.

## Final public notebook rerun

- Main launcher v14: COMPLETE; expected SETUP_REQUIRED handoff with no attached user inputs.
- 444 Transformer v54: COMPLETE, status success, BFS radius 4, maximum_validated_beam 74,203,136.
- Megaminx MLP v14: COMPLETE, status success, BFS radius 4, maximum_validated_beam 67,633,152.
- IHES MLP v14: COMPLETE, status success, BFS radius 4, maximum_validated_beam 67,239,936.
- Tetraminx MLP v11: COMPLETE, status success, BFS radius 4, maximum_validated_beam 67,239,936.