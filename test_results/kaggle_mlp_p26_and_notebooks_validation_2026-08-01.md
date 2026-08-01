# MLP p26 and five-notebook Kaggle verification (2026-08-01)

## Measured MLP profiles on Tesla T4 x2

| Model class | Kaggle kernel | Version | Depth 8 seconds | Local frontier per rank | Result |
|---|---|---:|---:|---:|---|
| output1 | trydotatwo/cayleypy-mlp-p26-acceptance-output1 | 4 | 1688.03 | 33,554,432 | success |
| output_move_count | trydotatwo/cayleypy-mlp-p26-acceptance-output24 | 2 | 52.3487 | 33,554,432 | success |

Both runs requested and used beam 67,108,864, retained the standard 768 MiB device headroom, completed saturated depth 8, and had no fatal or overflow match in either rank log.

## Public notebook reruns

| Notebook | Version | Kaggle status | Runtime contract |
|---|---:|---|---|
| main landing | 9 | COMPLETE | expected SETUP_REQUIRED without user inputs; pin verified |
| 4x4x4 | 48 | COMPLETE | success, BFS 4, beam 2**16 |
| Megaminx | 11 | COMPLETE | success, BFS 4, beam 2**16 |
| IHES | 9 | COMPLETE | success, BFS 4, beam 2**16 |
| Tetraminx | 8 | COMPLETE | success, BFS 4, beam 2**16 |

All five generated notebooks pin solver commit `cb9e911914bdb4ac63c2b5306f554e5a76b9884e` and set `TOUCH_BFS_RADIUS = 4`. The four runnable examples have zero fatal/overflow matches in downloaded rank logs.

Final local test gate: 318 passed in 13.84 seconds at repository commit `78565a7`.
