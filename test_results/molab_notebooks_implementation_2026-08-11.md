# Molab notebook implementation status — 2026-08-11

- Branch/worktree: `molab/notebooks` at `D:/100XH100/.worktrees/molab-notebooks`.
- Baseline: canonical `kaggle/notebooks` commit `bfc3448e`.
- Deliverables: universal marimo notebook plus 444, Megaminx, IHES, Tetraminx.
- Solver CUDA/C++ architecture: unchanged.

The pre-existing public test baseline was `273 passed, 3 failed`; the three
failures were the existing Windows `/tmp` 50-GiB history-budget gate. Molab
browser access returned a connection reset before the UI, so local checks are
not presented as Molab acceptance evidence.

| Notebook | GPU/build | Export | Solve | Artifacts | Cloudflare |
|---|---|---|---|---|---|
| universal | pending | pending | pending | pending | pending |
| 444 | pending | pending | pending | pending | pending |
| Megaminx | pending | pending | pending | pending | pending |
| IHES | pending | pending | pending | pending | pending |
| Tetraminx | pending | pending | pending | pending | pending |

Each row requires actual Molab evidence: GPU name/count, compute capability,
build target, export manifest, validated solution, `run_summary.json`,
`submission.csv`, and `publish_status.json`.
# Live Molab verification update — 2026-08-12

- Molab account authentication: confirmed.
- Requested compute: `4 CPU`, `32 GiB`, `RTX Pro 6000 (Blackwell)`; GPU selection is visibly persisted in Configure Resources.
- Real 444 run exposed and fixed marimo config serialization (`dict(locals())` captured functions).
- Real 444 retry reached Kaggle asset resolution and returned a controlled failure:
  `SETUP_REQUIRED ... kaggle_competition 'cayley-py-444-cube', status=401`.
- The Molab account session does not propagate Kaggle credentials to `kagglehub`. Full build/export/solve remains pending until `KAGGLE_USERNAME` and `KAGGLE_KEY` are added through Molab Secrets. Credentials were deliberately not pasted into a notebook cell or log.
- Runtime/notebook branch at verification: `molab/notebooks`, `f98fe746`.
