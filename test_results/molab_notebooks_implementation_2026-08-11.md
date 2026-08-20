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

# Native pair verification update — 2026-08-20

- Connected through the official `marimo-pair` HTTP execution protocol; session discovery and code execution returned HTTP 200.
- Hardware: one `NVIDIA RTX PRO 6000 Blackwell Server Edition`, 97,887 MiB VRAM, compute capability 12.0 (`sm_120`), driver 580.126.20, driver CUDA 13.0, PyTorch 2.11.0+cu130.
- The explicitly authorized Kaggle credential was installed in the private sandbox with mode 0600 and was not printed. Protected 444 competition data (501 KiB) and the Transformer dataset (12.1 MiB) downloaded successfully.
- Model export succeeded: `piece_transformer`, FP16, sequence length 57, d_model 256, four layers, output_dim 24.
- Live failures diagnosed in order: missing CMake; hidden system nvcc; mixed nvcc 13.3 versus headers/runtime 13.0; incomplete CUDA wheel namespace; missing CCCL `nv/target`; missing NVTX header. A fully matching CUDA 13.0 toolkit (nvcc/crt/nvvm 13.0.88, runtime 13.0.96, CCCL 13.0.85) reached actual `sm_120` CUDA compilation.
- The sandbox returned HTTP 410 `sandbox terminated` during the next long rebuild. No final solver result or Cloudflare publication can be claimed from this session.
- Targeted local regression for the new bootstrap: 10/10 Molab tests passed. Combined Molab/public suite: 267 passed and 21 failed solely because the Windows host had only 27.7 GB free under `/tmp` against pre-existing 32/50 GB history-budget tests; those failures are unrelated to the Molab toolchain change.
