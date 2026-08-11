# CayleyPy Beam Search on Molab

This branch contains a Molab-native marimo launcher and four ready-to-run
examples. It reuses the production CUDA/C++ solver; the notebooks contain no
second beam-search implementation.

## Notebooks

| File | Purpose | Default workload |
|---|---|---|
| `cayleypy-notebooks/cayleypy_molab.py` | Universal checkpoint-only launcher | User-supplied competition and checkpoint |
| `cayleypy-notebooks/cayleypy_molab_444.py` | Piece Transformer example | Cube 4x4x4, puzzle 0, beam `2**16` |
| `cayleypy-notebooks/cayleypy_molab_megaminx.py` | Output-24 MLP example | Megaminx, puzzle 10, beam `2**16` |
| `cayleypy-notebooks/cayleypy_molab_ihes.py` | Output-1 MLP example | IHES Cube, puzzle 1, beam `2**16` |
| `cayleypy-notebooks/cayleypy_molab_tetraminx.py` | Output-1 MLP example | Professor Tetraminx, puzzle 0, beam `2**16` |

## Model contract

`MODEL_SOURCE_KIND` is fixed to `checkpoint`. Automatic export supports
batchnorm-folded MLP, resmlp-layernorm MLP, and the supported piece Transformer
bundle. The head must be `output_dim=1` or `output_dim=move_count`. Model dtype
is automatically FP16 and checkpoint format is detected automatically.

## Use

1. Open Molab and attach a GPU compute spec.
2. Import one `.py` notebook from this directory.
3. For the universal notebook, upload/expose a standard CayleyPy competition
   directory and a supported checkpoint, then edit only `USER CONFIG`.
4. Run the notebook. Examples download their public Kaggle assets through
   `kagglehub`; Kaggle credentials/rules acceptance can still be required.
5. Read `/tmp/cayleypy_molab/output/run_summary.json`. Solutions,
   `submission.csv`, logs, preflight, and `publish_status.json` remain there.

The inclusive puzzle range, beam, `first`/`collect`, reflection, collection
depth/capacity, touch-BFS radius, debug switches, and result publication are
configurable. Publication is best effort and token-free.

## Runtime differences from Kaggle

- Molab notebooks are marimo `.py` notebooks.
- GPU count and compute capability come from the attached Molab spec.
- CMake builds for that compute capability and torchrun uses the actual count.
- History RAM/disk budgets come from the live sandbox.

## Regeneration

```bash
python tools/build_molab_cayleypy_notebooks.py
```

Acceptance requires real Molab GPU runs of the main launcher and all examples.
