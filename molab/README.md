# Molab Workspace

This directory holds molab-specific notebooks and run notes. It is separate from
the Kaggle notebooks because molab has different paths, authentication, and
runtime constraints.

The measured single-GPU SM120 Cube4 Transformer profile is stored in
`../configs/molab_sm120_cube4_transformer_profiles.json`. Profile key `25`
means beam `2**25`; it is valid only for the exact hardware/model contract in
that file. In particular, `ring_graph_execs_per_lane=12` means two graph-window
rings for 24 accumulation slots and four inference lanes, not twelve rings.

## Files
- `molab_probe.py`: marimo notebook/script that probes the molab runtime,
  checks GPU availability, runs a small GPU task, and optionally clones/builds
  this repository for a tiny CUDA sanity run.
- `probe_cell.py`: single-cell version of the same probe, useful when pasting
  into a fresh molab notebook.

## Intended Molab Flow
1. Open molab in the Codex in-app browser and sign in.
2. Create a new notebook or fork/import a notebook.
3. Paste `probe_cell.py` into one Python cell and run it.
4. Save the output under `test_results/` in this repository.

Molab cannot see local uncommitted files unless they are uploaded/imported or
pushed to a remote Git repository. The probe therefore clones the public GitHub
repository when it attempts a project build.

## 2026-06-03 Probe Result
- The signed-in notebook runtime was CPU-only by default: `nvidia-smi`, `nvcc`,
  `cmake`, and `ninja` were missing, and `torch.cuda.device_count()` was `0`.
- cgroup quota reported `20` CPU cores and `160 GiB` memory, although host-level
  `free -h` reported `1.0 TiB`.
- Official molab public preview documentation says GPU can be attached from the
  compute/specs control: one NVIDIA RTX Pro 6000 Blackwell GPU with 96 GB VRAM.
  A 2-GPU option was not verified in the current session.
- CPU-side project checks did run on molab from the public GitHub clone:
  `contract_tests=pass` and `history_tests=pass`.

See `test_results/molab_runtime_probe_2026-06-03.md` for the detailed run log.
