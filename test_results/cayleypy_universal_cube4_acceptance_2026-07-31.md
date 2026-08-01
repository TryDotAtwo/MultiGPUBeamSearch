# Universal CayleyPy Cube4 2xT4 acceptance (2026-07-31)

- Private Kaggle kernel: `trydotatwo/cayleypy-2xt4-universal-cube4-acceptance`, version 10, status `COMPLETE`.
- Solver pin: `2639bce6800c03753591dbbc7e85805442228068`.
- Hardware: exactly two Tesla T4 GPUs, world size 2.
- Model: supported Cube4 piece Transformer, checkpoint SHA-256 `58af301a4f2b77d503b6e12d450589c64c076624d3e1ff291128c23663ad3164`, automatic adjacent metadata/generator/source-root discovery, fp16, output_dim 24.
- Data contract: 96 state positions, 6 value classes, 24 generators.
- Search: puzzle 0, requested/effective beam 65,536 (`2**16`), depth 8, first-solution mode, reflection off.
- Selected backend profile: Transformer anchor 16, B_MICRO 384, 2 shards, Stream3 slots 2, Stream4 batch/trigger 16,384.
- Build/export: successful; `production_runner_libtorch_stream1` compiled and both ranks used `stream1_executor=libtorch_eager`, `stream1_backend=piece_transformer`.
- Runtime: return code 0, solve 15.483 s including orchestration; rank beam reached 32,768 at depth 4 and both ranks completed depth 8 in about 6.336 s. No solution was expected/required at this depth; `submission.csv` preserved the blank sample row.
- Artifacts: selected profile, preflight, export manifest/weights, combined/rank logs, beam results, empty validated solution tables, submission, publish status.
- Safety scan: no OOM, overflow, fatal error, or runner error in the successful rank logs. Publishing was disabled for acceptance.

Earlier private versions intentionally retained diagnostic evidence for mount timing, metadata discovery, state/value-class separation, runner reconciliation, compile boundaries, native helper linkage, and Cube4 `moves`/`move_names` generator schema restoration.

## Public release

- Public Kaggle notebook `trydotatwo/cayleypy-2xt4-checkpoint-beam-search`, version 4, status `COMPLETE`.
- The no-input landing run exited cleanly through the documented `SETUP_REQUIRED` path and printed solver pin `2639bce6800c03753591dbbc7e85805442228068`.
- Users can Copy & Edit, attach a standard CayleyPy competition plus a supported checkpoint, and run the same privately accepted seven-cell launcher.
