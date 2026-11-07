# Rust/native runtime

This is the first integration stage of the approved migration. Existing public
Python entry points are switched only after their comparative acceptance stage.
Profile selection, full history ownership and CNOT orchestration are still being
transferred. No CUDA sorting optimization or precision policy is enabled here.

## Build

Rust workspace builds do not require CUDA. `cargo build --release --workspace`
produces `beam-runner` and the PyO3 `_beam_runtime` library. The dependency graph
is pinned by `Cargo.lock`; the tested toolchain is recorded with each result.
Set `CARGO_TARGET_DIR` to a remote build directory (or an ignored directory on D
for local source-maintenance environments). The approved task runs builds on Vast.

The existing CMake configuration builds `beam_native` and, when
`BEAM_ENABLE_LIBTORCH_STREAM1=ON`, `beam_native_libtorch`. It uses the same CUDA
sources, explicit state specialization, CUTLASS and NCCL as the reference runner.
The LibTorch C++ ABI flag applies to all internal engine/control translation units.

Point `BEAM_NATIVE_LIBRARY` at the resulting absolute shared-library path before
invoking `beam-runner` with the existing native arguments. Without that variable,
the CLI looks for `libbeam_native.so` alongside its executable. Diagnostic modes
are forwarded through the coarse C ABI; `--resident-protocol-info` needs no CUDA
library. Torchrun and Slurm can launch one CLI process per rank with their existing
rank/environment contract.

## Context and sessions

`beam_context_create` checks ABI version and per-process exclusivity. `prepare`
copies argv and starts asynchronous resource preparation. `wait` observes either
readiness, task completion, timeout or terminal failure. `submit` copies the
small host job descriptor before returning and requires the next sequence.
Compatible jobs execute on the resident engine thread with the same resources.
`status`, `cancel`, bounded error retrieval and `destroy` complete the interface.

Rust `ResidentWorkers` owns only its launched process groups, atomic job/completion
files, sequence, deadline, cancellation token and log merge. A rank exit ends the
whole group. A two-second cleanup grace period precedes process-group termination
when a worker cannot leave an external collective. Paths and serialized states
are the process boundary; a GPU address is never a message payload.

The current native engine still uses process environment, cwd and stdout. Run it
in the dedicated worker, not in a notebook/Python process that owns CUDA tensors.
Independent Python sessions use independent worker groups. A `Context` cannot be
shared across Rust threads, and its native library cannot unload before it.

## Verification

- `cargo test --workspace`: layout, complete-hash order, domains, bounds, repeated
  jobs, stable worker PIDs, timeouts, cancellation and one-rank failure.
- `native_abi_tests`: the actual C ABI with an independent CPU engine fixture,
  including borrowed-input lifetime, invalid sequence, error text and cancellation.
- `BEAM_TEST_LIBRARY=... cargo test -p beam-native-sys --test context -- --ignored`:
  actual Rust calls through the CPU shared-library fixture.
- `rust/beam-python/tests/adapter.py`: repeated Python calls through the built
  PyO3 module, using owned CPU worker processes and no CUDA dependency.
- `tests/native_session_gpu.py`: real one/two-GPU comparison with the previous
  runner, exact global width, solved/unsolved jobs, independent path replay and
  resource reuse. Its exact Hamming MLP is a lifetime/correctness fixture.
- `tests/native_session_failure_gpu.py`: active cancellation and a missing native
  library in exactly one rank; all owned workers must be reaped.

The private extension can be packaged with `maturin build --release --manifest-path
rust/beam-python/Cargo.toml --out <wheel-directory>`. This component wheel does not
replace the public Python package. The recorded build used Rust 1.99.1, Python
3.11 and the CPython 3.10+ stable ABI; an older minimum Rust toolchain has not been
accepted.

These checks do not establish trained-model quality or a performance improvement.
Performance promotion requires the paired full-step measurements in the approved
plan. Per-device compilation and physical acceptance remain separate records.
