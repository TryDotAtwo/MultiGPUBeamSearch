# Stage 5: on-demand native library and ABI inspection

This stage adds an explicit preparatory operation. It does not switch the public
Python search entry points to Rust and it does not promote CUDA optimizations.

`ensure_native_library` accepts a previously verified managed runner and builds
the corresponding `beam_native` or `beam_native_libtorch` target. It verifies
source, CUTLASS, toolchain, NCCL, optional PyTorch, runner and Rust worker
identities; re-applies the managed CMake configuration on a new library build;
and atomically records the library SHA-256 and actual ABI layout. A cache hit
rechecks the inputs, binary hash and ABI in a separate Rust process. It rejects
an inconsistent cache rather than silently reusing it. A changed worker binary
currently produces an explicit metadata mismatch instead of an automatic cache
refresh. The operation does not load `libbeam_native.so` into its Python process.

Evidence under `stage5-native-library-3060/` was obtained on an owned Vast
machine with two physical RTX 3060 GPUs, CUDA 12.8, using the source manifest
in `reconfigure-real/source-manifest.json`:

* The full Python integration suite passed 532 tests, with 9 skips; 31 focused
  native-library/build/Rust-adapter tests passed. The new test was also observed
  failing before the reconfiguration code was implemented.
* The final wheel and a wheel rebuilt from its source distribution were
  installed successfully, with the same 331-file native-source manifest. Both
  package names remained importable without loading the CUDA library.
* The Rust ABI probe read actual layouts from separately built CUTLASS and
  LibTorch native libraries. In the tested shape the state is 96 logical / 112
  physical bytes, 16-byte aligned; the candidate is 32 bytes and 32-byte
  aligned; the hash is 16 bytes and 16-byte aligned. A missing library failed.
* An initial cold CUDA build from the packaged sources and repeated cache access
  passed. In the final wheel test, an experimental CMake option was deliberately
  changed to ON; the preparatory build restored it to OFF, rebuilt and checked
  the library, and a subsequent call reused the verified receipt. This covers
  that managed option; it is not a proof about arbitrary external CMake settings.
* With the resulting library and wheel worker, the bounded deterministic MLP
  fixture compared resident Rust sessions with the previous native runner on
  one and two physical GPUs. Each ran three searches, including a repeated
  session and an unsolved input; paths and exact beam width matched and worker
  processes exited. This is a correctness/lifetime fixture, not a trained-model
  performance or full application acceptance test.

Raw logs, command records, manifests and receipts are in the evidence directory.
The SHA-256-verified archive including binary artifacts and synthetic fixture
weights is retained locally under `.local/cache/vast-59551108/`; those artifacts
are excluded from Git. Archive SHA-256:
`d713dcf392ea66a67b73bbfbe3de691479d41357f91e07973456dd3d6dd82e79`.

No T4, Hopper or Blackwell execution, CUDA Compute Sanitizer run, broad CNOT
acceptance, or paired full-step performance benchmark was performed for this
stage. Those remain required before claiming the corresponding acceptance.
