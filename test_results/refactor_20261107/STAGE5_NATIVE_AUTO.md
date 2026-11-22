# Stage 5: resolve `auto` from the managed runner

The public configuration defaults to `inference_backend="auto"`. The runtime
resolves that value during runner preparation, but the initial on-demand shared
library builder compared the original `auto` with the resolved runner metadata
and rejected it. The builder now selects the target, CMake configuration and
cache identity from the verified runner's explicit `cutlass` or `libtorch`
value. An explicit backend mismatch, an unresolved `auto`, a missing value or
an unsupported value in runner metadata fails before starting a build.

Evidence in `stage5-native-auto-final-3060/` is pinned to its source manifests:

* Before the change, the new `auto` tests failed for both CUTLASS and LibTorch
  with the original metadata mismatch. Afterwards, 37 focused tests passed;
  the full Python integration suite passed 538 tests with 9 skips.
* On the owned Vast host with two physical RTX 3060 GPUs and CUDA 12.8, `auto`
  reused an existing verified CUTLASS library. A fresh managed LibTorch runner
  and library were also built and reused through `auto`. For each executor, the
  deterministic MLP fixture compared Rust resident sessions to the previous
  runner on one and two GPUs, including repeated and unsolved requests.
* The final wheel and a wheel rebuilt from its source distribution both
  installed successfully with the same 331-file native-source manifest. The
  installed checkout wheel discovered its own Rust worker, reused the verified
  CUTLASS library with `auto`, and passed the same one-/two-GPU resident fixture.
  Preparing or inspecting the library did not load it into the Python process.

These are bounded contract, packaging, and session correctness tests; they are
not a full application benchmark or trained-model quality test. The public
search orchestration has not yet switched to Rust. No T4, Hopper or Blackwell
execution, broad CNOT validation, or full-step performance claim follows from
this stage. Binary artifacts and synthetic fixture weights are in the verified
local evidence archive, not Git.
