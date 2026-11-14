# Public experimental Stream 4 index-sort selection

Status: implementation and CPU contract checks in progress. The default is
unchanged. The implementation connects the previously existing CUDA option to
`NativeOptions(experimental_index_sort=True)`; it does not change the CUDA sort,
reduction, inference kernels, or other experimental flags. Automatic profile
selection is not implemented.

Enabled and default-OFF managed builds have different content identities. A
cached runner is checked against the requested build specification and option;
an external runner must declare the enabled option in its receipt. The native
shared-library builder verifies the same mode and configures CMake accordingly.
Shared-profile compatibility distinguishes ON while keeping the legacy OFF
identity; malformed option metadata causes a profile miss. An independent
review identified the original shared-profile collision, reproduced in
`index-profile-red.log` and fixed with checks in `index-profile-green.log`.
Changing the option invalidates a persistent session's resource identity.
Rust's typed configuration accepts the boolean; Python omits the OFF field from
the Rust request so that a previously installed extension still understands
the default. Enabling it requires a matching rebuilt extension.

The test-first failures are retained in `index-public-red.log`. The Windows
mocked contract checks passed in `index-public-green.log`; a broader regression
initially exposed old NCCL test fixtures missing the new field and is retained
in `index-public-cpu-regression.log`. The final checks are in
`index-public-cpu-final.log` and `index-public-library-final.log`. These are
CPU tests, not CUDA execution.
The test runner's installed pyreadline3 emitted a console cleanup exception at
interpreter shutdown; pytest itself exited zero after reporting its result.

Hardware compilation, the Rust test suite, a real public session with option
changes, and paired end-to-end/step measurements remain pending in this stage.

The first Vast host had two RTX 3060 GPUs but NVIDIA driver 535.161.07. Its Rust
workspace tests and release build passed, while PyTorch CUDA 12.8 initialization
failed with error 804; no GPU workload ran. The Linux suite also found an older
mock that did not produce the now-required build metadata: 560 passed, 25
skipped, one failed. The corrected CPU fixture passed locally in
`index-public-backend-fixture-green.log`. This initial host was released after
its evidence archive was verified; see `public-index-first-3060/`. GPU checks
still require a host with a compatible driver.
