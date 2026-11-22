# Public experimental Stream 4 index-sort selection

Status: explicit public opt-in implemented and checked on a bounded two-RTX-3060
fixture. The default is
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

The first Vast host had two RTX 3060 GPUs but NVIDIA driver 535.161.07. Its Rust
workspace tests and release build passed, while PyTorch CUDA 12.8 initialization
failed with error 804; no GPU workload ran. The Linux suite also found an older
mock that did not produce the now-required build metadata: 560 passed, 25
skipped, one failed. The corrected CPU fixture passed locally in
`index-public-backend-fixture-green.log`. This initial host was released after
its evidence archive was verified; see `public-index-first-3060/`. GPU checks
were then repeated on the compatible host described below.

## Compatible-host results

The retry used two physical RTX 3060 GPUs, NVIDIA driver 595.99.02, CUDA 12.8,
PyTorch 2.9.0+cu128 and CUTLASS `afa1772203677c5118fcd82537a9c8fefbcc7008`.
It built and tested Git commit `84f8d308ea26dd6842c08f4973fe155d1919628e`.
The Rust workspace release tests and release build passed. The Linux Python
suite reported **577 passed, 9 skipped**. The previous Rust extension accepted
the default OFF descriptor and rejected the new ON field; the newly built
extension accepted ON. An initial retry failed because its helper had not yet
finished transferring; `legacy-compat-attempt1.log` and the continuation are
preserved in `public-index-3060/`.

On both CUTLASS and LibTorch, 28 bounded cases per backend passed: Python and
Rust session owners, one and two GPUs, first and repeated searches, unsolved
input, model change, ON-to-OFF index-sort change, beam change and solution
collection. The independent verifier replayed every returned path, checked
both owners' results and compared the same model/input with index sorting ON
and OFF. These synthetic Hamming cases do not establish CNOT, trained-model,
multi-node or cross-architecture acceptance.

The five alternating paired repeats used two GPUs, width 4096, six depths,
synthetic Hamming FP16 and warm Rust sessions. The median complete-call wall
times were **0.413773 s OFF** and **0.393924 s ON**; the median of the five
individual ON/OFF wall-time changes was **-2.74%**. Worker medians were
0.222066 s and 0.211925 s respectively. The reported slower-rank average
depth medians were 34.802 ms and 33.303 ms, but those averages include startup
and frontier growth: they are not isolated saturated full-step measurements.
Cold-session calls with already-built binaries were 2.257 s and 1.872 s in
that order; they are single observations, not a controlled cold-build test.
The single GPU-memory snapshot includes both resident sessions and cannot
compare their separate peaks. The earlier dedicated full-step and sampled VRAM
experiment remains in `../refactor_20261107/STAGE4_INDEX_SORT.md`.

The retry archive on D is `.local/cache/public-index-20261114/evidence.tar.gz`,
SHA-256 `d2878a7258afccdad18fa84fb7cbeb3d292036c454d71ccdd21783ac596978fd`.
Its 2,371 records and 734 unique blobs were verified and sample files restored;
3,908 rebuildable intermediates and external NCCL snapshots are listed as
omitted. `public-index-3060/verify_evidence.py` independently reproduced the
native build digest from 239 Git source files, checked all four ON/OFF and
CUTLASS/LibTorch binary receipts, and rechecked results and paired medians.
Both task-owned leases were destroyed and provider absence was verified. Their
combined elapsed-time GPU/disk estimate is about **$0.124**, excluding network
charges; this is not a provider invoice.

The option remains experimental. The public fixture does not independently
measure per-mode peak VRAM, allocation/transfer traffic or a saturated step;
automatic profile selection and T4/Hopper/Blackwell hardware acceptance remain
open. Existing inference kernels and numerical policy were not modified.
