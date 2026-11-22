# Roll back failed static arena construction

The previous allocator returned an error without freeing its `cudaMalloc`
allocation if a phase-layout extent check threw. The static-memory test now
invalidates the union layout and streams layout separately after making a valid
plan, and checks that the partially constructed view is reset. It also tries
to allocate over an existing live arena and verifies that rejection preserves
the original allocation. The red test reproduced the first leak on the
unmodified allocator (`red/test.log`); the test explicitly frees that allocation
after observing the failure so the diagnostic run does not leak it further.

The allocator now keeps a rollback guard from successful `cudaMalloc` until
all views and CUB initialization complete. CUDA allocation errors before the
guard do not replace an existing view; layout and initialization exceptions
release/reset the new allocation. Reallocation is rejected before the guard is
created. No per-search-step operations or new GPU allocations were added.

On each RTX 3060, the static-memory test passed memcheck with zero errors and
zero leaked bytes and passed initcheck with API access checking. The CUB arena
histogram and logical-union tests also passed memcheck on both GPUs. Repeated
compatible Rust/native sessions matched the prior runner on one and two physical
GPUs (`session/session.log`). These are scoped failure-injection and session
checks; they do not test a real CUDA out-of-memory event, an injected failure
in one rank during NCCL startup, other GPU architectures, or all application
paths. Performance was not rebenchmarked because the change affects arena
construction/error unwinding only, not the repeated GPU search step.

The local raw archive SHA-256 is
`8b47ad48036944c333691e5b54d35e22fee1a6756b36c19fa3c5105760eea94f`.
The archive contains 11 synthetic fixture model files retained locally only;
their sizes and hashes are in the manifest. The 140 published evidence files,
commands and source manifests are under `stage4-arena-rollback-3060/`.
