# Connect the CNOT device domain to the Rust search controller

Status: implemented as an explicitly selected distinct stage in the Rust CNOT
search. Set `MGBS_CNOT_OWNED_LIBRARY` to a trusted, compatible standalone CNOT
domain shared library to select it. With the variable absent, or for CPU
execution, the existing path remains active. Library discovery and automatic
building are not implemented by this change.

Each search rank loads the explicit library lazily and owns its own
thread-bound Rust domain; CUDA
addresses and streams remain rank-local and are never exchanged. Tensor storage
and the existing NCCL communication protocol remain unchanged during this
transition. The adapter passes validated, contiguous tensor views to the native
full-matrix first-unseen operation once per complete batch, with a caller-owned
workspace sized by the actual CUB query. Producer order, stream completion,
workspace lifetime and the lifetime of every borrowed tensor are explicit.
The C ABI writes the selected count on the GPU; the transitional Torch bridge
reads it on the host to form the result view.

Retain the current generation-order sort, full-state comparison, selected
candidate order, scorer precision, history, error coordination and collectives.
The existing implementation remains the default compatibility path. An
explicitly selected library missing the required ABI raises a coordinated
error; it is not silently substituted. Source-pinned one- and two-GPU tests
compare audited searches, checkpoint scorer behavior, mixed rank selection,
repeated calls and one-rank failure handling. Full-matrix identity, padding,
empty and capacity boundaries are covered by the standalone domain tests.
The GPU memory check required disabling CUDA API-error reporting because the
pre-existing Torch owner operation reports a reproducible invalid-handle API
error under Sanitizer; the unsuppressed failure is retained. Multi-node and
Slurm acceptance have not been established.
