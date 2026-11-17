# CNOT borrowed-device expansion

Status: implemented and checked as a standalone slice. It transfers the existing child-generation kernel
to the standalone Rust/CUDA domain; public search still uses its current Torch
operator and multi-rank protocol until separately compared and connected.

The optional device ABI adds a device-only context and a synchronous borrowed
expansion call. The context retains its geometry, creating thread and stream,
but does not allocate the host-ABI parent/child/descriptor/CUB buffers. The
host-buffer operations reject such a context; the existing full context and
host ABI are unchanged. Both kinds may run borrowed device operations.

Input parents, int64 parent indices, int64 or uint16 physical move IDs, and
int64 output matrices are caller-owned CUDA views. Their element counts use the
specified element type. The caller supplies actual accessible allocation bounds
and lifetimes, valid descriptor values, a live same-context stream, producer
ordering and nonoverlapping output. The implementation checks declared extents,
alignment, pointer device, capacities, stream context and output overlap; it
does not prove foreign allocation bounds or descriptor values. The existing
kernel's device-error behavior for invalid CUDA descriptors is preserved.
The context stream is drained before the launch; the borrowed stream is
synchronized before returning. No tensor data are copied through host memory.

Older libraries still load: the new symbols are optional and independent of
the existing device-owner ABI. Rust never turns a device address into a
reference, and the raw device operation remains unsafe with the stated caller
obligations. No GPU address is shared across ranks.

Compare outputs with the existing CPU oracle on both physical GPUs, several
matrix widths including tail padding, both move formats, default/non-default
streams, empty and exact-capacity inputs, invalid declared extents, overlaps,
device/stream mismatch, previous-device restoration and context reuse. Verify
old/new library loading, device-only rejection of host operations, the Rust
boundary, Compute Sanitizer, and the unchanged one/two-rank search. Record
cross-compilation separately from physical execution.
