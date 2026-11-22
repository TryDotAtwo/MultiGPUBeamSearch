# CNOT borrowed-device ownership slice

Status: implemented and checked as a standalone slice. The standalone domain
is not yet the active search backend.
This slice adds one optional, versioned borrowed-device operation for the
existing CNOT owner kernel. It does not change the search, transport, scorer,
precision, owner hash or result semantics.

## Contract

- The caller lends contiguous int64 input matrices and output owner slots from
  one CUDA device. Each view supplies a raw address, logical element count and
  device ID; a GPU address never becomes a Rust reference. Source and output
  regions must not overlap, and the caller guarantees the physical allocation
  extent, accessibility from the CUDA context and lifetime. The C ABI validates
  sizes, alignment, pointer kind and
  device, but cannot prove a foreign allocator's logical bounds.
- The caller supplies a valid CUDA stream for that same CUDA context. Producers
  on other streams must establish their own event dependency before this call.
  The kernel runs on the supplied stream; this first version synchronizes that
  stream before returning so all borrowed storage can immediately be reused.
  It does not copy matrix or owner data through host memory. The context's own
  stream is drained first; its previous CUDA device is restored on return.
- A foreign pointer, a mismatched stream/context, a short view, an overlap or
  a capacity overflow fails before launching the owner kernel. A CUDA runtime
  failure invalidates the context. Empty views permit null pointers.
- Rust exposes the operation only through an unsafe method with these explicit
  caller obligations. The versioned extension is optional: a library with only
  the prior host ABI still loads, and requesting this operation fails clearly.

The synchronous stream boundary is a correctness transition, not a proposed
performance optimization or a switch for the production Python search. The
later scorer bridge must retain tensor/allocation owners, stream/event
dependencies and context/library lifetime through the last GPU reader; no
cross-process device addresses are valid.

## Checks

On two physical GPUs, compare owner outputs to the existing CPU oracle with
one and multiple streams, including a producer and consumer on a non-default
stream. Exercise empty views, exact capacity, short/overlapping/host-pointer
views, wrong device and a stream from the other device, followed by context
reuse and previous-device restoration. Test old/new library loading, the Rust
unsafe boundary and Python/Torch-owned buffers, plus Compute Sanitizer on the
new path. Compile and physical execution are recorded separately. Existing
one/two-rank CNOT search remains a regression check, not proof that this
optional operation has been integrated into the search.
