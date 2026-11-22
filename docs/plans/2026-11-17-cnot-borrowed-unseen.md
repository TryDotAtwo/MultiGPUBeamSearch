# CNOT borrowed-device first-unseen selection

Status: implemented and checked as a standalone ABI. The operation remains separate
until it is connected to and compared against the public distributed search.

Expose the existing full-matrix CNOT sort and compact operation through the
thread-bound Rust/CUDA domain. Closed and child matrices, ordinal and flag
storage, selected child indices, device count, and CUB scratch are borrowed
from the caller. Their element types and required extents are explicit; scratch
size is obtained from the existing CUB query for the actual batch dimensions.
No temporary buffer is silently allocated, truncated or shared across ranks.

The caller retains the CUDA allocations and a same-context stream, orders all
producers before the call and prevents conflicting access. Mutable regions do
not overlap each other or an input. The operation synchronizes the supplied
stream before return and does not copy data through the host. Empty child
batches write a zero device count. Older domain libraries continue to load.

Check full matrix identity including the final word, canonical tail masking,
closed-set exclusion, first-occurrence order, empty and capacity boundaries,
workspace overlap and alignment, same-device/stream rules, context reuse,
old/new ABI loading, and the Rust FFI. Run CUDA sanitizer and one/two-rank
search comparisons on remote GPUs; report cross-compilation separately.
