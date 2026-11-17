# CNOT owned CUDA domain: implementation slice

Status: implemented as an optional standalone domain, with the limited checks
recorded in `test_results/refactor_20261117/STAGE_CNOT_OWNED_DOMAIN.md`. The
active CNOT search still uses the existing Torch-backed stage backend. This
slice does not switch the public search or claim full migration.

## Semantics and scope

- A state is `n` packed GF(2) rows, each containing `w = ceil(n/64)` 64-bit
  words, for `2 <= n <= 256`. The final word uses the existing tail mask.
- Each generated child is the existing physical CNOT applied to its original
  parent index. Move IDs, owner hashing, route meanings, full-word comparison,
  original child ordinal and first-unseen selection follow the current domain
  kernels. A hash never proves CNOT-state equality.
- One context belongs to one process, CUDA device and creating thread. It owns
  one stream and fixed-capacity device buffers. It does not contain or replace
  the caller's NCCL communicator; multi-rank search protocol remains unchanged.
- The ABI and Rust owner are intended for coarse batches, with no GPU pointer
  converted into a Rust reference or passed to another process. This slice uses
  explicit host upload/download for testing; active search must not switch to
  those copies as a replacement for existing GPU-resident tensor handoffs.

## Layout and ownership

For parent capacity `P`, child capacity `C`, closed capacity `V`, state stride
`S = n*w*8` bytes and exact CUB scratch request `T`:

| Buffer | Element and capacity | Owner / valid range |
| --- | --- | --- |
| parents | int64 `[P,n,w]`, `P*S` | context; uploaded `[0,parent_count)` |
| closed | int64 `[V,n,w]`, `V*S` | context; uploaded `[0,closed_count)` |
| indices, moves | two int64 `[C]`, `16*C` total | context; uploaded descriptor counts |
| children | int64 `[C,n,w]`, `C*S` | context; produced `[0,child_count)` |
| owners | int64 `[C]`, `8*C` | context; produced for valid children |
| ordinals | int32 `[V+C]`, `4*(V+C)` | context scratch; only during distinct |
| flags | uint8 `[C]`, `C` | context scratch; only during distinct |
| selected | int64 `[C]`, `8*C` | context; valid first-unseen ordinals |
| count | int32 `[1]`, 4 | context; written by CUB during distinct |
| temporary | bytes `[T]`, queried from CUB | context scratch; sort then select |

Every byte formula uses checked arithmetic. `V+C` must fit the existing CUB
int32 ordinal domain. Preparation queries temporary storage for the configured
capacities. Each distinct call queries its actual requirement and rejects it
with required/available bytes if it exceeds the reserved scratch. A zero child
count returns zero without launching sort/select.

All operations are serialized on the context stream. Upload synchronizes before
the borrowed host input may expire; download and the data-dependent distinct
count synchronize before returning. The context cannot be reused from another
thread. Drop/close drains its own stream before freeing its own allocations.
Updating an input invalidates derived output ranges; no stale output may be read
as current. Failed capacity checks must occur before submitting GPU writes.
Replacing parents also invalidates descriptors so no old parent index can be
used against a shorter frontier.

## ABI and transition

The C ABI is versioned and owns an opaque context. Rust dynamically loads it
inside `beam-native-sys`; the library must outlive all contexts, and unsafe
calls remain inside that crate. C++ exceptions never escape the ABI. Context
creation, preparation, batch upload, existing kernel launches, results,
synchronization, bounded error reporting and destruction have explicit status.

The first integration gate is standalone domain behavior on one/two physical
GPUs with each rank owning independent context memory. It is not a search or
multi-node acceptance gate. A later slice must define same-device borrowed
GPU views, stream/event readiness and scorer lifetime before connecting this
context to the search. PyTorch scorer compatibility, process collectives,
history and production CLI remain open until individually transferred and
verified. No new numerical mode or optimization is introduced here.
`BEAM_CNOT_OWNED_DOMAIN_ONLY=ON` configures just this library without requiring
CUTLASS or NCCL; the ordinary project configuration is unchanged by default.

## Verification before switching any public path

Compare generated matrices, owners and first-unseen indices with the existing
CPU/Python reference for small and wide matrices, duplicates and partial tail
words. Exercise zero and maximum capacity, exact overflow diagnostics,
repeated calls and invalidation, ABI mismatch, failure cleanup, wrong-thread
use and prior CUDA device restoration. Build for the requested CUDA targets;
record physical GPU results separately from compilation. Test two independent
contexts on two GPUs without exchanging their pointers. Run Compute Sanitizer
on target hardware before considering this slice accepted.
