# Standalone CNOT borrowed-device owner operation

The optional extension invokes the existing CNOT owner kernel on borrowed,
contiguous CUDA input and output views. Its caller retains the actual
allocations and a valid same-context stream and establishes producer ordering;
the operation synchronizes that stream before returning. Declared lengths,
alignment, overlap, device identity and stream context are checked. A foreign
allocation's actual extent and lifetime remain the caller's responsibility.
No GPU pointer becomes a Rust reference or is passed to another process.

This is a standalone migration step. It is not used by `search_cnot`, does not
change the current NCCL schedule or scorer and is not evidence that the full
beam search has been ported. The prior C ABI still loads without the extension.

The source-pinned run used base `4846dd9d222cfb09a0df4d1acd556ff81dda7ac2`
and seven overlays. The plan document's status was updated after execution;
the verifier confirms that the only post-run changes to code/test overlays are
comments clarifying CUDA-context accessibility in the C and Rust interfaces.
Evidence archives under `cnot-borrowed-device3/` contain the source manifest,
scripts, logs, receipts, and the following separate stages:

- `cnot-borrowed-device3-evidence.tar.gz`: old-ABI negative check, native
  builds, GPU tests, sanitizers and Rust workspace. Its script stopped when
  a supplemental sanitizer launcher searched `debug/deps` while Cargo placed
  the executable under `debug/build`; its exit was 1, not a test failure.
- `cnot-borrowed-device4-evidence.tar.gz`: continuation on the same remote
  checkout, with the source manifest verified before and after. It locates the
  tested executable from Cargo's log, then runs Rust memcheck, one/two-rank
  search comparisons, the integration suite and cross-compilation. Exit was 0.
- Earlier failed diagnostic archives and `diagnostics-evidence.tar.gz` are
  retained. Initially the PyTorch test frames retained GPU tensors under
  Compute Sanitizer; all six reported allocations led to PyTorch's caching
  allocator, and a reference inspection showed the retained test frame. The
  final tests release those references and flush the allocator. A later
  racecheck run exposed a missing wait in the test's own asynchronous Torch
  consumer; the corrected test waits for its stream before reading on CPU.

On one host with two physical RTX 3060 12 GiB GPUs, CUDA 12.8 and driver
580.178.04:

- Four borrowed-view tests passed, covering default and non-default streams,
  n=4/64/144, both devices, empty views, short/overlapping/host pointers,
  wrong-device streams, reuse after rejection and previous-device restoration.
  The older and newer native libraries both loaded from Rust; the old library
  correctly lacked the optional extension.
- Compute Sanitizer memcheck reported zero errors and zero leaked bytes for
  both the borrowed Torch test and the independent native-domain test. The
  Rust borrowed-device test also passed memcheck with zero errors and leaks.
  Racecheck displayed zero hazards for the borrowed-view workload.
- The Rust workspace, explicit Rust GPU tests and formatting passed. Existing
  CNOT searches matched their references on one and two GPUs. The Python suite
  reported 833 passed and 12 skipped; the native checkpoint-scorer fixture
  was not built in this run.
- The standalone library compiled for SM75, SM80, SM90 and SM120. This is
  compilation evidence only for those targets; physical T4, Hopper, Blackwell,
  multi-node and Slurm execution were not performed here.

The owned Vast instance `60996988` was destroyed and provider absence verified.
Its final reported duration and rate imply approximately $0.080 in instance
charges, excluding possible fees and traffic. `verify_results.py` checks the
source binding, diagnostic outcomes, runtime receipts and release record.
