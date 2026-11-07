# Stage 5: require the PyTorch dependency identity

The on-demand shared-library builder now rejects a managed runner without a
PyTorch build identity when the runner uses LibTorch, a Transformer/ensemble,
or a calibration binary. The normal cache validation already compares an
available PyTorch version, CUDA version, C++ ABI and CMake prefix with the
current dependency; this check prevents missing identity metadata from bypassing
that comparison.

Evidence in `stage5-native-torch-identity-3060/` is pinned to its source
manifests:

* Four tests first demonstrated that incomplete metadata was accepted; after
  the change all 41 focused tests passed. The full Python integration suite
  passed 542 tests with 9 skips.
* The final wheel and a wheel rebuilt from the source distribution installed
  successfully and contained the same 331-file native-source manifest. The
  installed wheel discovered its own Rust worker and reused valid CUTLASS and
  LibTorch libraries without loading either library into its Python process.
* On the owned Vast host, deterministic resident MLP sessions from the final
  installed wheel matched the previous native runner on one and two physical
  RTX 3060 GPUs for both CUTLASS and LibTorch. These fixtures cover repeated
  searches and an unsolved input; they do not establish trained-model quality
  or an end-to-end speedup.

The public Python search orchestration has not yet been switched to Rust.
T4, Hopper, Blackwell, broader CNOT validation and paired full-step performance
measurement remain outside this stage's evidence. Synthetic weights and binary
artifacts are preserved in verified local archives rather than Git.
