# Python/Rust distribution and bundled source

The prior setuptools wheel contained the native source but no PyO3 module. A
direct maturin conversion built the extension but omitted the bundled source;
`stage5-rust-distribution-3060/native-red/test.log` records that regression.
The PEP 517 backend now stages the complete native source separately from the
checkout and delegates the Rust extension build to maturin. A source
distribution contains the matching source and rejects a file that no longer
matches its manifest (`tamper/pip.log`).

The checkout wheel and a wheel rebuilt from its source distribution each
contained both Python package names, the ABI3 extension and 329 native source
files. Their source-manifest content hash was
`76e8bc6bc0ae3f5239724c9a6925e344b79539657c180f11a1ab41bcdd86e26a`.
The wheel hashes differ because they are separate builds; the test verifies the
source manifest and each bundled file, rather than claiming reproducible wheel
bytes. Normal pip build isolation and an editable install also passed. Importing
the installed package and PyO3 extension did not map either native CUDA engine
library. The test exercises the standard PEP 517 interface; invoking bare
`maturin build` bypasses this project's staging backend and is not a supported
way to produce the complete distribution.

On the two physical RTX 3060 host, the installed package passed 528 Python tests
with 9 skips; the focused packed-CNOT CPU/CUDA suite passed 33 tests. The
native engine was compiled for SM86 from the wheel's bundled source. Using that
library and the installed extension, the resident-session fixture matched the
previous runner on one and two GPUs across three sequential jobs, preserved
worker PIDs and validated solution paths. This is a bounded permutation fixture,
not an end-to-end benchmark or acceptance of every model, precision, CNOT
checkpoint scorer, hardware generation, cancellation or failure path.

The current public Python search still uses the existing orchestration. The
wheel contains neither a ready `beam-runner` executable nor a compiled CUDA
engine; their integrated discovery/building and the public runtime switch remain
separate work. CUDA engine loading is lazy, and no search-step speedup is claimed
for this packaging change.

The checked-out wheel SHA-256 is
`4be77348879f6df776eaf2997065a953a11bbf5367792702f63d3087c1539427`;
the source distribution SHA-256 is
`daf5d922fff7af8f0491dc37bd457bc9c03fa790c30272c67ab250554e7bf15b`.
The local raw archive SHA-256 is
`498262297deca0c34900ea2aaf5781bccf545c61e03df284a097d00ba183b101`.
Generated binaries, wheels and synthetic fixture weights remain in that
verified archive on drive D; their sizes and hashes are in the evidence manifest.
