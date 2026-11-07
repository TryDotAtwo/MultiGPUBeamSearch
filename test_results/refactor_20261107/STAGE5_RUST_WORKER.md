# Rust worker in the Python distribution

Previously the installed package had the PyO3 module but no `beam-runner`.
The wheel now compiles and includes the executable from the matching bundled
Rust workspace. The adapter resolves its installed path and rejects an absent
or non-executable file. Editable installations put the generated executable in
the ignored source-package directory. Building an ordinary wheel or sdist uses a
temporary staging copy; the C++/CUDA engine remains a separate lazy dependency.

On the owned 2x RTX 3060 host, wheels built from the checkout and a rebuilt
source distribution both contained the extension, worker and identical
329-file native-source manifests. Both workers answered the resident protocol
probe. Normal isolated pip build and editable install passed. Worker discovery
tests passed, including missing and non-executable paths. The isolated build
test checks the current host target; cross-compilation and other operating
systems were not exercised.

Using the installed wheel's worker and extension with two separately built
native engines, the repeated-session fixture matched the previous runner on one
and two physical GPUs. A separate Rust `SearchSession` fixture passed six cases
for each combination of one/two GPUs and CUTLASS/LibTorch execution: initial,
reused and unsolved jobs, model and beam changes, and solution collection. The
same fixture tested cancellation and an injected failed rank on two GPUs and
confirmed worker processes exited. These are bounded correctness and lifetime
checks, not broad acceptance of all model families or GPU architectures.

The test did not switch the public Python search controller or benchmark search
performance. Automatic production building and selection of the C++/CUDA
shared library remain to be integrated and compared before the public switch.
The checkout wheel SHA-256 is
`869c1036cff809602363f967cf178daace8d28a59924bda092fac3ff953bf185`.
The raw evidence archive SHA-256 is
`32a97675054996e5f8f446812d304a4f7c6ebea5f06477b985ab4eb3831cf1b3`.
Generated binaries, wheels and synthetic fixture weights are kept in the
verified local archive on drive D; their sizes and hashes appear in its manifest.
