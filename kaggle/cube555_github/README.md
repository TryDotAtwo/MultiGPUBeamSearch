# Cube555: CayleyPy + MultiGPUBeamSearch

[Open the public Kaggle notebook](https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend).
Run the cells in order with GPU acceleration enabled. The first cell controls
puzzle index, exact global beam, GPU count, depth limit and explicit blend weights.
The setup cell fetches immutable solver commit
`ed343a4f84c0e6900941bd057504683e044741bb`, installs the adapter and pinned CayleyPy,
and checks the requested GPUs. Checkpoints and inputs are checksummed GitHub assets.
No Kaggle dataset or competition attachment is required.

The tuner freezes the measured inference batch, generates hardware-specific
candidates, and compares complete steps before refining around the winner.
It reports its finite stopping reason and reuses compatible cached measurements
only after fresh memory admission. Capacity discovery never changes the beam.

Current defaults: index1020, beam1,048,576,2GPUs,140steps. That exact workload
and2xT4 execution have not been verified by this publication. Single-GPU
Cube555 replay and exact-width/adaptive tests passed;
see [the validation report](../../docs/validation/cube555-adaptive-20261025.md).

For local Linux use, open the same notebook with NVIDIA drivers, CUDA Toolkit
and a compatible CUDA PyTorch installation. Source/build/cache trees are under
`/tmp`; output summaries go to `/kaggle/working` when present. Every returned
solution is independently replayed before saving. No automatic publication of
solutions is enabled.
