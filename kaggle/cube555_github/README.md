# Cube555 blend through CayleyPy

[Open the Kaggle notebook](https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend).

Run the same notebook on Kaggle or one allocated Linux NVIDIA GPU node. The first cell sets puzzle index, exact global beam and Transformer/MLP weights (0.8/0.2). `NUM_GPUS=None` uses all GPUs visible to the process; set a number to use that many. Cluster resources must already be allocated by your scheduler. This notebook does not launch jobs on other nodes.

The search uses `graph.beam_search(..., backend="native")` after `enable_native(options)`. `NativeSession.activate()` keeps compatible models, processes and GPU buffers alive when the search cell is repeated. The last cell independently replays any solution, saves evidence and closes the session.

Source, checkpoints and puzzle inputs come from pinned GitHub artifacts. No Kaggle account, dataset attachment or Kaggle filesystem path is required on a cluster. Cache and source files live under `~/.cache/multigpubeamsearch/cube555`; override this with `MULTIGPUBEAMSEARCH_HOME`. Results are saved under `cube555_runs` in the current working directory. Internet, available disk space, a working NVIDIA driver, CUDA-enabled PyTorch, CUDA Toolkit (`nvcc`) and a C++ compiler are required for source builds. Python dependencies install into a private environment inside the cache, including on PEP 668-managed hosts. Existing CUDA PyTorch is reused. The package downloads its pinned CUTLASS sources and selects the backend for the visible GPUs.

Profiles are checked in order: local cache, compatible verified GitHub record, automatic calibration. GitHub proposals receive fresh numerical/timing and memory checks on the current GPUs. Matching includes the exact frontier, model/blend, native source and hardware configuration. The default short component calibration is not presented as a full-step performance measurement. An explicit frontier is never silently reduced if memory admission fails.

Only source-bound completed GPU evidence qualifies as a ready profile; see [the registry](../../profiles/README.md). A missing matching profile triggers calibration. Historical validation of another revision is not acceptance of the current complete Cube555 workload.
