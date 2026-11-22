# MultiGPUBeamSearch

Native GPU beam search with a CayleyPy API adapter and automatic profile selection.

[Run the simple Kaggle notebook](https://www.kaggle.com/code/trydotatwo/cayleypy-multigpubeamsearch) · [API and installation](integrations/cayleypy_native/README.md)

```python
import cayleypy
import multigpubeamsearch

graph = cayleypy.CayleyGraph(cayleypy.PermutationGroups.lrx(4), device="cuda")
multigpubeamsearch.enable_native(multigpubeamsearch.NativeOptions(num_gpus=2))
result = graph.beam_search(start_state=[3, 2, 1, 0], beam_width=4096,
                           max_steps=16, return_path=True, backend="native")
```

Use `backend="torch"` for CayleyPy search. Native supports Hamming scoring and the documented neural model families/blends. Build dependencies and native sources are included in the package setup; CUDA GPU/driver and toolkit are required. First use builds and calibrates; compatible later calls reuse cached profiles.

For supported neural workloads, `beam_width="max"` reserves the fastest verified inference batch first, then sizes the frontier from remaining GPU memory. Capacity depends on the active GPU memory snapshot; global optimality and 128-GPU hardware validation are not claimed.

[Latest real-model acceptance](test_results/real_neural_autotune_acceptance_20261010.md). Historical branch heads are retained as `archive/20261010/*` tags; the active development branch is `main`.
