# Cube555 blend through CayleyPy

[Open the notebook](https://www.kaggle.com/code/trydotatwo/cube555-native-2xt4-blend).

Choose the puzzle and beam, then Run All. On Kaggle select GPU T4 x2 and enable Internet. The folded installation cell runs automatically. The visible example is imports, puzzle/model loading, CayleyPy search and the solution.

```python
import cayleypy as cp
import multigpubeamsearch as beam
from multigpubeamsearch import cube555

puzzle = cube555.load(1020)
graph = cp.CayleyGraph(puzzle.definition, device="cpu", random_seed=42)
predictor = puzzle.blend(graph, transformer_weight=0.8)
```

`cube555.load` downloads and checks the public GitHub inputs internally. `puzzle.blend` prepares the trained Transformer/MLP. Search remains `graph.beam_search(backend="native", ...)`. The result helper replays any solution before saving it. No hashes, manifests or provenance plumbing are needed in user code.

The same notebook works on an allocated Linux GPU node with CUDA PyTorch, nvcc and a C++ compiler. Installation reuses CUDA PyTorch in a private environment. `num_gpus=None` uses all visible GPUs. Profiles and GPU backend are selected automatically; an explicit beam is not silently reduced. Repeat the search cell before the final cell to reuse resident models/workers; the final cell closes the session.

The simplified notebook is a new presentation and Python-helper path; the older v165 full cold T4 run is separate evidence, not a completed test of this revision.
