# TheQuantumProject checkpoint scorer

Each worker restores its own 102-input field and constructs the original feature plan on its local CUDA device. The native library borrows contiguous CUDA tensor storage from PyTorch in the same process. Parent calls return raw Qhat costs (ascending) or raw proposal logits (descending) in control-major, off-diagonal order. Child calls return scalar V (ascending). The adapter checks every scored row against the producer's feature coverage and rejects nonfinite scores.

## Build

Supply a local TheQuantumProject Git checkout containing commit `c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb` and a matching training fixture. Use a private output directory: it will contain the exported upstream sources and the linked library. The checkout and checkpoint are read-only inputs. The build uses the exact Git object and its Cargo.lock.

Install CUDA-enabled PyTorch 2.9.0 in the Python environment that will run the adapter. The upstream crate requires tch 0.22.0, a matching CUDA toolkit, a C++17 compiler, Rust/Cargo, and CMake 3.24 or newer. For a CUDA 12.8 Linux build containing T4 and Ampere laptop kernels:

```sh
export CUDA_PATH=/usr/local/cuda
export CUDACXX="$CUDA_PATH/bin/nvcc"
export CNOT103_CUDA_ARCHITECTURES='75;86'
export LIBTORCH_USE_PYTORCH=1
export PYTHON="$(command -v python)"
python integrations/cnot_scorer/build.py \
  --source /private/TheQuantumProject --build-dir /private/mgbs-cnot-build
```

The shared library is `/private/mgbs-cnot-build/target/release/libcnot103_native.so`. Use it with the same CUDA/PyTorch build on every worker. To separate private source preparation from compilation, pass `--prepare-only` and run `cargo build --locked --release --lib --features cuda` in the resulting `native` directory.

## Load and score

The scorer takes a `cnot103_dev_search_v1` manifest, such as `search-qhat.json` from the privately supplied `device-search-gl4-fixture.tar.gz`. For that fixture, keep eight files at their original relative paths: the search manifest, `run.json`, `training-result.json`, `dataset.json`, `dataset.sha256`, `catalogs.json`, `catalogs.sha256`, and the exact `checkpoint/step-*.qcp` named by the training result. The upstream loader verifies the training manifest, result, dataset, registry, objective, exposure audit and checkpoint before scoring. Its frozen dev indices bind the available feature-plan dimensions. Search accepts any admitted matrix of that width.

```python
import torch
from multigpubeamsearch.cnot_scorer import TQPScorer

device = torch.device("cuda", local_rank)
rows = torch.tensor([1, 2, 7, 8], device=device, dtype=torch.int64).reshape(1, 4, 1)
with TQPScorer(library, manifest, n=4, device=device, inference_batch=4) as scorer:
    costs = scorer.parent_scores(rows, "qhat")
    logits = scorer.parent_scores(rows, "proposal")
    values = scorer.child_values(rows)
    checkpoint_sha256 = scorer.metadata["checkpoint_sha256"]
```

The original GL(4,2) one-update fixture has checkpoint SHA-256 `92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4`. Run the optional tests with that fixture:

```sh
export MGBS_TQP_LIBRARY=/private/mgbs-cnot-build/target/release/libcnot103_native.so
export MGBS_TQP_MANIFEST=/private/gl4/search-qhat.json
# Optional numerical comparison against the upstream canonical V predictions.
export MGBS_TQP_H07_REFERENCE=/private/receipts/view-spread-result.json
PYTHONPATH=integrations/cayleypy_native python -m unittest discover \
  -s integrations/cayleypy_native/tests -p test_cnot_scorer.py -v
```
