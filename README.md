# CayleyBeam100H100

GPU-resident distributed beam search for the Cayley puzzle workload. The current runnable path uses CUDA, PyTorch tensors, a pybind11/C++ extension, CUDA Graphs, and NCCL across multiple GPUs.

## Quick Docker Run

Requirements:

```text
Linux VM or cluster node
NVIDIA driver with CUDA 12.x support
Docker Engine
NVIDIA Container Toolkit
2 visible NVIDIA GPUs for the default command
```

Build:

```bash
git clone <github-repo-url> CayleyBeam100H100
cd CayleyBeam100H100
docker build -t cayley-beam-h100:latest .
```

Run the default 2-GPU `test.csv` beam-search solver:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/output:/workspace/CayleyBeam100H100/output" \
  -e SUBMISSION_PATH=/workspace/CayleyBeam100H100/output/submission.csv \
  cayley-beam-h100:latest
```

Default command:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/solve_testcsv_2gpu.py
```

Default runtime settings:

```text
GLOBAL_BEAM_WIDTH=65536
MAX_DEPTH=100
USE_CUDA_GRAPHS=1
INFERENCE_BACKEND=torchscript_ensemble
INFERENCE_PARALLELISM=1
B_MICRO=32768
K_EXPAND_TILE=32768
SCORE_RING_DEPTH=8
BETA=1.20
HASH_LOAD_FACTOR=0.45
PROBE_LIMIT=256
```

## Smoke Run

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  -e KNOWN_SCRAMBLE=U,R \
  -e MAX_DEPTH=4 \
  cayley-beam-h100:latest
```

## 2xA100

No algorithm changes are required for 2xA100. The Docker image builds CUDA code for both A100 and H100:

```text
TORCH_CUDA_ARCH_LIST="8.0;9.0"
```

For one-node 2xA100 without InfiniBand, use:

```bash
-e NCCL_IB_DISABLE=1
```

For one-node 2xA100 with healthy InfiniBand/RDMA, use:

```bash
-e NCCL_IB_DISABLE=0
```

If NCCL P2P auto-detection behaves better than a fixed policy on the host, omit `NCCL_P2P_LEVEL`.

## Documentation

Detailed deployment notes: [docs/DEPLOY_DOCKER_SLURM.md](docs/DEPLOY_DOCKER_SLURM.md)

Project rules and memory: [AGENTS.md](AGENTS.md), [docs/PROJECT_MEMORY.md](docs/PROJECT_MEMORY.md)
