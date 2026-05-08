# Docker and SLURM deployment

## Base container

`Dockerfile` uses `nvcr.io/nvidia/pytorch:25.04-py3`. NVIDIA Transformer Engine documentation says Transformer Engine is preinstalled in NVIDIA PyTorch containers `22.09+`; recent Transformer Engine releases require Linux x86_64, CUDA `12.1+`, a matching NVIDIA driver, and cuDNN `9.3+`.

Source references:
- [Transformer Engine installation](https://docs.nvidia.com/deeplearning/transformer-engine-releases/release-2.14/user-guide/installation.html)
- [NGC containers overview](https://www.nvidia.com/en-gb/gpu-cloud/containers)

## 2xH100 debug configuration

```text
WORLD_SIZE=2
GLOBAL_BEAM_WIDTH=16777216
B_MICRO=131072
FANOUT=24
SCORE_RING_DEPTH=64
NET_RING_DEPTH=3
BUCKET_CAP_PER_PEER=3145728
USE_CUDA_GRAPHS=1
NCCL_P2P_LEVEL=NVL
NCCL_IB_DISABLE=0
```

Expected static GPU buffer memory per rank is about `8.80 GiB`, excluding TE weights, CUDA context, NCCL internals, and allocator fragmentation. This leaves about `71.20 GiB` H100 memory headroom for debugging, logging, and TE integration.

## 100xH100 production configuration

```text
WORLD_SIZE=100
GLOBAL_BEAM_WIDTH=1073741824
B_MICRO=131072
FANOUT=24
SCORE_RING_DEPTH=64
NET_RING_DEPTH=3
BUCKET_CAP_PER_PEER=65536
USE_CUDA_GRAPHS=1
NCCL_P2P_LEVEL=NVL
NCCL_NET_GDR_LEVEL=PHB
NCCL_IB_DISABLE=0
```

Expected static GPU buffer memory per rank is about `9.82 GiB`, excluding TE weights, CUDA context, NCCL internals, and allocator fragmentation. Fixed-size all-to-all sends `65536 * 160 = 10 MiB` per peer per network slot; with 100 ranks and 3 slots, send plus receive buffers reserve about `5.86 GiB` per rank.

## Build

```bash
docker build -t cayley-beam-h100:latest .
```

## SSH quickstart on one VM with 2 GPUs

Prerequisites on the VM:

```text
NVIDIA driver supports CUDA 12.x
Docker Engine is installed
NVIDIA Container Toolkit is installed
docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi works
```

Clone, build, and run:

```bash
git clone <github-repo-url> CayleyBeam100H100
cd CayleyBeam100H100
docker build -t cayley-beam-h100:latest .
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

Default container command runs:

```text
torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/solve_testcsv_2gpu.py
```

Default search settings mirror the latest successful Kaggle `test.csv` run:

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

For a short smoke run over the first two `test.csv` rows:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  -v "$PWD/output:/workspace/CayleyBeam100H100/output" \
  -e TEST_COUNT=2 \
  -e MAX_DEPTH=10 \
  -e SUBMISSION_PATH=/workspace/CayleyBeam100H100/output/smoke.csv \
  cayley-beam-h100:latest
```

For a known two-move correctness probe:

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  -e KNOWN_SCRAMBLE=U,R \
  -e MAX_DEPTH=4 \
  cayley-beam-h100:latest
```

## 2xA100 notes

No algorithm changes are required for 2xA100. Required deployment changes are configuration and build-target changes only:

```text
TORCH_CUDA_ARCH_LIST must include 8.0 for A100 sm80.
Dockerfile now uses TORCH_CUDA_ARCH_LIST="8.0;9.0".
NCCL_IB_DISABLE=0 should be used when InfiniBand/RDMA works.
NCCL_P2P_LEVEL=NVL is H100/NVLink-oriented; for A100 PCIe hosts use NCCL_P2P_LEVEL=PXB or omit the variable if NCCL auto-detection is better.
For one-node 2xA100 without IB, set NCCL_IB_DISABLE=1.
```

Memory: A100 80GB should have enough memory for the default `GLOBAL_BEAM_WIDTH=65536` run and much larger debug beams. A100 40GB should also have enough memory for the default run. Before increasing beam width, run:

```bash
docker run --rm -it --gpus all cayley-beam-h100:latest \
  python scripts/h100_sizing.py --world-size 2 --global-beam-width 16777216 --bucket-cap-per-peer 3145728 --max-depth 100 --beta 1.20 --hash-load-factor 0.45
```

## Run on one VM with 2xH100

```bash
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e WORLD_SIZE=2 \
  -e GLOBAL_BEAM_WIDTH=65536 \
  cayley-beam-h100:latest \
  bash scripts/run_local_2h100.sh
```

Alternative:

```bash
docker compose -f docker-compose.2h100.yml up --build
```

## Run with SLURM on 2xH100

```bash
sbatch scripts/run_slurm_2h100.sh
```

## Run with SLURM on 100xH100

`scripts/run_slurm_100h100.sh` reserves 13 nodes with 8 GPUs per node, then runs 100 tasks through `srun`. Each task receives `RANK=$SLURM_PROCID`, `WORLD_SIZE=100`, and `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`. The last 4 GPUs remain unused unless cluster policy requires exact node shapes.

```bash
sbatch scripts/run_slurm_100h100.sh
```

## Container on SLURM

If cluster uses Pyxis/Enroot:

```bash
srun --container-image=cayley-beam-h100:latest --container-mounts="$PWD:/workspace/CayleyBeam100H100" bash scripts/run_slurm_2h100.sh
```

If cluster uses Apptainer:

```bash
apptainer exec --nv docker-daemon://cayley-beam-h100:latest bash scripts/run_slurm_2h100.sh
```

## Validation commands

```bash
python - <<'PY'
import torch
import transformer_engine
print("cuda_available", torch.cuda.is_available())
print("gpu_count", torch.cuda.device_count())
print("te_import_ok", transformer_engine is not None)
PY
```

```bash
python scripts/h100_sizing.py --world-size 2 --global-beam-width 16777216 --bucket-cap-per-peer 3145728
python scripts/h100_sizing.py --world-size 100 --global-beam-width 1073741824 --bucket-cap-per-peer 65536
```

## Current limitations

`INFERENCE_BACKEND=te` still reaches `TEInferenceBackend::forward()` stub until real Transformer Engine C++ forward, weights, and tensor layout are implemented. Use `INFERENCE_BACKEND=dummy` for container, NCCL, CUDA Graph, and memory validation.

`apply_move` uses table-driven CUDA constant memory only after `BeamEngine.set_action_permutation_table(table_bytes)` receives canonical `24*120` permutation bytes.
