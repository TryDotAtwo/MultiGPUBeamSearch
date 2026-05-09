# Yandex 2xA100 Runbook

entity_id=yandex_2xa100_runbook; type=deployment_runbook; state=active

## Registry

```text
registry=cr.yandex/crp7o66ucs8c14sjctp5
image=cr.yandex/crp7o66ucs8c14sjctp5/multigpu-beam-search:a100-kaggle-2t4-baseline
source_repo=https://github.com/TryDotAtwo/MultiGPUBeamSearch
```

## Local Build And Push From Windows

Auth option A, with Yandex CLI:

```powershell
yc container registry configure-docker
.\scripts\push_yandex_container.ps1
```

Auth option B, without Yandex CLI:

```powershell
docker login --username oauth --password <YANDEX_OAUTH_TOKEN> cr.yandex
.\scripts\push_yandex_container.ps1
```

Build-only retry:

```powershell
docker build -t cayley-beam-h100:latest .
```

Push-only after successful local build:

```powershell
.\scripts\push_yandex_container.ps1 -SkipBuild
```

## SSH Key Preparation

```powershell
Expand-Archive .\ssh-key-1778315981146.zip -DestinationPath .\.ssh -Force
icacls .\.ssh\ssh-key-1778315981146 /inheritance:r
icacls .\.ssh\ssh-key-1778315981146 /grant:r "$env:USERNAME:R"
```

SSH format:

```powershell
ssh -i .\.ssh\ssh-key-1778315981146 trydotatwo@<VM_PUBLIC_IP>
```

## Remote Prerequisite Checks

```bash
hostname
uname -a
nvidia-smi
nvidia-smi topo -m
docker version
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

## InfiniBand And RDMA Checks

```bash
lsmod | egrep 'mlx5|ib_uverbs|rdma' || true
ibv_devinfo || true
ibstat || true
rdma link || true
ip link
```

Interpretation:

```text
IB/RDMA available: mlx5 device present, ibv_devinfo shows active port, rdma link shows active link.
IB/RDMA unavailable: use NCCL_IB_DISABLE=1 and rely on PCIe/NVLink/P2P/socket path.
```

## Pull Image On VM

```bash
docker login --username oauth --password <YANDEX_OAUTH_TOKEN> cr.yandex
docker pull cr.yandex/crp7o66ucs8c14sjctp5/multigpu-beam-search:a100-kaggle-2t4-baseline
```

## Full Kaggle-Equivalent 2-GPU Beam

This is not a smoke test. This matches the latest successful 2xT4 `test.csv` beam configuration.

```bash
mkdir -p output
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --network=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD/output:/workspace/CayleyBeam100H100/output" \
  -e SUBMISSION_PATH=/workspace/CayleyBeam100H100/output/submission.csv \
  -e WORLD_SIZE=2 \
  -e NPROC_PER_NODE=2 \
  -e GLOBAL_BEAM_WIDTH=65536 \
  -e MAX_DEPTH=100 \
  -e USE_CUDA_GRAPHS=1 \
  -e INFERENCE_BACKEND=torchscript_ensemble \
  -e INFERENCE_PARALLELISM=1 \
  -e B_MICRO=32768 \
  -e K_EXPAND_TILE=32768 \
  -e SCORE_RING_DEPTH=8 \
  -e NET_RING_DEPTH=2 \
  -e BUCKET_CAP_PER_PEER=65536 \
  -e BETA=1.20 \
  -e HASH_LOAD_FACTOR=0.45 \
  -e PROBE_LIMIT=256 \
  -e LOG_EVERY=25 \
  -e NCCL_DEBUG=INFO \
  -e NCCL_IB_DISABLE=0 \
  cr.yandex/crp7o66ucs8c14sjctp5/multigpu-beam-search:a100-kaggle-2t4-baseline
```

Fallback when IB fails:

```bash
NCCL_IB_DISABLE=1
```

## A100 Tuning Matrix

First pass keeps algorithm unchanged and only changes static runtime parameters.

```text
baseline: GLOBAL_BEAM_WIDTH=65536; B_MICRO=32768; K_EXPAND_TILE=32768; SCORE_RING_DEPTH=8; INFERENCE_PARALLELISM=1
candidate_1: GLOBAL_BEAM_WIDTH=131072; B_MICRO=32768; K_EXPAND_TILE=32768; SCORE_RING_DEPTH=8; INFERENCE_PARALLELISM=1
candidate_2: GLOBAL_BEAM_WIDTH=262144; B_MICRO=65536; K_EXPAND_TILE=32768; SCORE_RING_DEPTH=8; INFERENCE_PARALLELISM=1
candidate_3: GLOBAL_BEAM_WIDTH=524288; B_MICRO=65536; K_EXPAND_TILE=65536; SCORE_RING_DEPTH=8; INFERENCE_PARALLELISM=1
candidate_4: GLOBAL_BEAM_WIDTH=1048576; B_MICRO=65536; K_EXPAND_TILE=65536; SCORE_RING_DEPTH=8; INFERENCE_PARALLELISM=1
```

Required pass criteria:

```text
bucket_overflow=0
hash_overflow=0
cuda_graph_captured_sum=2
submission.csv written
NCCL INFO confirms selected transport
```
