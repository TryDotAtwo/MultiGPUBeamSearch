ARG BASE_IMAGE=pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel
FROM ${BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV CUDACXX=/usr/local/cuda/bin/nvcc
ENV NVTE_CUDA_INCLUDE_PATH=/usr/local/cuda/include
ENV TORCH_CUDA_ARCH_LIST="8.0;9.0"
ENV NCCL_IB_DISABLE=0
ENV NCCL_ASYNC_ERROR_HANDLING=1
ENV NCCL_DEBUG=WARN
ENV NCCL_NET_GDR_LEVEL=PHB
ENV NCCL_P2P_LEVEL=NVL
ENV OMP_NUM_THREADS=8
ENV MAX_JOBS=8
ENV USE_CUDA_GRAPHS=1

WORKDIR /workspace/CayleyBeam100H100

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pciutils \
    infiniband-diags \
    ibverbs-utils \
    rdma-core \
    openssh-client \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/CayleyBeam100H100

RUN chmod +x scripts/*.sh scripts/*.py

RUN python - <<'PY'
import importlib.util
mods = ["torch"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"missing_modules={missing}; use a PyTorch CUDA devel image")
PY

RUN python setup.py build_ext --inplace

ENTRYPOINT ["/workspace/CayleyBeam100H100/scripts/entrypoint.sh"]
CMD ["bash", "scripts/run_local_2h100.sh"]
