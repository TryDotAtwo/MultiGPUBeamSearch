FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        ninja-build \
        ca-certificates \
        git \
        python3 \
        nsight-systems-2025.6.3 \
        nsight-compute-2026.1.1 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/NVIDIA/cutlass.git /opt/cutlass \
    && test -f /opt/cutlass/include/cutlass/gemm/device/gemm.h

RUN ln -s /usr/bin/python3 /usr/local/bin/python \
    && command -v python \
    && command -v cmake \
    && command -v ninja \
    && command -v nsys \
    && command -v ncu

WORKDIR /work

CMD ["bash", "-lc", "cmake -S . -B build-docker -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build-docker && ctest --test-dir build-docker --output-on-failure"]
