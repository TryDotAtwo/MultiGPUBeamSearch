# syntax=docker/dockerfile:1
# Packaging only: runner-h200 is compiled on the inexpensive remote builder.
ARG BASE_IMAGE=vastai/pytorch@sha256:f74d98bbc821beebb8a98a5aa875f3561b218fbe2fd4bf9f26851e7360c43dcb
FROM ${BASE_IMAGE}
COPY --chmod=0555 runner-h200 /workspace/runner-frozen
COPY build_receipt.json /workspace/results/build_receipt.json
COPY sweep.py archive.py /workspace/
COPY source /workspace/source
COPY prepare_inputs.py /workspace/
ENV PATH=/venv/main/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
RUN /venv/main/bin/python -c "import torch; assert torch.__version__ == '2.11.0+cu128', torch.__version__" \
    && ldd /workspace/runner-frozen | tee /workspace/results/image-ldd.txt \
    && ! grep 'not found' /workspace/results/image-ldd.txt
RUN /venv/main/bin/python /workspace/prepare_inputs.py
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace
# Keep the Vast base image's SSH startup; no compilation during startup.

