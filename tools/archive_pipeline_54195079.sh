#!/usr/bin/env bash
set -eu
cd /workspace/pipeline_audit_54195079
# Preserve source and measured evidence; exclude preallocated history arenas.
tar --exclude='./build' --exclude='./build-before' --exclude='./evidence' \
    --exclude='./.git' --exclude='__pycache__' --exclude='./.pytest_cache' \
    -czf /workspace/source_pipeline_final_54195079.tar.gz .
tar --exclude='*history_static_arena.bin' --exclude='__pycache__' \
    -czf /workspace/evidence_pipeline_final_54195079.tar.gz \
    evidence build/production_runner build/stream4_cuda_tests \
    build/final_union_dispatcher_cuda_tests build/CMakeCache.txt \
    build-before/production_runner
cd /workspace
find . -maxdepth 1 -type f -name '*54195079.log' -print0 | \
    tar --null -T - -czf /workspace/logs_pipeline_final_54195079.tar.gz
sha256sum /workspace/source_pipeline_final_54195079.tar.gz \
    /workspace/evidence_pipeline_final_54195079.tar.gz \
    /workspace/logs_pipeline_final_54195079.tar.gz
