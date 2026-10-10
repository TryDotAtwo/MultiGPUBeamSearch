#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
namespace beam {
void stream1_gnn_projection(const __half* input,const __half* weight,const __half* bias,
                           __half* output,int rows,int inputs,int outputs,cudaStream_t stream);
void stream1_gnn_gat(const __half* left,const __half* right,const __half* edge,
    const __half* attention,const __half* bias,const std::int64_t* indices,
    const std::int64_t* types,int* incoming,__half* output,
    int nodes,int edges,int channels,cudaStream_t stream);
}
