#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
namespace beam {
void stream1_gnn_projection(const __half* input,const __half* weight,const __half* bias,
                           __half* output,int rows,int inputs,int outputs,cudaStream_t stream);
}
