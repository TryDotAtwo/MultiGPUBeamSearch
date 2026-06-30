#include "stream1.hpp"

#include "config.hpp"
#include "cuda_check.hpp"
#include "nvtx_ranges.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#if BEAM_HAS_CUTLASS
#include <cutlass/bfloat16.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_array.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/layout/matrix.h>
#endif

#include <stdexcept>

namespace beam {

__global__ void stream1_score_contract_kernel(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    std::uint32_t* score_ring,
    std::uint32_t ring,
    std::uint32_t ring_slot,
    std::uint32_t ring_slot_count,
    std::uint32_t b_micro) {
    const std::uint32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (candidate >= total) {
        return;
    }
    const std::uint32_t parent_local = candidate / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint32_t move = candidate % static_cast<std::uint32_t>(MOVE_COUNT);
    if (parent_local >= count[ring * ring_slot_count + ring_slot]) {
        return;
    }

    const std::uint64_t parent_idx = parent_base[ring * ring_slot_count + ring_slot] + parent_local;
    const State128 state = current_frontier_states[parent_idx];
    std::uint32_t accum = 0;
    for (std::uint32_t p = 0; p < STATE_LEN; ++p) {
        accum += static_cast<std::uint32_t>(state.v[p]) * static_cast<std::uint32_t>((p + 1U) * (move + 1U));
    }
    const std::uint32_t score_key = accum % (SCORE_MAX_KEY + 1U);
    const std::uint64_t offset =
        (((static_cast<std::uint64_t>(ring) * ring_slot_count + ring_slot) * b_micro + parent_local) * MOVE_COUNT) + move;
    score_ring[offset] = score_key;
}

void stream1_score_contract_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    std::uint32_t* score_ring,
    std::uint32_t ring,
    std::uint32_t ring_slot,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    NvtxRange range("Stream1_contract_score_launch");
    const std::uint32_t ring_slot_count = 1;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    const dim3 block(128);
    const dim3 grid((total + block.x - 1) / block.x);
    stream1_score_contract_kernel<<<grid, block, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        score_ring,
        ring,
        ring_slot,
        ring_slot_count,
        b_micro);
}

__device__ float relu_device(float x) {
    return x > 0.0f ? x : 0.0f;
}
__device__ float stream1_warp_reduce_sum_device(float value) {
    constexpr unsigned mask = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return __shfl_sync(mask, value, 0);
}

__device__ float stream1_warp_reduce_max_device(float value) {
    constexpr unsigned mask = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return __shfl_sync(mask, value, 0);
}

__device__ std::uint32_t score_key_from_float_device(float q) {
    q = fminf(fmaxf(q, 0.0f), SCORE_MAX_Q);
    return static_cast<std::uint32_t>(rintf(q * static_cast<float>(SCORE_SCALE)));
}

__device__ float stream1_load_scalar_device(const half* ptr, std::uint64_t idx, std::uint32_t dtype) {
    if (dtype == STREAM1_DTYPE_BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
    }
    return __half2float(ptr[idx]);
}

__device__ void stream1_store_scalar_device(half* ptr, std::uint64_t idx, float value, std::uint32_t dtype) {
    if (dtype == STREAM1_DTYPE_BF16) {
        reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16(value);
        return;
    }
    ptr[idx] = __float2half(value);
}

__global__ void stream1_inference_custom_kernel(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    Stream1NetworkView network,
    std::uint32_t* score_ring,
    std::uint32_t ring,
    std::uint32_t ring_slot,
    std::uint32_t b_micro) {
    const std::uint32_t parent_local = blockIdx.x;
    const std::uint32_t move = threadIdx.x;
    if (parent_local >= b_micro || parent_local >= *count || move >= MOVE_COUNT) {
        return;
    }
    extern __shared__ float scratch[];
    float* hidden1 = scratch;
    float* hidden2 = scratch + network.dims.hidden1;

    const State128 state = current_frontier_states[*parent_base + parent_local];
    for (std::uint32_t h = move; h < network.dims.hidden1; h += MOVE_COUNT) {
        float acc = __half2float(network.input_bias[h]);
        for (std::uint32_t p = 0; p < network.dims.state_len; ++p) {
            const std::uint32_t value = static_cast<std::uint32_t>(state.v[p]);
            const std::uint32_t idx = (p * network.dims.num_classes + value) * network.dims.hidden1 + h;
            acc += __half2float(network.input_weight[idx]);
        }
        hidden1[h] = relu_device(acc);
    }
    __syncthreads();

    for (std::uint32_t h = move; h < network.dims.hidden2; h += MOVE_COUNT) {
        float acc = __half2float(network.hidden_bias[h]);
        for (std::uint32_t k = 0; k < network.dims.hidden1; ++k) {
            const std::uint32_t idx = k * network.dims.hidden2 + h;
            acc += hidden1[k] * __half2float(network.hidden_weight[idx]);
        }
        hidden2[h] = relu_device(acc);
    }
    __syncthreads();

    float q = __half2float(network.output_bias[move]);
    for (std::uint32_t h = 0; h < network.dims.hidden2; ++h) {
        q += hidden2[h] * __half2float(network.output_weight[h * MOVE_COUNT + move]);
    }
    score_ring[parent_local * MOVE_COUNT + move] = score_key_from_float_device(q);
}

__global__ void stream1_folded_input_half4_tiled_kernel(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    const std::uint32_t* __restrict__ count,
    const std::uint8_t* __restrict__ generators,
    const half* __restrict__ input_weight,
    const half* __restrict__ input_bias,
    half* __restrict__ hidden1,
    Stream1NetworkDims dims,
    std::uint32_t b_micro) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t h = (blockIdx.y * blockDim.x + threadIdx.x) * 4U;
    const std::uint32_t count_value = *count;
    const bool child_rows = dims.output_dim == STREAM1_SINGLE_SCORE_OUTPUT_DIM;
    const std::uint32_t parent_local = child_rows ? row / static_cast<std::uint32_t>(MOVE_COUNT) : row;
    const std::uint32_t move = child_rows ? row % static_cast<std::uint32_t>(MOVE_COUNT) : 0U;
    const std::uint32_t row_count = b_micro * (child_rows ? static_cast<std::uint32_t>(MOVE_COUNT) : 1U);
    const bool row_valid = row < row_count && parent_local < count_value;
    __shared__ std::uint8_t state_shared[STATE_STORAGE_LEN];
    if (row_valid && threadIdx.x < STATE_STORAGE_LEN) {
        const std::uint64_t parent_idx = *parent_base + parent_local;
        state_shared[threadIdx.x] = current_frontier_states[parent_idx].v[threadIdx.x];
    }
    __syncthreads();
    if (!row_valid || h >= dims.hidden1) {
        return;
    }

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    float acc2 = 0.0f;
    float acc3 = 0.0f;
    const bool has01 = (h + 1U) < dims.hidden1;
    const bool has2 = (h + 2U) < dims.hidden1;
    const bool has23 = (h + 3U) < dims.hidden1;
    const bool fp16_folded_fast_path =
        dims.dtype == STREAM1_DTYPE_FP16 && dims.normalization != STREAM1_NORM_LAYERNORM;
    if (fp16_folded_fast_path) {
        if (has01) {
            const float2 bias = __half22float2(*reinterpret_cast<const half2*>(input_bias + h));
            acc0 = bias.x;
            acc1 = bias.y;
        } else {
            acc0 = __half2float(input_bias[h]);
        }
        if (has23) {
            const float2 bias = __half22float2(*reinterpret_cast<const half2*>(input_bias + h + 2U));
            acc2 = bias.x;
            acc3 = bias.y;
        } else if (has2) {
            acc2 = __half2float(input_bias[h + 2U]);
        }
    } else if (dims.normalization != STREAM1_NORM_LAYERNORM) {
        acc0 = stream1_load_scalar_device(input_bias, h, dims.dtype);
        if (has01) {
            acc1 = stream1_load_scalar_device(input_bias, h + 1U, dims.dtype);
        }
        if (has2) {
            acc2 = stream1_load_scalar_device(input_bias, h + 2U, dims.dtype);
        }
        if (has23) {
            acc3 = stream1_load_scalar_device(input_bias, h + 3U, dims.dtype);
        }
    }

    for (std::uint32_t p = 0; p < dims.state_len; ++p) {
        const std::uint32_t source =
            child_rows
                ? static_cast<std::uint32_t>(generators[move * STATE_STORAGE_LEN + p])
                : p;
        const std::uint32_t value = static_cast<std::uint32_t>(state_shared[source]);
        const std::uint32_t idx = (p * dims.num_classes + value) * dims.hidden1 + h;
        if (fp16_folded_fast_path) {
            if (has01) {
                const float2 w = __half22float2(*reinterpret_cast<const half2*>(input_weight + idx));
                acc0 += w.x;
                acc1 += w.y;
            } else {
                acc0 += __half2float(input_weight[idx]);
            }
            if (has23) {
                const float2 w = __half22float2(*reinterpret_cast<const half2*>(input_weight + idx + 2U));
                acc2 += w.x;
                acc3 += w.y;
            } else if (has2) {
                acc2 += __half2float(input_weight[idx + 2U]);
            }
        } else {
            acc0 += stream1_load_scalar_device(input_weight, idx, dims.dtype);
            if (has01) {
                acc1 += stream1_load_scalar_device(input_weight, idx + 1U, dims.dtype);
            }
            if (has2) {
                acc2 += stream1_load_scalar_device(input_weight, idx + 2U, dims.dtype);
            }
            if (has23) {
                acc3 += stream1_load_scalar_device(input_weight, idx + 3U, dims.dtype);
            }
        }
    }

    half* out = hidden1 + row * dims.hidden1 + h;
    if (dims.normalization != STREAM1_NORM_LAYERNORM) {
        acc0 = relu_device(acc0);
    }
    if (has01) {
        if (dims.normalization != STREAM1_NORM_LAYERNORM) {
            acc1 = relu_device(acc1);
        }
        if (dims.dtype == STREAM1_DTYPE_BF16) {
            stream1_store_scalar_device(out, 0, acc0, dims.dtype);
            stream1_store_scalar_device(out, 1, acc1, dims.dtype);
        } else {
            *reinterpret_cast<half2*>(out) = __floats2half2_rn(acc0, acc1);
        }
    } else {
        stream1_store_scalar_device(out, 0, acc0, dims.dtype);
    }
    if (has23) {
        if (dims.normalization != STREAM1_NORM_LAYERNORM) {
            acc2 = relu_device(acc2);
            acc3 = relu_device(acc3);
        }
        if (dims.dtype == STREAM1_DTYPE_BF16) {
            stream1_store_scalar_device(out, 2, acc2, dims.dtype);
            stream1_store_scalar_device(out, 3, acc3, dims.dtype);
        } else {
            *reinterpret_cast<half2*>(out + 2U) = __floats2half2_rn(acc2, acc3);
        }
    } else if (has2) {
        if (dims.normalization != STREAM1_NORM_LAYERNORM) {
            acc2 = relu_device(acc2);
        }
        stream1_store_scalar_device(out, 2, acc2, dims.dtype);
    }
}

__global__ void stream1_bias_relu_kernel(
    half* matrix,
    const half* bias,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * cols;
    if (i >= total) {
        return;
    }
    const std::uint32_t col = i % cols;
    const float x = stream1_load_scalar_device(matrix, i, dtype) +
                    stream1_load_scalar_device(bias, col, dtype);
    stream1_store_scalar_device(matrix, i, relu_device(x), dtype);
}

__global__ void stream1_residual_add_bias_relu_kernel(
    half* matrix,
    const half* residual,
    const half* bias,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * cols;
    if (i >= total) {
        return;
    }
    const std::uint32_t col = i % cols;
    const float x = stream1_load_scalar_device(matrix, i, dtype) +
                    stream1_load_scalar_device(residual, i, dtype) +
                    stream1_load_scalar_device(bias, col, dtype);
    stream1_store_scalar_device(matrix, i, relu_device(x), dtype);
}

__global__ void stream1_layernorm_relu_kernel(
    half* matrix,
    const half* bias,
    const half* gamma,
    const half* beta,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float reduce[];
    const std::uint64_t base = static_cast<std::uint64_t>(row) * cols;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += stream1_load_scalar_device(matrix, base + col, dtype) +
               stream1_load_scalar_device(bias, col, dtype);
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = reduce[0] / static_cast<float>(cols);
    float var_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = stream1_load_scalar_device(matrix, base + col, dtype) +
                        stream1_load_scalar_device(bias, col, dtype);
        const float d = x - mean;
        var_sum += d * d;
    }
    reduce[threadIdx.x] = var_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(reduce[0] / static_cast<float>(cols) + 1.0e-5f);
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = stream1_load_scalar_device(matrix, base + col, dtype) +
                        stream1_load_scalar_device(bias, col, dtype);
        const float y = (x - mean) * inv_std *
                        stream1_load_scalar_device(gamma, col, dtype) +
                        stream1_load_scalar_device(beta, col, dtype);
        stream1_store_scalar_device(matrix, base + col, relu_device(y), dtype);
    }
}

__global__ void stream1_layernorm_residual_relu_kernel(
    half* matrix,
    const half* residual,
    const half* bias,
    const half* gamma,
    const half* beta,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float reduce[];
    const std::uint64_t base = static_cast<std::uint64_t>(row) * cols;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += stream1_load_scalar_device(matrix, base + col, dtype) +
               stream1_load_scalar_device(bias, col, dtype);
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = reduce[0] / static_cast<float>(cols);
    float var_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = stream1_load_scalar_device(matrix, base + col, dtype) +
                        stream1_load_scalar_device(bias, col, dtype);
        const float d = x - mean;
        var_sum += d * d;
    }
    reduce[threadIdx.x] = var_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(reduce[0] / static_cast<float>(cols) + 1.0e-5f);
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = stream1_load_scalar_device(matrix, base + col, dtype) +
                        stream1_load_scalar_device(bias, col, dtype);
        const float y = (x - mean) * inv_std *
                        stream1_load_scalar_device(gamma, col, dtype) +
                        stream1_load_scalar_device(beta, col, dtype) +
                        stream1_load_scalar_device(residual, base + col, dtype);
        stream1_store_scalar_device(matrix, base + col, relu_device(y), dtype);
    }
}

__global__ void stream1_score_epilogue_quantize_kernel(
    const half* output,
    const half* output_bias,
    const std::uint32_t* count,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    std::uint32_t dtype) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (i >= total) {
        return;
    }
    const std::uint32_t row = i / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint32_t move = i % static_cast<std::uint32_t>(MOVE_COUNT);
    if (row >= *count) {
        return;
    }
    const float q = stream1_load_scalar_device(output, i, dtype) +
                    stream1_load_scalar_device(output_bias, move, dtype);
    score_ring[i] = score_key_from_float_device(q);
}

__global__ void stream1_single_output_quantize_kernel(
    const half* residual,
    const half* output_weight,
    const half* output_bias,
    const std::uint32_t* count,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    std::uint32_t hidden2,
    std::uint32_t dtype) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (i >= total) {
        return;
    }
    const std::uint32_t parent_local = i / static_cast<std::uint32_t>(MOVE_COUNT);
    if (parent_local >= *count) {
        return;
    }
    float q = stream1_load_scalar_device(output_bias, 0, dtype);
    const half* row = residual + static_cast<std::uint64_t>(i) * hidden2;
    for (std::uint32_t h = 0; h < hidden2; ++h) {
        q += stream1_load_scalar_device(row, h, dtype) *
             stream1_load_scalar_device(output_weight, h, dtype);
    }
    q += STREAM1_SINGLE_OUTPUT_SCORE_OFFSET;
    score_ring[i] = score_key_from_float_device(q);
}

__global__ void stream1_transformer_build_input_kernel(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    const std::uint32_t* __restrict__ count,
    Stream1TransformerNetworkView network,
    half* __restrict__ tokens,
    std::uint32_t b_micro) {
    const std::uint32_t row_token = blockIdx.x;
    const std::uint32_t dim = blockIdx.y * blockDim.x + threadIdx.x;
    const std::uint32_t row = row_token / network.dims.seq_len;
    const std::uint32_t token = row_token % network.dims.seq_len;
    const std::uint32_t active_count = *count;
    if (row >= b_micro || row >= active_count || dim >= network.dims.d_model) {
        return;
    }
    float value = 0.0f;
    if (token == 0U) {
        value = stream1_load_scalar_device(network.cls_token, dim, network.dims.dtype);
    } else {
        const std::uint32_t piece = token - 1U;
        value = stream1_load_scalar_device(
            network.fast_piece_static,
            static_cast<std::uint64_t>(piece) * network.dims.d_model + dim,
            network.dims.dtype);
        const std::uint64_t parent_idx = *parent_base + static_cast<std::uint64_t>(row);
        const State128* state = current_frontier_states + parent_idx;
        for (std::uint32_t slot = 0; slot < network.dims.max_piece_size; ++slot) {
            const std::uint64_t piece_slot = static_cast<std::uint64_t>(piece) * network.dims.max_piece_size + slot;
            if (network.piece_mask[piece_slot] == 0U) {
                continue;
            }
            const std::uint32_t pos = static_cast<std::uint32_t>(network.piece_positions[piece_slot]);
            const std::uint32_t state_value = static_cast<std::uint32_t>(state->v[pos]);
            const std::uint64_t table_idx =
                ((static_cast<std::uint64_t>(slot) * network.dims.num_classes + state_value) * network.dims.d_model) + dim;
            value += stream1_load_scalar_device(network.fast_slot_projected, table_idx, network.dims.dtype);
        }
    }
    const std::uint64_t out_idx = static_cast<std::uint64_t>(row_token) * network.dims.d_model + dim;
    stream1_store_scalar_device(tokens, out_idx, value, network.dims.dtype);
}

__global__ void stream1_transformer_layernorm_copy_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype) {
    const std::uint32_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float reduce[];
    const std::uint64_t base = static_cast<std::uint64_t>(row) * cols;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += stream1_load_scalar_device(input, base + col, dtype);
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = reduce[0] / static_cast<float>(cols);
    float var_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = stream1_load_scalar_device(input, base + col, dtype);
        const float d = x - mean;
        var_sum += d * d;
    }
    reduce[threadIdx.x] = var_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(reduce[0] / static_cast<float>(cols) + 1.0e-5f);
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = stream1_load_scalar_device(input, base + col, dtype);
        const float y = (x - mean) * inv_std *
            stream1_load_scalar_device(gamma, col, dtype) +
            stream1_load_scalar_device(beta, col, dtype);
        stream1_store_scalar_device(output, base + col, y, dtype);
    }
}

__global__ void stream1_transformer_layernorm256_copy_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint32_t rows,
    std::uint32_t dtype) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows || tid >= 256U) {
        return;
    }

    extern __shared__ float warp_scratch[];
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t idx = static_cast<std::uint64_t>(row) * 256ULL + tid;
    const float x = stream1_load_scalar_device(input, idx, dtype);

    const float warp_sum = stream1_warp_reduce_sum_device(x);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = tid < 8U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_sum = stream1_warp_reduce_sum_device(block_sum);
        if (lane == 0U) {
            warp_scratch[0] = block_sum * (1.0f / 256.0f);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[0];

    const float centered = x - mean;
    const float warp_var_sum = stream1_warp_reduce_sum_device(centered * centered);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();

    float block_var_sum = tid < 8U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_var_sum = stream1_warp_reduce_sum_device(block_var_sum);
        if (lane == 0U) {
            warp_scratch[0] = rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[0];

    const float y = centered * inv_std *
        stream1_load_scalar_device(gamma, tid, dtype) +
        stream1_load_scalar_device(beta, tid, dtype);
    stream1_store_scalar_device(output, idx, y, dtype);
}

void stream1_transformer_layernorm_copy_launch(
    const half* input,
    half* output,
    const half* gamma,
    const half* beta,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    if (cols == 256U) {
        stream1_transformer_layernorm256_copy_kernel<<<rows, 256, 8 * sizeof(float), stream>>>(
            input,
            output,
            gamma,
            beta,
            rows,
            dtype);
        return;
    }
    stream1_transformer_layernorm_copy_kernel<<<rows, 256, 256 * sizeof(float), stream>>>(
        input,
        output,
        gamma,
        beta,
        rows,
        cols,
        dtype);
}

__global__ void stream1_transformer_residual_bias_add_kernel(
    half* __restrict__ tokens,
    const half* __restrict__ branch,
    const half* __restrict__ bias,
    std::uint32_t total_tokens,
    std::uint32_t d_model,
    std::uint32_t dtype) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = total_tokens * d_model;
    if (i >= total) {
        return;
    }
    const std::uint32_t col = i % d_model;
    const float y = stream1_load_scalar_device(tokens, i, dtype) +
        stream1_load_scalar_device(branch, i, dtype) +
        stream1_load_scalar_device(bias, col, dtype);
    stream1_store_scalar_device(tokens, i, y, dtype);
}

__global__ void stream1_transformer_bias_silu_kernel(
    half* __restrict__ matrix,
    const half* __restrict__ bias,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * cols;
    if (i >= total) {
        return;
    }
    const std::uint32_t col = i % cols;
    const float x = stream1_load_scalar_device(matrix, i, dtype) +
        stream1_load_scalar_device(bias, col, dtype);
    const float y = x / (1.0f + expf(-x));
    stream1_store_scalar_device(matrix, i, y, dtype);
}

constexpr std::uint32_t STREAM1_TRANSFORMER_SEQ51 = 51U;
constexpr std::uint32_t STREAM1_TRANSFORMER_DMODEL256 = 256U;
constexpr std::uint32_t STREAM1_TRANSFORMER_HEAD_DIM32 = 32U;
constexpr std::uint32_t STREAM1_TRANSFORMER_NHEAD8 = 8U;
constexpr std::uint32_t STREAM1_TRANSFORMER_QKV_STRIDE51 = 3U * STREAM1_TRANSFORMER_DMODEL256;
constexpr std::uint32_t STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51 = 64U;
constexpr std::uint32_t STREAM1_TRANSFORMER_PROB_STRIDE51 = STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51;
constexpr std::uint32_t STREAM1_TRANSFORMER_VPACK_STRIDE51 = STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51 * STREAM1_TRANSFORMER_HEAD_DIM32;
constexpr std::uint32_t STREAM1_TRANSFORMER_SCORE_STRIDE51 = STREAM1_TRANSFORMER_PROB_STRIDE51 + STREAM1_TRANSFORMER_VPACK_STRIDE51;
constexpr int STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX = 32768;

__global__ void stream1_transformer_add_qkv_bias_kernel(
    half* __restrict__ qkv,
    const half* __restrict__ qkv_bias,
    std::uint32_t dtype,
    std::uint32_t b_micro) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(b_micro) * STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_QKV_STRIDE51;
    if (i >= total) {
        return;
    }
    const std::uint32_t col = static_cast<std::uint32_t>(i % STREAM1_TRANSFORMER_QKV_STRIDE51);
    const float x = stream1_load_scalar_device(qkv, i, dtype) + stream1_load_scalar_device(qkv_bias, col, dtype);
    stream1_store_scalar_device(qkv, i, x, dtype);
}

__global__ void stream1_transformer_softmax51_kernel(
    half* __restrict__ scores_probs,
    std::uint32_t dtype,
    std::uint32_t b_micro) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t head = blockIdx.y;
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint32_t warp_count = blockDim.x >> 5U;
    if (row >= b_micro || head >= STREAM1_TRANSFORMER_NHEAD8) {
        return;
    }
    const std::uint64_t matrix_base =
        (static_cast<std::uint64_t>(head) * b_micro + row) * STREAM1_TRANSFORMER_SCORE_STRIDE51;
    for (std::uint32_t query = warp; query < STREAM1_TRANSFORMER_SEQ51; query += warp_count) {
        const std::uint64_t query_base = matrix_base + static_cast<std::uint64_t>(query) * STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51;
        float local_max = -3.4028234663852886e38f;
        for (std::uint32_t key = lane; key < STREAM1_TRANSFORMER_SEQ51; key += 32U) {
            local_max = fmaxf(local_max, stream1_load_scalar_device(scores_probs, query_base + key, dtype));
        }
        const float max_score = stream1_warp_reduce_max_device(local_max);
        float local_sum = 0.0f;
        float local_values[2] = {0.0f, 0.0f};
        std::uint32_t local_keys[2] = {0xffffffffU, 0xffffffffU};
        std::uint32_t slot = 0U;
        for (std::uint32_t key = lane; key < STREAM1_TRANSFORMER_SEQ51; key += 32U) {
            const float exp_value = expf(stream1_load_scalar_device(scores_probs, query_base + key, dtype) - max_score);
            local_sum += exp_value;
            local_values[slot] = exp_value;
            local_keys[slot] = key;
            ++slot;
        }
        const float inv_sum = 1.0f / stream1_warp_reduce_sum_device(local_sum);
        for (std::uint32_t i = 0; i < slot; ++i) {
            stream1_store_scalar_device(scores_probs, query_base + local_keys[i], local_values[i] * inv_sum, dtype);
        }
        for (std::uint32_t key = STREAM1_TRANSFORMER_SEQ51 + lane; key < STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51; key += 32U) {
            stream1_store_scalar_device(scores_probs, query_base + key, 0.0f, dtype);
        }
    }
}

__global__ void stream1_transformer_pack_v51_kernel(
    const half* __restrict__ qkv,
    half* __restrict__ scores_probs,
    std::uint32_t dtype,
    std::uint32_t b_micro) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t per_matrix = static_cast<std::uint64_t>(STREAM1_TRANSFORMER_VPACK_STRIDE51);
    const std::uint64_t total = static_cast<std::uint64_t>(STREAM1_TRANSFORMER_NHEAD8) * b_micro * per_matrix;
    if (i >= total) {
        return;
    }
    const std::uint32_t lane = static_cast<std::uint32_t>(i % STREAM1_TRANSFORMER_HEAD_DIM32);
    const std::uint32_t key = static_cast<std::uint32_t>((i / STREAM1_TRANSFORMER_HEAD_DIM32) % STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51);
    const std::uint64_t row_head = i / per_matrix;
    const std::uint32_t row = static_cast<std::uint32_t>(row_head % b_micro);
    const std::uint32_t head = static_cast<std::uint32_t>(row_head / b_micro);
    const std::uint64_t matrix_base =
        (static_cast<std::uint64_t>(head) * b_micro + row) * STREAM1_TRANSFORMER_SCORE_STRIDE51;
    float value = 0.0f;
    if (key < STREAM1_TRANSFORMER_SEQ51) {
        const std::uint64_t qkv_idx =
            static_cast<std::uint64_t>(row) * STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_QKV_STRIDE51 +
            static_cast<std::uint64_t>(key) * STREAM1_TRANSFORMER_QKV_STRIDE51 +
            2ULL * STREAM1_TRANSFORMER_DMODEL256 +
            static_cast<std::uint64_t>(head) * STREAM1_TRANSFORMER_HEAD_DIM32 +
            lane;
        value = stream1_load_scalar_device(qkv, qkv_idx, dtype);
    }
    stream1_store_scalar_device(
        scores_probs,
        matrix_base + STREAM1_TRANSFORMER_PROB_STRIDE51 + static_cast<std::uint64_t>(key) * STREAM1_TRANSFORMER_HEAD_DIM32 + lane,
        value,
        dtype);
}

__global__ void stream1_transformer_build_attention_ptrs51_kernel(
    const half* __restrict__ qkv,
    half* __restrict__ scores_probs,
    half* __restrict__ context,
    const half** __restrict__ q_ptrs,
    const half** __restrict__ k_ptrs,
    const half** __restrict__ score_const_ptrs,
    half** __restrict__ score_ptrs,
    const half** __restrict__ prob_ptrs,
    const half** __restrict__ v_ptrs,
    const half** __restrict__ context_const_ptrs,
    half** __restrict__ context_ptrs,
    std::uint32_t b_micro) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * STREAM1_TRANSFORMER_NHEAD8;
    if (i >= total) {
        return;
    }
    const std::uint32_t row = i % b_micro;
    const std::uint32_t head = i / b_micro;
    const std::uint64_t qkv_row_base = static_cast<std::uint64_t>(row) *
        STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_QKV_STRIDE51;
    const std::uint64_t head_offset = static_cast<std::uint64_t>(head) * STREAM1_TRANSFORMER_HEAD_DIM32;
    half* score_base = scores_probs + static_cast<std::uint64_t>(i) * STREAM1_TRANSFORMER_SCORE_STRIDE51;
    half* context_base = context +
        static_cast<std::uint64_t>(row) * STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_DMODEL256 + head_offset;
    const half* q_base = qkv + qkv_row_base + head_offset;
    const half* k_base = qkv + qkv_row_base + STREAM1_TRANSFORMER_DMODEL256 + head_offset;
    const half* v_base = score_base + STREAM1_TRANSFORMER_PROB_STRIDE51;
    q_ptrs[i] = q_base;
    k_ptrs[i] = k_base;
    score_const_ptrs[i] = score_base;
    score_ptrs[i] = score_base;
    prob_ptrs[i] = score_base;
    v_ptrs[i] = v_base;
    context_const_ptrs[i] = context_base;
    context_ptrs[i] = context_base;
}
#if BEAM_HAS_CUTLASS
void stream1_transformer_prepare_attention_ptrs_launch(
    const Stream1TransformerScratchView& scratch,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    if (scratch.attention_scores_probs == nullptr || scratch.attention_context == nullptr ||
        scratch.attention_q_ptrs == nullptr || scratch.attention_k_ptrs == nullptr ||
        scratch.attention_score_const_ptrs == nullptr || scratch.attention_score_ptrs == nullptr ||
        scratch.attention_prob_ptrs == nullptr || scratch.attention_v_ptrs == nullptr ||
        scratch.attention_context_const_ptrs == nullptr || scratch.attention_context_ptrs == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer tensor attention requires pointer-array scratch");
    }
    const std::uint32_t total = b_micro * STREAM1_TRANSFORMER_NHEAD8;
    stream1_transformer_build_attention_ptrs51_kernel<<<(total + 255U) / 256U, 256, 0, stream>>>(
        scratch.qkv,
        scratch.attention_scores_probs,
        scratch.attention_context,
        scratch.attention_q_ptrs,
        scratch.attention_k_ptrs,
        scratch.attention_score_const_ptrs,
        scratch.attention_score_ptrs,
        scratch.attention_prob_ptrs,
        scratch.attention_v_ptrs,
        scratch.attention_context_const_ptrs,
        scratch.attention_context_ptrs,
        b_micro);
}
void stream1_transformer_qk_gemm_array_launch(
    const Stream1TransformerScratchView& scratch,
    std::uint32_t dtype,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    const cutlass::gemm::GemmCoord problem(
        static_cast<int>(STREAM1_TRANSFORMER_SEQ51),
        static_cast<int>(STREAM1_TRANSFORMER_SEQ51),
        static_cast<int>(STREAM1_TRANSFORMER_HEAD_DIM32));
    const int batch_count = static_cast<int>(b_micro * STREAM1_TRANSFORMER_NHEAD8);
    constexpr float scale = 0.1767766952966369f;
    cutlass::Status status = cutlass::Status::kSuccess;
    if (dtype == STREAM1_DTYPE_BF16) {
        using Gemm = cutlass::gemm::device::GemmArray<
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            cutlass::bfloat16_t,
            cutlass::layout::ColumnMajor,
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>>;
        Gemm gemm;
        for (int batch_offset = 0; batch_offset < batch_count; batch_offset += STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX) {
            const int remaining = batch_count - batch_offset;
            const int this_batch = remaining < STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX ? remaining : STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX;
            typename Gemm::Arguments args{
                problem,
                reinterpret_cast<cutlass::bfloat16_t const * const *>(scratch.attention_q_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_QKV_STRIDE51)),
                reinterpret_cast<cutlass::bfloat16_t const * const *>(scratch.attention_k_ptrs + batch_offset),
                cutlass::layout::ColumnMajor(static_cast<int>(STREAM1_TRANSFORMER_QKV_STRIDE51)),
                reinterpret_cast<cutlass::bfloat16_t const * const *>(scratch.attention_score_const_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)),
                reinterpret_cast<cutlass::bfloat16_t * const *>(scratch.attention_score_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)),
                {scale, 0.0f},
                this_batch};
            status = gemm(args, nullptr, stream);
            if (status != cutlass::Status::kSuccess) {
                break;
            }
        }
    } else if (dtype == STREAM1_DTYPE_FP16) {
        using Gemm = cutlass::gemm::device::GemmArray<
            cutlass::half_t,
            cutlass::layout::RowMajor,
            cutlass::half_t,
            cutlass::layout::ColumnMajor,
            cutlass::half_t,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 8>>;
        Gemm gemm;
        for (int batch_offset = 0; batch_offset < batch_count; batch_offset += STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX) {
            const int remaining = batch_count - batch_offset;
            const int this_batch = remaining < STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX ? remaining : STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX;
            typename Gemm::Arguments args{
                problem,
                reinterpret_cast<cutlass::half_t const * const *>(scratch.attention_q_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_QKV_STRIDE51)),
                reinterpret_cast<cutlass::half_t const * const *>(scratch.attention_k_ptrs + batch_offset),
                cutlass::layout::ColumnMajor(static_cast<int>(STREAM1_TRANSFORMER_QKV_STRIDE51)),
                reinterpret_cast<cutlass::half_t const * const *>(scratch.attention_score_const_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)),
                reinterpret_cast<cutlass::half_t * const *>(scratch.attention_score_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)),
                {scale, 0.0f},
                this_batch};
            status = gemm(args, nullptr, stream);
            if (status != cutlass::Status::kSuccess) {
                break;
            }
        }
    } else {
        throw std::invalid_argument("Stream1 piece_transformer dtype must be fp16 or bf16");
    }
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Stream1 piece_transformer QK GemmArray launch failed");
    }
}

void stream1_transformer_pv_gemm_array_launch(
    const Stream1TransformerScratchView& scratch,
    std::uint32_t dtype,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    const cutlass::gemm::GemmCoord problem(
        static_cast<int>(STREAM1_TRANSFORMER_SEQ51),
        static_cast<int>(STREAM1_TRANSFORMER_HEAD_DIM32),
        static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51));
    const int batch_count = static_cast<int>(b_micro * STREAM1_TRANSFORMER_NHEAD8);
    cutlass::Status status = cutlass::Status::kSuccess;
    if (dtype == STREAM1_DTYPE_BF16) {
        using Gemm = cutlass::gemm::device::GemmArray<
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>>;
        Gemm gemm;
        for (int batch_offset = 0; batch_offset < batch_count; batch_offset += STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX) {
            const int remaining = batch_count - batch_offset;
            const int this_batch = remaining < STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX ? remaining : STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX;
            typename Gemm::Arguments args{
                problem,
                reinterpret_cast<cutlass::bfloat16_t const * const *>(scratch.attention_prob_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)),
                reinterpret_cast<cutlass::bfloat16_t const * const *>(scratch.attention_v_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_HEAD_DIM32)),
                reinterpret_cast<cutlass::bfloat16_t const * const *>(scratch.attention_context_const_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_DMODEL256)),
                reinterpret_cast<cutlass::bfloat16_t * const *>(scratch.attention_context_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_DMODEL256)),
                {1.0f, 0.0f},
                this_batch};
            status = gemm(args, nullptr, stream);
            if (status != cutlass::Status::kSuccess) {
                break;
            }
        }
    } else if (dtype == STREAM1_DTYPE_FP16) {
        using Gemm = cutlass::gemm::device::GemmArray<
            cutlass::half_t,
            cutlass::layout::RowMajor,
            cutlass::half_t,
            cutlass::layout::RowMajor,
            cutlass::half_t,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 8>>;
        Gemm gemm;
        for (int batch_offset = 0; batch_offset < batch_count; batch_offset += STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX) {
            const int remaining = batch_count - batch_offset;
            const int this_batch = remaining < STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX ? remaining : STREAM1_TRANSFORMER_GEMM_ARRAY_BATCH_MAX;
            typename Gemm::Arguments args{
                problem,
                reinterpret_cast<cutlass::half_t const * const *>(scratch.attention_prob_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)),
                reinterpret_cast<cutlass::half_t const * const *>(scratch.attention_v_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_HEAD_DIM32)),
                reinterpret_cast<cutlass::half_t const * const *>(scratch.attention_context_const_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_DMODEL256)),
                reinterpret_cast<cutlass::half_t * const *>(scratch.attention_context_ptrs + batch_offset),
                cutlass::layout::RowMajor(static_cast<int>(STREAM1_TRANSFORMER_DMODEL256)),
                {1.0f, 0.0f},
                this_batch};
            status = gemm(args, nullptr, stream);
            if (status != cutlass::Status::kSuccess) {
                break;
            }
        }
    } else {
        throw std::invalid_argument("Stream1 piece_transformer dtype must be fp16 or bf16");
    }
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Stream1 piece_transformer PV GemmArray launch failed");
    }
}

#endif

void stream1_transformer_attention_launch(
    half* qkv,
    const half* qkv_bias,
    const Stream1TransformerScratchView& scratch,
    Stream1TransformerDims dims,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    if (dims.seq_len != STREAM1_TRANSFORMER_SEQ51 ||
        dims.d_model != STREAM1_TRANSFORMER_DMODEL256 ||
        dims.nhead != STREAM1_TRANSFORMER_NHEAD8 ||
        dims.head_dim != STREAM1_TRANSFORMER_HEAD_DIM32) {
        throw std::invalid_argument("Stream1 piece_transformer tensor attention requires seq_len=51 d_model=256 nhead=8 head_dim=32");
    }
    if (scratch.attention_scores_probs == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer tensor attention requires attention_scores_probs scratch");
    }
#if BEAM_HAS_CUTLASS
    const std::uint64_t qkv_total = static_cast<std::uint64_t>(b_micro) *
        STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_QKV_STRIDE51;
    stream1_transformer_add_qkv_bias_kernel<<<
        static_cast<unsigned>((qkv_total + 255ULL) / 256ULL),
        256,
        0,
        stream>>>(qkv, qkv_bias, dims.dtype, b_micro);
    stream1_transformer_qk_gemm_array_launch(scratch, dims.dtype, b_micro, stream);
    stream1_transformer_pack_v51_kernel<<<
        static_cast<unsigned>(((static_cast<std::uint64_t>(b_micro) * STREAM1_TRANSFORMER_NHEAD8 * STREAM1_TRANSFORMER_VPACK_STRIDE51) + 255ULL) / 256ULL),
        256,
        0,
        stream>>>(qkv, scratch.attention_scores_probs, dims.dtype, b_micro);
    stream1_transformer_softmax51_kernel<<<
        dim3(b_micro, STREAM1_TRANSFORMER_NHEAD8),
        256,
        0,
        stream>>>(scratch.attention_scores_probs, dims.dtype, b_micro);
    stream1_transformer_pv_gemm_array_launch(scratch, dims.dtype, b_micro, stream);
#else
    (void)qkv;
    (void)qkv_bias;
    (void)scratch;
    (void)dims;
    (void)b_micro;
    (void)stream;
    throw std::invalid_argument("Stream1 piece_transformer tensor attention requires CUTLASS-enabled build");
#endif
}

__global__ void stream1_transformer_cls_layernorm_kernel(
    const half* __restrict__ tokens,
    half* __restrict__ cls,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    Stream1TransformerDims dims,
    std::uint32_t b_micro) {
    const std::uint32_t row = blockIdx.x;
    if (row >= b_micro) {
        return;
    }
    extern __shared__ float reduce[];
    const std::uint64_t in_base = static_cast<std::uint64_t>(row) * dims.seq_len * dims.d_model;
    const std::uint64_t out_base = static_cast<std::uint64_t>(row) * dims.d_model;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < dims.d_model; col += blockDim.x) {
        sum += stream1_load_scalar_device(tokens, in_base + col, dims.dtype);
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float mean = reduce[0] / static_cast<float>(dims.d_model);
    float var_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < dims.d_model; col += blockDim.x) {
        const float x = stream1_load_scalar_device(tokens, in_base + col, dims.dtype);
        const float d = x - mean;
        var_sum += d * d;
    }
    reduce[threadIdx.x] = var_sum;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x / 2U; stride > 0U; stride >>= 1U) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_std = rsqrtf(reduce[0] / static_cast<float>(dims.d_model) + 1.0e-5f);
    for (std::uint32_t col = threadIdx.x; col < dims.d_model; col += blockDim.x) {
        const float x = stream1_load_scalar_device(tokens, in_base + col, dims.dtype);
        const float y = (x - mean) * inv_std *
            stream1_load_scalar_device(gamma, col, dims.dtype) +
            stream1_load_scalar_device(beta, col, dims.dtype);
        stream1_store_scalar_device(cls, out_base + col, y, dims.dtype);
    }
}

__global__ void stream1_transformer_score_quantize_kernel(
    const half* __restrict__ logits,
    const half* __restrict__ output_bias,
    const std::uint32_t* __restrict__ count,
    std::uint32_t* __restrict__ score_ring,
    std::uint32_t b_micro,
    Stream1TransformerDims dims) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (i >= total) {
        return;
    }
    const std::uint32_t row = i / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint32_t move = i % static_cast<std::uint32_t>(MOVE_COUNT);
    if (row >= *count) {
        return;
    }
    const float q = stream1_load_scalar_device(logits, static_cast<std::uint64_t>(row) * dims.output_dim + move, dims.dtype) +
        stream1_load_scalar_device(output_bias, move, dims.dtype);
    score_ring[i] = score_key_from_float_device(q);
}

void stream1_transformer_inference_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    NvtxRange range("Stream1_transformer_inference_launch");
    const Stream1TransformerDims dims = network.dims;
    if (dims.output_dim != MOVE_COUNT) {
        throw std::invalid_argument("Stream1 piece_transformer output_dim must equal MOVE_COUNT");
    }
    if (dims.d_model == 0U || dims.nhead == 0U || dims.head_dim == 0U ||
        dims.d_model != dims.nhead * dims.head_dim) {
        throw std::invalid_argument("Stream1 piece_transformer d_model must equal nhead * head_dim");
    }
    if (dims.seq_len != dims.num_pieces + 1U || dims.max_piece_size == 0U) {
        throw std::invalid_argument("Stream1 piece_transformer sequence/piece dimensions are inconsistent");
    }
    if (dims.seq_len > 64U || dims.head_dim > 64U) {
        throw std::invalid_argument("Stream1 piece_transformer fused attention tile requires seq_len<=64 and head_dim<=64");
    }
    if (dims.dtype == STREAM1_DTYPE_BF16) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 8) {
            throw std::invalid_argument("Stream1 piece_transformer bf16 path requires SM80+ GPU; use fp16 weights on T4/SM75");
        }
    } else if (dims.dtype != STREAM1_DTYPE_FP16) {
        throw std::invalid_argument("Stream1 piece_transformer dtype must be fp16 or bf16");
    }
    if (b_micro == 0U) {
        return;
    }
#if BEAM_HAS_CUTLASS
    const std::uint32_t token_rows = b_micro * dims.seq_len;
    const dim3 token_block(128);
    const dim3 token_grid(token_rows, (dims.d_model + token_block.x - 1U) / token_block.x);
    stream1_transformer_build_input_kernel<<<token_grid, token_block, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        network,
        scratch.tokens,
        b_micro);

    stream1_transformer_layernorm_copy_launch(
        scratch.tokens,
        scratch.tokens,
        network.input_ln_gamma,
        network.input_ln_beta,
        token_rows,
        dims.d_model,
        dims.dtype,
        stream);

    stream1_transformer_prepare_attention_ptrs_launch(scratch, b_micro, stream);

    for (std::uint32_t layer = 0; layer < dims.transformer_layers; ++layer) {
        const Stream1TransformerBlockView block = network.blocks[layer];
        stream1_transformer_layernorm_copy_launch(
            scratch.tokens,
            scratch.attention_context,
            block.ln1_gamma,
            block.ln1_beta,
            token_rows,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_cutlass_linear_cuda(
            scratch.attention_context,
            block.attn_qkv_weight,
            scratch.qkv,
            token_rows,
            dims.d_model,
            3U * dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_attention_launch(
            scratch.qkv,
            block.attn_qkv_bias,
            scratch,
            dims,
            b_micro,
            stream);
        stream1_cutlass_linear_cuda(
            scratch.attention_context,
            block.attn_out_weight,
            scratch.qkv,
            token_rows,
            dims.d_model,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_residual_bias_add_kernel<<<(token_rows * dims.d_model + 255U) / 256U, 256, 0, stream>>>(
            scratch.tokens,
            scratch.qkv,
            block.attn_out_bias,
            token_rows,
            dims.d_model,
            dims.dtype);

        stream1_transformer_layernorm_copy_launch(
            scratch.tokens,
            scratch.attention_context,
            block.ln2_gamma,
            block.ln2_beta,
            token_rows,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_cutlass_linear_cuda(
            scratch.attention_context,
            block.ff1_weight,
            scratch.ff_hidden,
            token_rows,
            dims.d_model,
            dims.ff_dim,
            dims.dtype,
            stream);
        stream1_transformer_bias_silu_kernel<<<(token_rows * dims.ff_dim + 255U) / 256U, 256, 0, stream>>>(
            scratch.ff_hidden,
            block.ff1_bias,
            token_rows,
            dims.ff_dim,
            dims.dtype);
        stream1_cutlass_linear_cuda(
            scratch.ff_hidden,
            block.ff2_weight,
            scratch.attention_context,
            token_rows,
            dims.ff_dim,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_residual_bias_add_kernel<<<(token_rows * dims.d_model + 255U) / 256U, 256, 0, stream>>>(
            scratch.tokens,
            scratch.attention_context,
            block.ff2_bias,
            token_rows,
            dims.d_model,
            dims.dtype);
    }

    stream1_transformer_cls_layernorm_kernel<<<b_micro, 256, 256 * sizeof(float), stream>>>(
        scratch.tokens,
        scratch.attention_context,
        network.output_ln_gamma,
        network.output_ln_beta,
        dims,
        b_micro);
    stream1_cutlass_linear_cuda(
        scratch.attention_context,
        network.output_weight,
        scratch.logits,
        b_micro,
        dims.d_model,
        dims.output_dim,
        dims.dtype,
        stream);
    stream1_transformer_score_quantize_kernel<<<(b_micro * MOVE_COUNT + 255U) / 256U, 256, 0, stream>>>(
        scratch.logits,
        network.output_bias,
        count,
        score_ring,
        b_micro,
        dims);
#else
    (void)current_frontier_states;
    (void)parent_base;
    (void)count;
    (void)network;
    (void)scratch;
    (void)score_ring;
    (void)b_micro;
    (void)stream;
    throw std::invalid_argument("Stream1 piece_transformer CUDA forward requires CUTLASS-enabled build");
#endif
}
void stream1_inference_custom_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const Stream1NetworkView& network,
    std::uint32_t* score_ring,
    std::uint32_t,
    std::uint32_t,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    NvtxRange range("Stream1_custom_inference_launch");
    const std::size_t shared_bytes =
        (static_cast<std::size_t>(network.dims.hidden1) + static_cast<std::size_t>(network.dims.hidden2)) * sizeof(float);
    stream1_inference_custom_kernel<<<b_micro, MOVE_COUNT, shared_bytes, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        network,
        score_ring,
        0,
        0,
        b_micro);
}

void stream1_inference_cutlass_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const std::uint8_t* generators,
    const Stream1NetworkView& network,
    const Stream1CutlassScratch& scratch,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    NvtxRange range("Stream1_CUTLASS_inference_launch");
    if (network.dims.output_dim != MOVE_COUNT &&
        network.dims.output_dim != STREAM1_SINGLE_SCORE_OUTPUT_DIM) {
        throw std::invalid_argument("Stream1 output_dim must be 1 or MOVE_COUNT");
    }
    if (network.dims.output_dim == STREAM1_SINGLE_SCORE_OUTPUT_DIM && generators == nullptr) {
        throw std::invalid_argument("Stream1 output_dim=1 requires generator table");
    }
    const std::uint32_t inference_rows =
        b_micro * (network.dims.output_dim == STREAM1_SINGLE_SCORE_OUTPUT_DIM
            ? static_cast<std::uint32_t>(MOVE_COUNT)
            : 1U);
    if (network.dims.dtype == STREAM1_DTYPE_BF16) {
        int device = 0;
        cudaGetDevice(&device);
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 8) {
            throw std::invalid_argument("Stream1 bf16 CUTLASS path requires SM80+ GPU");
        }
    }
#if BEAM_HAS_CUTLASS
    const dim3 input_block(128);
    const dim3 input_grid(inference_rows, (network.dims.hidden1 + input_block.x * 4U - 1U) / (input_block.x * 4U));
    stream1_folded_input_half4_tiled_kernel<<<input_grid, input_block, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        generators,
        network.input_weight,
        network.input_bias,
        scratch.hidden1,
        network.dims,
        b_micro);
    if (network.dims.normalization == STREAM1_NORM_LAYERNORM) {
        stream1_layernorm_relu_kernel<<<inference_rows, 256, 256 * sizeof(float), stream>>>(
            scratch.hidden1,
            network.input_bias,
            network.input_ln_gamma,
            network.input_ln_beta,
            inference_rows,
            network.dims.hidden1,
            network.dims.dtype);
    }

    stream1_cutlass_linear_cuda(
        scratch.hidden1,
        network.hidden_weight,
        scratch.hidden2,
        inference_rows,
        network.dims.hidden1,
        network.dims.hidden2,
        network.dims.dtype,
        stream);

    const std::uint32_t hidden2_total = inference_rows * network.dims.hidden2;
    if (network.dims.normalization == STREAM1_NORM_LAYERNORM) {
        stream1_layernorm_relu_kernel<<<inference_rows, 256, 256 * sizeof(float), stream>>>(
            scratch.hidden2,
            network.hidden_bias,
            network.hidden_ln_gamma,
            network.hidden_ln_beta,
            inference_rows,
            network.dims.hidden2,
            network.dims.dtype);
    } else {
        stream1_bias_relu_kernel<<<(hidden2_total + 255U) / 256U, 256, 0, stream>>>(
            scratch.hidden2,
            network.hidden_bias,
            inference_rows,
            network.dims.hidden2,
            network.dims.dtype);
    }

    half* residual_in = scratch.hidden2;
    half* residual_fc1 = scratch.residual_tmp;
    half* residual_fc2 = scratch.hidden1;
    for (std::uint32_t block = 0; block < network.dims.residual_count; ++block) {
        stream1_cutlass_linear_cuda(
            residual_in,
            network.residual_fc1_weight[block],
            residual_fc1,
            inference_rows,
            network.dims.hidden2,
            network.dims.hidden2,
            network.dims.dtype,
            stream);

        if (network.dims.normalization == STREAM1_NORM_LAYERNORM) {
            stream1_layernorm_relu_kernel<<<inference_rows, 256, 256 * sizeof(float), stream>>>(
                residual_fc1,
                network.residual_fc1_bias[block],
                network.residual_fc1_ln_gamma[block],
                network.residual_fc1_ln_beta[block],
                inference_rows,
                network.dims.hidden2,
                network.dims.dtype);
        } else {
            stream1_bias_relu_kernel<<<(hidden2_total + 255U) / 256U, 256, 0, stream>>>(
                residual_fc1,
                network.residual_fc1_bias[block],
                inference_rows,
                network.dims.hidden2,
                network.dims.dtype);
        }

        stream1_cutlass_linear_cuda(
            residual_fc1,
            network.residual_fc2_weight[block],
            residual_fc2,
            inference_rows,
            network.dims.hidden2,
            network.dims.hidden2,
            network.dims.dtype,
            stream);

        if (network.dims.normalization == STREAM1_NORM_LAYERNORM) {
            stream1_layernorm_residual_relu_kernel<<<inference_rows, 256, 256 * sizeof(float), stream>>>(
                residual_fc2,
                residual_in,
                network.residual_fc2_bias[block],
                network.residual_fc2_ln_gamma[block],
                network.residual_fc2_ln_beta[block],
                inference_rows,
                network.dims.hidden2,
                network.dims.dtype);
        } else {
            stream1_residual_add_bias_relu_kernel<<<(hidden2_total + 255U) / 256U, 256, 0, stream>>>(
                residual_fc2,
                residual_in,
                network.residual_fc2_bias[block],
                inference_rows,
                network.dims.hidden2,
                network.dims.dtype);
        }

        residual_in = residual_fc2;
        residual_fc2 = residual_in == scratch.hidden1 ? scratch.hidden2 : scratch.hidden1;
    }

    const std::uint32_t output_total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (network.dims.output_dim == STREAM1_SINGLE_SCORE_OUTPUT_DIM) {
        stream1_single_output_quantize_kernel<<<(output_total + 255U) / 256U, 256, 0, stream>>>(
            residual_in,
            network.output_weight,
            network.output_bias,
            count,
            score_ring,
            b_micro,
            network.dims.hidden2,
            network.dims.dtype);
    } else {
        stream1_cutlass_linear_cuda(
            residual_in,
            network.output_weight,
            scratch.output,
            inference_rows,
            network.dims.hidden2,
            network.dims.output_dim,
            network.dims.dtype,
            stream);

        stream1_score_epilogue_quantize_kernel<<<(output_total + 255U) / 256U, 256, 0, stream>>>(
            scratch.output,
            network.output_bias,
            count,
            score_ring,
            b_micro,
            network.dims.dtype);
    }
#else
    (void)current_frontier_states;
    (void)parent_base;
    (void)count;
    (void)generators;
    (void)network;
    (void)scratch;
    (void)score_ring;
    (void)b_micro;
    (void)stream;
    cudaGetLastError();
#endif
}

void stream1_cutlass_linear_cuda(
    const half* input,
    const half* weight,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    NvtxRange range("Stream1_CUTLASS_linear_launch");
#if BEAM_HAS_CUTLASS
    cutlass::gemm::GemmCoord problem(static_cast<int>(rows), static_cast<int>(output_cols), static_cast<int>(input_cols));
    cutlass::Status status = cutlass::Status::kSuccess;
    if (dtype == STREAM1_DTYPE_BF16) {
        int device = 0;
        cudaDeviceProp prop{};
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        if (prop.major < 8) {
            throw std::invalid_argument("Stream1 bf16 weights require SM80+ GPU; use fp16 weights on T4/SM75");
        }
        using Gemm = cutlass::gemm::device::Gemm<
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            cutlass::bfloat16_t,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 16>>;
        Gemm gemm;
        typename Gemm::Arguments args{
            problem,
            {reinterpret_cast<const cutlass::bfloat16_t*>(input), static_cast<int>(input_cols)},
            {reinterpret_cast<const cutlass::bfloat16_t*>(weight), static_cast<int>(output_cols)},
            {reinterpret_cast<cutlass::bfloat16_t*>(output), static_cast<int>(output_cols)},
            {reinterpret_cast<cutlass::bfloat16_t*>(output), static_cast<int>(output_cols)},
            {1.0f, 0.0f}};
        status = gemm(args, nullptr, stream);
    } else {
        using Gemm = cutlass::gemm::device::Gemm<
            cutlass::half_t,
            cutlass::layout::RowMajor,
            cutlass::half_t,
            cutlass::layout::RowMajor,
            cutlass::half_t,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<16, 8, 8>>;
        Gemm gemm;
        typename Gemm::Arguments args{
            problem,
            {reinterpret_cast<const cutlass::half_t*>(input), static_cast<int>(input_cols)},
            {reinterpret_cast<const cutlass::half_t*>(weight), static_cast<int>(output_cols)},
            {reinterpret_cast<cutlass::half_t*>(output), static_cast<int>(output_cols)},
            {reinterpret_cast<cutlass::half_t*>(output), static_cast<int>(output_cols)},
            {1.0f, 0.0f}};
        status = gemm(args, nullptr, stream);
    }
    if (status != cutlass::Status::kSuccess) {
        cudaGetLastError();
    }
#else
    (void)input;
    (void)weight;
    (void)output;
    (void)rows;
    (void)input_cols;
    (void)output_cols;
    (void)dtype;
    (void)stream;
#endif
}

void stream1_cutlass_linear_relu_cuda(
    const half* input,
    const half* weight,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream) {
    NvtxRange range("Stream1_CUTLASS_linear_relu_epilogue_launch");
#if BEAM_HAS_CUTLASS
    using Epilogue = cutlass::epilogue::thread::LinearCombinationRelu<
        cutlass::half_t,
        1,
        float,
        float>;
    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        cutlass::half_t,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm75,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        Epilogue>;
    Gemm gemm;
    cutlass::gemm::GemmCoord problem(static_cast<int>(rows), static_cast<int>(output_cols), static_cast<int>(input_cols));
    typename Gemm::Arguments args{
        problem,
        {reinterpret_cast<const cutlass::half_t*>(input), static_cast<int>(input_cols)},
        {reinterpret_cast<const cutlass::half_t*>(weight), static_cast<int>(output_cols)},
        {reinterpret_cast<cutlass::half_t*>(output), static_cast<int>(output_cols)},
        {reinterpret_cast<cutlass::half_t*>(output), static_cast<int>(output_cols)},
        {1.0f, 0.0f}};
    const auto status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        cudaGetLastError();
    }
#else
    (void)input;
    (void)weight;
    (void)output;
    (void)rows;
    (void)input_cols;
    (void)output_cols;
    (void)stream;
#endif
}

} // namespace beam
