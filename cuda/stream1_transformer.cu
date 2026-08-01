#include "stream1.hpp"
#include "stream1_transformer_fmha.hpp"
#include "stream1_transformer_gemm_policy.hpp"
#include "stream1_transformer_layernorm_policy.hpp"

#include "config.hpp"
#include "cuda_check.hpp"
#include "nvtx_ranges.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#if BEAM_HAS_CUTLASS
#include <cutlass/bfloat16.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#endif

#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace beam {

__device__ __forceinline__ float stream1_transformer_warp_reduce_sum_device(float value) {
    constexpr unsigned mask = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return __shfl_sync(mask, value, 0);
}

__device__ __forceinline__ float stream1_transformer_warp_reduce_max_device(float value) {
    constexpr unsigned mask = 0xffffffffU;
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(mask, value, offset));
    }
    return __shfl_sync(mask, value, 0);
}

__device__ __forceinline__ std::uint32_t stream1_transformer_score_key_from_float_device(float q) {
    q = fminf(fmaxf(q, 0.0f), SCORE_MAX_Q);
    return static_cast<std::uint32_t>(rintf(q * static_cast<float>(SCORE_SCALE)));
}

__device__ __forceinline__ float stream1_transformer_load_scalar_device(const half* ptr, std::uint64_t idx, std::uint32_t dtype) {
    if (dtype == STREAM1_DTYPE_BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
    }
    return __half2float(ptr[idx]);
}

__device__ __forceinline__ void stream1_transformer_store_scalar_device(half* ptr, std::uint64_t idx, float value, std::uint32_t dtype) {
    if (dtype == STREAM1_DTYPE_BF16) {
        reinterpret_cast<__nv_bfloat16*>(ptr)[idx] = __float2bfloat16(value);
        return;
    }
    ptr[idx] = __float2half(value);
}
__global__ void stream1_transformer_build_input_kernel(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    const std::uint32_t* __restrict__ count,
    Stream1TransformerNetworkView network,
    half* __restrict__ tokens,
    std::uint32_t b_micro,
    std::uint32_t parent_offset) {
    const std::uint32_t row_token = blockIdx.x;
    const std::uint32_t dim = blockIdx.y * blockDim.x + threadIdx.x;
    const std::uint32_t row = row_token / network.dims.padded_seq_len;
    const std::uint32_t token = row_token % network.dims.padded_seq_len;
    const std::uint32_t active_count = *count;
    if (row >= b_micro || dim >= network.dims.d_model) {
        return;
    }
    if (parent_offset + row >= active_count || token >= network.dims.seq_len) {
        const std::uint64_t out_idx = static_cast<std::uint64_t>(row_token) * network.dims.d_model + dim;
        stream1_transformer_store_scalar_device(tokens, out_idx, 0.0f, network.dims.dtype);
        return;
    }
    float value = 0.0f;
    if (token == 0U) {
        value = stream1_transformer_load_scalar_device(network.cls_token, dim, network.dims.dtype);
    } else {
        const std::uint32_t piece = token - 1U;
        value = stream1_transformer_load_scalar_device(
            network.fast_piece_static,
            static_cast<std::uint64_t>(piece) * network.dims.d_model + dim,
            network.dims.dtype);
        const std::uint64_t parent_idx = *parent_base + static_cast<std::uint64_t>(parent_offset + row);
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
            value += stream1_transformer_load_scalar_device(network.fast_slot_projected, table_idx, network.dims.dtype);
        }
    }
    const std::uint64_t out_idx = static_cast<std::uint64_t>(row_token) * network.dims.d_model + dim;
    stream1_transformer_store_scalar_device(tokens, out_idx, value, network.dims.dtype);
}

__global__ void stream1_transformer_zero_padded_rows_kernel(
    half* values,
    Stream1TransformerDims dims,
    std::uint32_t rows) {
    const std::uint64_t idx = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * dims.padded_seq_len * dims.d_model;
    if (idx >= total) {
        return;
    }
    const std::uint32_t token = static_cast<std::uint32_t>((idx / dims.d_model) % dims.padded_seq_len);
    if (token >= dims.seq_len) {
        stream1_transformer_store_scalar_device(values, idx, 0.0f, dims.dtype);
    }
}

void stream1_transformer_zero_padded_rows_launch(
    half* values,
    Stream1TransformerDims dims,
    std::uint32_t rows,
    cudaStream_t stream) {
    if (dims.padded_seq_len == dims.seq_len || rows == 0U) {
        return;
    }
    const std::uint64_t total = static_cast<std::uint64_t>(rows) * dims.padded_seq_len * dims.d_model;
    constexpr std::uint32_t threads = 256U;
    const std::uint64_t blocks = (total + threads - 1U) / threads;
    if (blocks > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Stream1 transformer zero-tail launch grid overflow");
    }
    stream1_transformer_zero_padded_rows_kernel<<<static_cast<std::uint32_t>(blocks), threads, 0, stream>>>(
        values, dims, rows);
}
__device__ float stream1_transformer_input_token_value51x256_device(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    Stream1TransformerNetworkView network,
    std::uint32_t parent_offset,
    std::uint32_t row,
    std::uint32_t token,
    std::uint32_t dim) {
    if (token == 0U) {
        return stream1_transformer_load_scalar_device(network.cls_token, dim, network.dims.dtype);
    }
    const std::uint32_t piece = token - 1U;
    float value = stream1_transformer_load_scalar_device(
        network.fast_piece_static,
        static_cast<std::uint64_t>(piece) * 256ULL + dim,
        network.dims.dtype);
    const std::uint64_t parent_idx = *parent_base + static_cast<std::uint64_t>(parent_offset + row);
    const State128* state = current_frontier_states + parent_idx;
    for (std::uint32_t slot = 0; slot < 3U; ++slot) {
        const std::uint64_t piece_slot = static_cast<std::uint64_t>(piece) * 3ULL + slot;
        if (network.piece_mask[piece_slot] == 0U) {
            continue;
        }
        const std::uint32_t pos = static_cast<std::uint32_t>(network.piece_positions[piece_slot]);
        const std::uint32_t state_value = static_cast<std::uint32_t>(state->v[pos]);
        const std::uint64_t table_idx =
            ((static_cast<std::uint64_t>(slot) * 120ULL + state_value) * 256ULL) + dim;
        value += stream1_transformer_load_scalar_device(network.fast_slot_projected, table_idx, network.dims.dtype);
    }
    return value;
}

__global__ void stream1_transformer_build_input_layernorm51x256_kernel(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    const std::uint32_t* __restrict__ count,
    Stream1TransformerNetworkView network,
    half* __restrict__ tokens,
    std::uint32_t b_micro,
    std::uint32_t parent_offset) {
    const std::uint32_t row_token = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (tid >= 128U) {
        return;
    }
    const std::uint32_t row = row_token / 51U;
    const std::uint32_t token = row_token % 51U;
    if (row >= b_micro) {
        return;
    }

    extern __shared__ float warp_scratch[];
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t base = static_cast<std::uint64_t>(row_token) * 256ULL;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    const bool active = parent_offset + row < *count;
    const float x0 = active ? stream1_transformer_input_token_value51x256_device(
        current_frontier_states, parent_base, network, parent_offset, row, token, col0) : 0.0f;
    const float x1 = active ? stream1_transformer_input_token_value51x256_device(
        current_frontier_states, parent_base, network, parent_offset, row, token, col1) : 0.0f;

    const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
    __syncthreads();
#endif

    const float centered0 = x0 - mean;
    const float centered1 = x1 - mean;
    const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
        centered0 * centered0 + centered1 * centered1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();

    float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] = rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

    const float y0 = centered0 * inv_std *
        stream1_transformer_load_scalar_device(network.input_ln_gamma, col0, network.dims.dtype) +
        stream1_transformer_load_scalar_device(network.input_ln_beta, col0, network.dims.dtype);
    const float y1 = centered1 * inv_std *
        stream1_transformer_load_scalar_device(network.input_ln_gamma, col1, network.dims.dtype) +
        stream1_transformer_load_scalar_device(network.input_ln_beta, col1, network.dims.dtype);
    stream1_transformer_store_scalar_device(tokens, base + col0, y0, network.dims.dtype);
    stream1_transformer_store_scalar_device(tokens, base + col1, y1, network.dims.dtype);
}

void stream1_transformer_build_input_layernorm51x256_launch(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const Stream1TransformerNetworkView& network,
    half* tokens,
    std::uint32_t b_micro,
    std::uint32_t parent_offset,
    cudaStream_t stream) {
    const Stream1TransformerDims dims = network.dims;
    if (dims.seq_len != 51U || dims.d_model != 256U || dims.num_pieces != 50U ||
        dims.max_piece_size != 3U || dims.num_classes != 120U) {
        throw std::invalid_argument("Stream1 piece_transformer block51 fused input requires seq=51 d=256 pieces=50 max_piece=3 classes=120");
    }
    stream1_transformer_build_input_layernorm51x256_kernel<<<b_micro * 51U, 128, STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float), stream>>>(
        current_frontier_states,
        parent_base,
        count,
        network,
        tokens,
        b_micro,
        parent_offset);
}

__global__ void stream1_transformer_build_input_kernel_graph_job(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    const std::uint32_t* __restrict__ count,
    const std::uint32_t* __restrict__ graph_job_index,
    Stream1TransformerNetworkView network,
    half* __restrict__ tokens,
    std::uint32_t b_micro,
    std::uint32_t parent_offset) {
    const std::uint32_t row_token = blockIdx.x;
    const std::uint32_t dim = blockIdx.y * blockDim.x + threadIdx.x;
    const std::uint32_t row = row_token / network.dims.padded_seq_len;
    const std::uint32_t token = row_token % network.dims.padded_seq_len;
    const std::uint32_t job = *graph_job_index;
    const std::uint32_t active_count = count[job];
    if (row >= b_micro || dim >= network.dims.d_model) {
        return;
    }
    if (parent_offset + row >= active_count || token >= network.dims.seq_len) {
        const std::uint64_t out_idx = static_cast<std::uint64_t>(row_token) * network.dims.d_model + dim;
        stream1_transformer_store_scalar_device(tokens, out_idx, 0.0f, network.dims.dtype);
        return;
    }
    float value = 0.0f;
    if (token == 0U) {
        value = stream1_transformer_load_scalar_device(network.cls_token, dim, network.dims.dtype);
    } else {
        const std::uint32_t piece = token - 1U;
        value = stream1_transformer_load_scalar_device(
            network.fast_piece_static,
            static_cast<std::uint64_t>(piece) * network.dims.d_model + dim,
            network.dims.dtype);
        const std::uint64_t parent_idx = parent_base[job] + static_cast<std::uint64_t>(parent_offset + row);
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
            value += stream1_transformer_load_scalar_device(network.fast_slot_projected, table_idx, network.dims.dtype);
        }
    }
    const std::uint64_t out_idx = static_cast<std::uint64_t>(row_token) * network.dims.d_model + dim;
    stream1_transformer_store_scalar_device(tokens, out_idx, value, network.dims.dtype);
}

__device__ float stream1_transformer_input_token_value51x256_graph_job_device(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    const std::uint32_t* __restrict__ graph_job_index,
    Stream1TransformerNetworkView network,
    std::uint32_t parent_offset,
    std::uint32_t row,
    std::uint32_t token,
    std::uint32_t dim) {
    if (token == 0U) {
        return stream1_transformer_load_scalar_device(network.cls_token, dim, network.dims.dtype);
    }
    const std::uint32_t piece = token - 1U;
    float value = stream1_transformer_load_scalar_device(
        network.fast_piece_static,
        static_cast<std::uint64_t>(piece) * 256ULL + dim,
        network.dims.dtype);
    const std::uint32_t job = *graph_job_index;
    const std::uint64_t parent_idx = parent_base[job] + static_cast<std::uint64_t>(parent_offset + row);
    const State128* state = current_frontier_states + parent_idx;
    for (std::uint32_t slot = 0; slot < 3U; ++slot) {
        const std::uint64_t piece_slot = static_cast<std::uint64_t>(piece) * 3ULL + slot;
        if (network.piece_mask[piece_slot] == 0U) {
            continue;
        }
        const std::uint32_t pos = static_cast<std::uint32_t>(network.piece_positions[piece_slot]);
        const std::uint32_t state_value = static_cast<std::uint32_t>(state->v[pos]);
        const std::uint64_t table_idx =
            ((static_cast<std::uint64_t>(slot) * 120ULL + state_value) * 256ULL) + dim;
        value += stream1_transformer_load_scalar_device(network.fast_slot_projected, table_idx, network.dims.dtype);
    }
    return value;
}

__global__ void stream1_transformer_build_input_layernorm51x256_graph_job_kernel(
    const State128* __restrict__ current_frontier_states,
    const std::uint64_t* __restrict__ parent_base,
    const std::uint32_t* __restrict__ count,
    const std::uint32_t* __restrict__ graph_job_index,
    Stream1TransformerNetworkView network,
    half* __restrict__ tokens,
    std::uint32_t b_micro,
    std::uint32_t parent_offset) {
    const std::uint32_t row_token = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (tid >= 128U) {
        return;
    }
    const std::uint32_t row = row_token / 51U;
    const std::uint32_t token = row_token % 51U;
    if (row >= b_micro) {
        return;
    }

    extern __shared__ float warp_scratch[];
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t base = static_cast<std::uint64_t>(row_token) * 256ULL;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    const std::uint32_t job = *graph_job_index;
    const bool active = parent_offset + row < count[job];
    const float x0 = active ? stream1_transformer_input_token_value51x256_graph_job_device(
        current_frontier_states, parent_base, graph_job_index, network, parent_offset, row, token, col0) : 0.0f;
    const float x1 = active ? stream1_transformer_input_token_value51x256_graph_job_device(
        current_frontier_states, parent_base, graph_job_index, network, parent_offset, row, token, col1) : 0.0f;

    const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
    __syncthreads();
#endif

    const float centered0 = x0 - mean;
    const float centered1 = x1 - mean;
    const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
        centered0 * centered0 + centered1 * centered1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();

    float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] = rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

    const float y0 = centered0 * inv_std *
        stream1_transformer_load_scalar_device(network.input_ln_gamma, col0, network.dims.dtype) +
        stream1_transformer_load_scalar_device(network.input_ln_beta, col0, network.dims.dtype);
    const float y1 = centered1 * inv_std *
        stream1_transformer_load_scalar_device(network.input_ln_gamma, col1, network.dims.dtype) +
        stream1_transformer_load_scalar_device(network.input_ln_beta, col1, network.dims.dtype);
    stream1_transformer_store_scalar_device(tokens, base + col0, y0, network.dims.dtype);
    stream1_transformer_store_scalar_device(tokens, base + col1, y1, network.dims.dtype);
}

void stream1_transformer_build_input_layernorm51x256_graph_job_launch(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const std::uint32_t* graph_job_index,
    const Stream1TransformerNetworkView& network,
    half* tokens,
    std::uint32_t b_micro,
    std::uint32_t parent_offset,
    cudaStream_t stream) {
    const Stream1TransformerDims dims = network.dims;
    if (dims.seq_len != 51U || dims.d_model != 256U || dims.num_pieces != 50U ||
        dims.max_piece_size != 3U || dims.num_classes != 120U) {
        throw std::invalid_argument("Stream1 piece_transformer block51 fused input requires seq=51 d=256 pieces=50 max_piece=3 classes=120");
    }
    stream1_transformer_build_input_layernorm51x256_graph_job_kernel<<<b_micro * 51U, 128, STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float), stream>>>(
        current_frontier_states,
        parent_base,
        count,
        graph_job_index,
        network,
        tokens,
        b_micro,
        parent_offset);
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
        sum += stream1_transformer_load_scalar_device(input, base + col, dtype);
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
    __syncthreads();
    float var_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float x = stream1_transformer_load_scalar_device(input, base + col, dtype);
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
        const float x = stream1_transformer_load_scalar_device(input, base + col, dtype);
        const float y = (x - mean) * inv_std *
            stream1_transformer_load_scalar_device(gamma, col, dtype) +
            stream1_transformer_load_scalar_device(beta, col, dtype);
        stream1_transformer_store_scalar_device(output, base + col, y, dtype);
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
    if (row >= rows || tid >= 128U) {
        return;
    }

    extern __shared__ float warp_scratch[];
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * 256ULL;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    const float x0 = stream1_transformer_load_scalar_device(input, base + col0, dtype);
    const float x1 = stream1_transformer_load_scalar_device(input, base + col1, dtype);

    const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
    __syncthreads();
#endif

    const float centered0 = x0 - mean;
    const float centered1 = x1 - mean;
    const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
        centered0 * centered0 + centered1 * centered1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();

    float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] = rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

    const float y0 = centered0 * inv_std *
        stream1_transformer_load_scalar_device(gamma, col0, dtype) +
        stream1_transformer_load_scalar_device(beta, col0, dtype);
    const float y1 = centered1 * inv_std *
        stream1_transformer_load_scalar_device(gamma, col1, dtype) +
        stream1_transformer_load_scalar_device(beta, col1, dtype);
    stream1_transformer_store_scalar_device(output, base + col0, y0, dtype);
    stream1_transformer_store_scalar_device(output, base + col1, y1, dtype);
}

__global__ void stream1_transformer_bias_layernorm256_copy_kernel(
    half* __restrict__ input_inout,
    half* __restrict__ output,
    const half* __restrict__ bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint32_t rows,
    std::uint32_t dtype) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows || tid >= 128U) {
        return;
    }

    extern __shared__ float warp_scratch[];
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * 256ULL;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    const float x0 = stream1_transformer_load_scalar_device(input_inout, base + col0, dtype) +
        stream1_transformer_load_scalar_device(bias, col0, dtype);
    const float x1 = stream1_transformer_load_scalar_device(input_inout, base + col1, dtype) +
        stream1_transformer_load_scalar_device(bias, col1, dtype);

    stream1_transformer_store_scalar_device(input_inout, base + col0, x0, dtype);
    stream1_transformer_store_scalar_device(input_inout, base + col1, x1, dtype);

    const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
    __syncthreads();
#endif

    const float centered0 = x0 - mean;
    const float centered1 = x1 - mean;
    const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
        centered0 * centered0 + centered1 * centered1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();

    float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] = rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

    const float y0 = centered0 * inv_std *
        stream1_transformer_load_scalar_device(gamma, col0, dtype) +
        stream1_transformer_load_scalar_device(beta, col0, dtype);
    const float y1 = centered1 * inv_std *
        stream1_transformer_load_scalar_device(gamma, col1, dtype) +
        stream1_transformer_load_scalar_device(beta, col1, dtype);
    stream1_transformer_store_scalar_device(output, base + col0, y0, dtype);
    stream1_transformer_store_scalar_device(output, base + col1, y1, dtype);
}

__device__ __forceinline__ float stream1_transformer_round_to_model_dtype_device(float value, std::uint32_t dtype) {
    if (dtype == STREAM1_DTYPE_BF16) {
        return __bfloat162float(__float2bfloat16(value));
    }
    return __half2float(__float2half(value));
}

__global__ void stream1_transformer_bias_round_layernorm256_copy_kernel(
    half* __restrict__ input_inout,
    half* __restrict__ output,
    const half* __restrict__ bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint32_t rows,
    std::uint32_t dtype) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows || tid >= 128U) {
        return;
    }

    extern __shared__ float warp_scratch[];
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * 256ULL;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;

    const float biased0 = stream1_transformer_load_scalar_device(input_inout, base + col0, dtype) +
        stream1_transformer_load_scalar_device(bias, col0, dtype);
    const float biased1 = stream1_transformer_load_scalar_device(input_inout, base + col1, dtype) +
        stream1_transformer_load_scalar_device(bias, col1, dtype);
    const float x0 = stream1_transformer_round_to_model_dtype_device(biased0, dtype);
    const float x1 = stream1_transformer_round_to_model_dtype_device(biased1, dtype);

    stream1_transformer_store_scalar_device(input_inout, base + col0, x0, dtype);
    stream1_transformer_store_scalar_device(input_inout, base + col1, x1, dtype);

    const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
    __syncthreads();
#endif

    const float centered0 = x0 - mean;
    const float centered1 = x1 - mean;
    const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
        centered0 * centered0 + centered1 * centered1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();

    float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] = rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

    const float y0 = centered0 * inv_std *
        stream1_transformer_load_scalar_device(gamma, col0, dtype) +
        stream1_transformer_load_scalar_device(beta, col0, dtype);
    const float y1 = centered1 * inv_std *
        stream1_transformer_load_scalar_device(gamma, col1, dtype) +
        stream1_transformer_load_scalar_device(beta, col1, dtype);
    stream1_transformer_store_scalar_device(output, base + col0, y0, dtype);
    stream1_transformer_store_scalar_device(output, base + col1, y1, dtype);
}
__global__ void stream1_transformer_bias_round_layernorm256_copy_persistent_kernel(
    half* __restrict__ input_inout,
    half* __restrict__ output,
    const half* __restrict__ bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint32_t rows,
    std::uint32_t dtype) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    extern __shared__ float warp_scratch[];


    for (std::uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const std::uint64_t base = static_cast<std::uint64_t>(row) * 256ULL;
        const float biased0 = stream1_transformer_load_scalar_device(input_inout, base + col0, dtype) +
            stream1_transformer_load_scalar_device(bias, col0, dtype);
        const float biased1 = stream1_transformer_load_scalar_device(input_inout, base + col1, dtype) +
            stream1_transformer_load_scalar_device(bias, col1, dtype);
        const float x0 = stream1_transformer_round_to_model_dtype_device(biased0, dtype);
        const float x1 = stream1_transformer_round_to_model_dtype_device(biased1, dtype);
        stream1_transformer_store_scalar_device(input_inout, base + col0, x0, dtype);
        stream1_transformer_store_scalar_device(input_inout, base + col1, x1, dtype);

        const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
        if (lane == 0U) {
            warp_scratch[warp] = warp_sum;
        }
        __syncthreads();

        float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
        if (warp == 0U) {
            block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
            if (lane == 0U) {
                warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
            }
        }
        __syncthreads();
        const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
        __syncthreads();
#endif

        const float centered0 = x0 - mean;
        const float centered1 = x1 - mean;
        const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
            centered0 * centered0 + centered1 * centered1);
        if (lane == 0U) {
            warp_scratch[warp] = warp_var_sum;
        }
        __syncthreads();

        float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
        if (warp == 0U) {
            block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
            if (lane == 0U) {
                warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] =
                    rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
            }
        }
        __syncthreads();
        const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

        const float y0 = centered0 * inv_std *
            stream1_transformer_load_scalar_device(gamma, col0, dtype) +
            stream1_transformer_load_scalar_device(beta, col0, dtype);
        const float y1 = centered1 * inv_std *
            stream1_transformer_load_scalar_device(gamma, col1, dtype) +
            stream1_transformer_load_scalar_device(beta, col1, dtype);
        stream1_transformer_store_scalar_device(output, base + col0, y0, dtype);
        stream1_transformer_store_scalar_device(output, base + col1, y1, dtype);
        __syncthreads();
    }
}
__global__ void stream1_transformer_bias_round_layernorm256_copy_block2_kernel(
    half* __restrict__ input_inout,
    half* __restrict__ output,
    const half* __restrict__ bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint32_t rows,
    std::uint32_t dtype) {
    const std::uint32_t row_slot = threadIdx.x >> 7U;
    const std::uint32_t tid = threadIdx.x & 127U;
    const std::uint32_t row = blockIdx.x * 2U + row_slot;
    const bool active = row < rows;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint64_t base = static_cast<std::uint64_t>(row) * 256ULL;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    extern __shared__ float shared_scratch[];
    float* const warp_scratch =
        shared_scratch + row_slot * STREAM1_TRANSFORMER_LN256_SHARED_FLOATS;

    float x0 = 0.0f;
    float x1 = 0.0f;
    if (active) {
        const float biased0 = stream1_transformer_load_scalar_device(input_inout, base + col0, dtype) +
            stream1_transformer_load_scalar_device(bias, col0, dtype);
        const float biased1 = stream1_transformer_load_scalar_device(input_inout, base + col1, dtype) +
            stream1_transformer_load_scalar_device(bias, col1, dtype);
        x0 = stream1_transformer_round_to_model_dtype_device(biased0, dtype);
        x1 = stream1_transformer_round_to_model_dtype_device(biased1, dtype);
        stream1_transformer_store_scalar_device(input_inout, base + col0, x0, dtype);
        stream1_transformer_store_scalar_device(input_inout, base + col1, x1, dtype);
    }

    const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_sum;
    }
    __syncthreads();

    float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
        }
    }
    __syncthreads();
    const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
    __syncthreads();
#endif

    const float centered0 = x0 - mean;
    const float centered1 = x1 - mean;
    const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
        centered0 * centered0 + centered1 * centered1);
    if (lane == 0U) {
        warp_scratch[warp] = warp_var_sum;
    }
    __syncthreads();

    float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
    if (warp == 0U) {
        block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
        if (lane == 0U) {
            warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] =
                rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
        }
    }
    __syncthreads();
    const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

    if (active) {
        const float y0 = centered0 * inv_std *
            stream1_transformer_load_scalar_device(gamma, col0, dtype) +
            stream1_transformer_load_scalar_device(beta, col0, dtype);
        const float y1 = centered1 * inv_std *
            stream1_transformer_load_scalar_device(gamma, col1, dtype) +
            stream1_transformer_load_scalar_device(beta, col1, dtype);
        stream1_transformer_store_scalar_device(output, base + col0, y0, dtype);
        stream1_transformer_store_scalar_device(output, base + col1, y1, dtype);
    }
}
__global__ void stream1_transformer_layernorm256_copy_persistent_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    std::uint32_t rows,
    std::uint32_t dtype) {
    const std::uint32_t tid = threadIdx.x;
    const std::uint32_t lane = tid & 31U;
    const std::uint32_t warp = tid >> 5U;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    extern __shared__ float warp_scratch[];

    const float gamma0 = stream1_transformer_load_scalar_device(gamma, col0, dtype);
    const float gamma1 = stream1_transformer_load_scalar_device(gamma, col1, dtype);
    const float beta0 = stream1_transformer_load_scalar_device(beta, col0, dtype);
    const float beta1 = stream1_transformer_load_scalar_device(beta, col1, dtype);

    for (std::uint32_t row = blockIdx.x; row < rows; row += gridDim.x) {
        const std::uint64_t base = static_cast<std::uint64_t>(row) * 256ULL;
        const float x0 = stream1_transformer_load_scalar_device(input, base + col0, dtype);
        const float x1 = stream1_transformer_load_scalar_device(input, base + col1, dtype);

        const float warp_sum = stream1_transformer_warp_reduce_sum_device(x0 + x1);
        if (lane == 0U) {
            warp_scratch[warp] = warp_sum;
        }
        __syncthreads();

        float block_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
        if (warp == 0U) {
            block_sum = stream1_transformer_warp_reduce_sum_device(block_sum);
            if (lane == 0U) {
                warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT] = block_sum * (1.0f / 256.0f);
            }
        }
        __syncthreads();
        const float mean = warp_scratch[STREAM1_TRANSFORMER_LN256_MEAN_SLOT];
#if !STREAM1_TRANSFORMER_LN256_SPLIT_SLOTS
        __syncthreads();
#endif

        const float centered0 = x0 - mean;
        const float centered1 = x1 - mean;
        const float warp_var_sum = stream1_transformer_warp_reduce_sum_device(
            centered0 * centered0 + centered1 * centered1);
        if (lane == 0U) {
            warp_scratch[warp] = warp_var_sum;
        }
        __syncthreads();

        float block_var_sum = tid < 4U ? warp_scratch[tid] : 0.0f;
        if (warp == 0U) {
            block_var_sum = stream1_transformer_warp_reduce_sum_device(block_var_sum);
            if (lane == 0U) {
                warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT] =
                    rsqrtf(block_var_sum * (1.0f / 256.0f) + 1.0e-5f);
            }
        }
        __syncthreads();
        const float inv_std = warp_scratch[STREAM1_TRANSFORMER_LN256_INV_STD_SLOT];

        const float y0 = __fadd_rn(__fmul_rn(__fmul_rn(centered0, inv_std), gamma0), beta0);
        const float y1 = __fadd_rn(__fmul_rn(__fmul_rn(centered1, inv_std), gamma1), beta1);
        stream1_transformer_store_scalar_device(output, base + col0, y0, dtype);
        stream1_transformer_store_scalar_device(output, base + col1, y1, dtype);
        __syncthreads();
    }
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
        const Stream1TransformerLayerNormRowsPolicy policy =
            parse_stream1_transformer_layernorm_rows_policy(
                std::getenv("BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY"));
        if (!stream1_transformer_layernorm_copy_policy_supported(policy)) {
            throw std::invalid_argument("LayerNorm copy path supports only row or persistent policy");
        }
        if (policy == Stream1TransformerLayerNormRowsPolicy::PersistentRows) {
            constexpr int threads = 128;
            const std::size_t shared_bytes = STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float);
            int active_blocks_per_sm = 0;
            BEAM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &active_blocks_per_sm,
                stream1_transformer_layernorm256_copy_persistent_kernel,
                threads,
                shared_bytes));
            int device = 0;
            cudaDeviceProp prop{};
            BEAM_CUDA_CHECK(cudaGetDevice(&device));
            BEAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
            const int blocks_per_sm = stream1_transformer_layernorm_persistent_blocks_per_sm(
                std::getenv("BEAM_STREAM1_TRANSFORMER_LAYERNORM_PERSISTENT_BLOCKS_PER_SM"),
                active_blocks_per_sm);
            const std::uint32_t resident_grid = static_cast<std::uint32_t>(
                blocks_per_sm * prop.multiProcessorCount);
            const std::uint32_t grid = rows < resident_grid ? rows : resident_grid;
            stream1_transformer_layernorm256_copy_persistent_kernel<<<
                grid, threads, shared_bytes, stream>>>(
                input,
                output,
                gamma,
                beta,
                rows,
                dtype);
            return;
        }
        stream1_transformer_layernorm256_copy_kernel<<<rows, 128, STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float), stream>>>(
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

void stream1_transformer_bias_layernorm_copy_launch(
    half* input_inout,
    half* output,
    const half* bias,
    const half* gamma,
    const half* beta,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    if (cols != 256U) {
        throw std::invalid_argument("Stream1 piece_transformer fused bias LayerNorm requires cols=256");
    }
    stream1_transformer_bias_layernorm256_copy_kernel<<<rows, 128, STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float), stream>>>(
        input_inout,
        output,
        bias,
        gamma,
        beta,
        rows,
        dtype);
}

void stream1_transformer_bias_round_layernorm_copy_launch(
    half* input_inout,
    half* output,
    const half* bias,
    const half* gamma,
    const half* beta,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    if (cols != 256U) {
        throw std::invalid_argument("Stream1 piece_transformer rounded fused bias LayerNorm requires cols=256");
    }
    const Stream1TransformerLayerNormRowsPolicy policy =
        parse_stream1_transformer_layernorm_rows_policy(
            std::getenv("BEAM_STREAM1_TRANSFORMER_LAYERNORM_ROWS_POLICY"));
    if (policy == Stream1TransformerLayerNormRowsPolicy::PersistentRows) {
        constexpr int threads = 128;
        const std::size_t shared_bytes = STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float);
        int active_blocks_per_sm = 0;
        BEAM_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks_per_sm,
            stream1_transformer_bias_round_layernorm256_copy_persistent_kernel,
            threads,
            shared_bytes));
        int device = 0;
        cudaDeviceProp prop{};
        BEAM_CUDA_CHECK(cudaGetDevice(&device));
        BEAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        const std::uint32_t resident_grid = static_cast<std::uint32_t>(
            active_blocks_per_sm * prop.multiProcessorCount);
        const std::uint32_t grid = rows < resident_grid ? rows : resident_grid;
        stream1_transformer_bias_round_layernorm256_copy_persistent_kernel<<<
            grid, threads, shared_bytes, stream>>>(
            input_inout,
            output,
            bias,
            gamma,
            beta,
            rows,
            dtype);
        return;
    }
    if (policy == Stream1TransformerLayerNormRowsPolicy::TwoRowsPerBlock) {
        stream1_transformer_bias_round_layernorm256_copy_block2_kernel<<<
            (rows + 1U) / 2U,
            256,
            2U * STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float),
            stream>>>(
            input_inout,
            output,
            bias,
            gamma,
            beta,
            rows,
            dtype);
        return;
    }

    stream1_transformer_bias_round_layernorm256_copy_kernel<<<rows, 128, STREAM1_TRANSFORMER_LN256_SHARED_FLOATS * sizeof(float), stream>>>(
        input_inout,
        output,
        bias,
        gamma,
        beta,
        rows,
        dtype);
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
    const float x = stream1_transformer_load_scalar_device(matrix, i, dtype) +
        stream1_transformer_load_scalar_device(bias, col, dtype);
    const float y = x / (1.0f + expf(-x));
    stream1_transformer_store_scalar_device(matrix, i, y, dtype);
}

__global__ void stream1_transformer_bias_add_kernel(
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
    const float x = stream1_transformer_load_scalar_device(matrix, i, dtype) +
        stream1_transformer_load_scalar_device(bias, col, dtype);
    stream1_transformer_store_scalar_device(matrix, i, x, dtype);
}

__global__ void stream1_transformer_bias_add256_fp16_kernel(
    half* __restrict__ matrix,
    const half* __restrict__ bias,
    std::uint32_t rows) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= rows || tid >= 128U) {
        return;
    }
    const std::uint64_t idx = static_cast<std::uint64_t>(row) * 256ULL + static_cast<std::uint64_t>(tid) * 2ULL;
    const half2 x = *reinterpret_cast<const half2*>(matrix + idx);
    const half2 b = *reinterpret_cast<const half2*>(bias + static_cast<std::uint64_t>(tid) * 2ULL);
    *reinterpret_cast<half2*>(matrix + idx) = __hadd2(x, b);
}

void stream1_transformer_bias_add_launch(
    half* matrix,
    const half* bias,
    std::uint32_t rows,
    std::uint32_t cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    if (cols == 256U && dtype == STREAM1_DTYPE_FP16) {
        stream1_transformer_bias_add256_fp16_kernel<<<rows, 128, 0, stream>>>(matrix, bias, rows);
        return;
    }
    stream1_transformer_bias_add_kernel<<<(rows * cols + 255U) / 256U, 256, 0, stream>>>(
        matrix,
        bias,
        rows,
        cols,
        dtype);
}
constexpr std::uint32_t STREAM1_TRANSFORMER_SEQ51 = 51U;
constexpr std::uint32_t STREAM1_TRANSFORMER_DMODEL256 = 256U;
constexpr std::uint32_t STREAM1_TRANSFORMER_HEAD_DIM32 = 32U;
constexpr std::uint32_t STREAM1_TRANSFORMER_NHEAD8 = 8U;
constexpr std::uint32_t STREAM1_TRANSFORMER_QKV_STRIDE51 = 3U * STREAM1_TRANSFORMER_DMODEL256;
constexpr std::uint32_t STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51 = 64U;
constexpr std::uint32_t STREAM1_TRANSFORMER_PROB_STRIDE51 =
    STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51;
constexpr std::uint32_t STREAM1_TRANSFORMER_HEAD_OUTPUT_STRIDE51 =
    STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51 * STREAM1_TRANSFORMER_HEAD_DIM32;
constexpr std::uint32_t STREAM1_TRANSFORMER_SCORE_STRIDE51 =
    STREAM1_TRANSFORMER_PROB_STRIDE51 + STREAM1_TRANSFORMER_HEAD_OUTPUT_STRIDE51;


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
        const std::uint64_t query_base = matrix_base +
            static_cast<std::uint64_t>(query) * STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51;
        float local_max = -3.4028234663852886e38f;
        for (std::uint32_t key = lane; key < STREAM1_TRANSFORMER_SEQ51; key += 32U) {
            local_max = fmaxf(local_max, stream1_transformer_load_scalar_device(scores_probs, query_base + key, dtype));
        }
        const float max_score = stream1_transformer_warp_reduce_max_device(local_max);
        float local_sum = 0.0f;
        float local_values[2] = {0.0f, 0.0f};
        std::uint32_t local_keys[2] = {0xffffffffU, 0xffffffffU};
        std::uint32_t slot = 0U;
        for (std::uint32_t key = lane; key < STREAM1_TRANSFORMER_SEQ51; key += 32U) {
            const float exp_value = expf(stream1_transformer_load_scalar_device(scores_probs, query_base + key, dtype) - max_score);
            local_sum += exp_value;
            local_values[slot] = exp_value;
            local_keys[slot] = key;
            ++slot;
        }
        const float inv_sum = 1.0f / stream1_transformer_warp_reduce_sum_device(local_sum);
        for (std::uint32_t i = 0; i < slot; ++i) {
            stream1_transformer_store_scalar_device(scores_probs, query_base + local_keys[i], local_values[i] * inv_sum, dtype);
        }
        for (std::uint32_t key = STREAM1_TRANSFORMER_SEQ51 + lane;
             key < STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51;
             key += 32U) {
            stream1_transformer_store_scalar_device(scores_probs, query_base + key, 0.0f, dtype);
        }
    }
}
__global__ void stream1_transformer_pack_v51_kernel(
    const half* __restrict__ qkv,
    half* __restrict__ scores_probs,
    std::uint32_t dtype,
    std::uint32_t b_micro) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::uint64_t per_matrix = static_cast<std::uint64_t>(STREAM1_TRANSFORMER_HEAD_OUTPUT_STRIDE51);
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
        value = stream1_transformer_load_scalar_device(qkv, qkv_idx, dtype);
    }
    stream1_transformer_store_scalar_device(
        scores_probs,
        matrix_base + STREAM1_TRANSFORMER_PROB_STRIDE51 +
            static_cast<std::uint64_t>(key) * STREAM1_TRANSFORMER_HEAD_DIM32 + lane,
        value,
        dtype);
}
enum class Stream1TransformerAttentionBackend : std::uint32_t {
    Sm75Fp16Fmha,
    Sm80Fp16Fmha,
    Sm80Bf16Fmha,
};

Stream1TransformerAttentionBackend stream1_transformer_select_attention_backend(std::uint32_t dtype) {
    int device = 0;
    BEAM_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    BEAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
#if BEAM_HAS_CUTLASS_FMHA
    if (dtype == STREAM1_DTYPE_BF16) {
        if (prop.major >= 8) {
            return Stream1TransformerAttentionBackend::Sm80Bf16Fmha;
        }
        throw std::invalid_argument("Stream1 piece_transformer bf16 attention requires SM80+; export fp16 weights for T4/SM75");
    }
    if (dtype == STREAM1_DTYPE_FP16) {
        if (prop.major >= 8) {
            return Stream1TransformerAttentionBackend::Sm80Fp16Fmha;
        }
        if (prop.major > 7 || (prop.major == 7 && prop.minor >= 5)) {
            return Stream1TransformerAttentionBackend::Sm75Fp16Fmha;
        }
        throw std::invalid_argument("Stream1 piece_transformer fp16 FMHA requires SM75+ GPU");
    }
#else
    if (dtype == STREAM1_DTYPE_BF16) {
        throw std::invalid_argument("Stream1 piece_transformer bf16 requires CUTLASS FMHA example headers for SM80+");
    }
    if (dtype == STREAM1_DTYPE_FP16) {
        if (prop.major > 7 || (prop.major == 7 && prop.minor >= 5)) {
            throw std::invalid_argument("Stream1 piece_transformer fp16 requires CUTLASS FMHA example headers for SDPA-style attention");
        }
        throw std::invalid_argument("Stream1 piece_transformer fp16 attention requires SM75+ GPU");
    }
#endif
    throw std::invalid_argument("Stream1 piece_transformer dtype must be fp16 or bf16");
}

#if BEAM_HAS_CUTLASS
bool stream1_transformer_current_device_sm80_or_newer() {
    int device = 0;
    BEAM_CUDA_CHECK(cudaGetDevice(&device));
    static thread_local int cached_device = -1;
    static thread_local bool cached_sm80 = false;
    if (cached_device == device) {
        return cached_sm80;
    }
    cudaDeviceProp prop{};
    BEAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cached_device = device;
    cached_sm80 = prop.major >= 8;
    return cached_sm80;
}

template <
    typename Element,
    typename ArchTag,
    typename InstructionShape,
    typename ThreadblockShape,
    typename WarpShape>
void stream1_transformer_linear_residual_typed(
    const half* input,
    const half* weight,
    half* residual_inout,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream) {
    const cutlass::gemm::GemmCoord problem(static_cast<int>(rows), static_cast<int>(output_cols), static_cast<int>(input_cols));
    using Gemm = cutlass::gemm::device::Gemm<
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape>;
    Gemm gemm;
    typename Gemm::Arguments args{
        problem,
        {reinterpret_cast<Element const*>(input), static_cast<int>(input_cols)},
        {reinterpret_cast<Element const*>(weight), static_cast<int>(output_cols)},
        {reinterpret_cast<Element const*>(residual_inout), static_cast<int>(output_cols)},
        {reinterpret_cast<Element*>(residual_inout), static_cast<int>(output_cols)},
        {1.0f, 1.0f}};
    const cutlass::Status status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Stream1 piece_transformer residual Gemm launch failed");
    }
}

void stream1_transformer_linear_residual_cuda(
    const half* input,
    const half* weight,
    half* residual_inout,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    if (dtype == STREAM1_DTYPE_BF16) {
        int device = 0;
        cudaDeviceProp prop{};
        BEAM_CUDA_CHECK(cudaGetDevice(&device));
        BEAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 8) {
            throw std::invalid_argument("Stream1 piece_transformer bf16 residual GEMM requires SM80+");
        }
        stream1_transformer_linear_residual_typed<
            cutlass::bfloat16_t,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>>(input, weight, residual_inout, rows, input_cols, output_cols, stream);
        return;
    }
    if (dtype == STREAM1_DTYPE_FP16) {
        if (stream1_transformer_current_device_sm80_or_newer()) {
            const bool is_ff2 = input_cols == 1024U && output_cols == 256U;
            const Stream1TransformerGemmFamily family = is_ff2
                ? Stream1TransformerGemmFamily::Ff2
                : Stream1TransformerGemmFamily::AttentionOut;
            const char* env_name = is_ff2
                ? "BEAM_STREAM1_TRANSFORMER_FF2_POLICY"
                : "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY";
            const Stream1TransformerGemmPolicy policy = parse_stream1_transformer_gemm_policy(
                family, std::getenv(env_name));
            if (policy == Stream1TransformerGemmPolicy::M64N64) {
                stream1_transformer_linear_residual_typed<
                    cutlass::half_t, cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<64, 64, 32>,
                    cutlass::gemm::GemmShape<32, 32, 32>>(input, weight, residual_inout, rows, input_cols, output_cols, stream);
                return;
            }
            if (policy == Stream1TransformerGemmPolicy::M128N128) {
                stream1_transformer_linear_residual_typed<
                    cutlass::half_t, cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<64, 64, 32>>(input, weight, residual_inout, rows, input_cols, output_cols, stream);
                return;
            }
            stream1_transformer_linear_residual_typed<
                cutlass::half_t,
                cutlass::arch::Sm80,
                cutlass::gemm::GemmShape<16, 8, 16>,
                cutlass::gemm::GemmShape<128, 64, 32>,
                cutlass::gemm::GemmShape<64, 32, 32>>(input, weight, residual_inout, rows, input_cols, output_cols, stream);
            return;
        }
        const bool is_ff2 = input_cols == 1024U && output_cols == 256U;
        const Stream1TransformerGemmFamily family = is_ff2
            ? Stream1TransformerGemmFamily::Ff2
            : Stream1TransformerGemmFamily::AttentionOut;
        const char* env_name = is_ff2
            ? "BEAM_STREAM1_TRANSFORMER_FF2_POLICY"
            : "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY";
        const Stream1TransformerGemmPolicy policy = parse_stream1_transformer_gemm_policy(
            family, std::getenv(env_name));
        if (!stream1_transformer_gemm_policy_supported_on_sm(family, policy, 75)) {
            throw std::invalid_argument("Stream1 transformer residual GEMM policy is not compiled for SM75");
        }
        if (policy == Stream1TransformerGemmPolicy::M64N64) {
            stream1_transformer_linear_residual_typed<
                cutlass::half_t, cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<64, 64, 32>,
                cutlass::gemm::GemmShape<32, 32, 32>>(
                    input, weight, residual_inout, rows, input_cols, output_cols, stream);
            return;
        }
        if (policy == Stream1TransformerGemmPolicy::M128N128) {
            stream1_transformer_linear_residual_typed<
                cutlass::half_t, cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<128, 128, 32>,
                cutlass::gemm::GemmShape<64, 64, 32>>(
                    input, weight, residual_inout, rows, input_cols, output_cols, stream);
            return;
        }
        stream1_transformer_linear_residual_typed<
            cutlass::half_t,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>>(
                input, weight, residual_inout, rows, input_cols, output_cols, stream);
        return;
    }
    throw std::invalid_argument("Stream1 piece_transformer residual GEMM dtype must be fp16 or bf16");
}

template <typename Element>
struct Stream1TransformerRoundThenAdd {
    CUTLASS_HOST_DEVICE
    float operator()(float value, float bias) const {
        const Element rounded = cutlass::NumericConverter<Element, float>()(value);
        const float rounded_value = cutlass::NumericConverter<float, Element>()(rounded);
        return rounded_value + bias;
    }
};

template <
    typename Element,
    typename ArchTag,
    typename InstructionShape,
    typename ThreadblockShape,
    typename WarpShape,
    int Swizzle = 1,
    int Stages = 3>
void stream1_transformer_linear_residual_bias_round_typed(
    const half* input,
    const half* weight,
    half* residual_inout,
    const half* bias,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream) {
    constexpr int elements_per_access = 128 / cutlass::sizeof_bits<Element>::value;
    using Epilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        Element,
        float,
        float,
        Element,
        Element,
        elements_per_access,
        cutlass::epilogue::thread::Identity<float>,
        Stream1TransformerRoundThenAdd<Element>,
        false,
        Element>;
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
        Stages>;

    const cutlass::gemm::GemmCoord problem(
        static_cast<int>(rows),
        static_cast<int>(output_cols),
        static_cast<int>(input_cols));
    typename Epilogue::Params epilogue_params{1.0f, 1.0f};
    const std::int64_t batch_stride_a = static_cast<std::int64_t>(rows) * input_cols;
    const std::int64_t batch_stride_b = static_cast<std::int64_t>(input_cols) * output_cols;
    const std::int64_t batch_stride_output = static_cast<std::int64_t>(rows) * output_cols;
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        1,
        epilogue_params,
        reinterpret_cast<Element const*>(input),
        reinterpret_cast<Element const*>(weight),
        reinterpret_cast<Element const*>(residual_inout),
        reinterpret_cast<Element*>(residual_inout),
        const_cast<Element*>(reinterpret_cast<Element const*>(bias)),
        nullptr,
        batch_stride_a,
        batch_stride_b,
        batch_stride_output,
        batch_stride_output,
        0,
        0,
        static_cast<int>(input_cols),
        static_cast<int>(output_cols),
        static_cast<int>(output_cols),
        static_cast<int>(output_cols),
        0,
        static_cast<int>(output_cols));

    Gemm gemm;
    const cutlass::Status status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Stream1 piece_transformer fused residual+bias-round GEMM launch failed");
    }
}

void stream1_transformer_residual_bias_round_layernorm_cuda(
    const half* input,
    const half* weight,
    half* residual_inout,
    const half* bias,
    half* normalized_output,
    const half* gamma,
    const half* beta,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    const bool is_ff2 = input_cols == 1024U && output_cols == 256U;

    const Stream1TransformerGemmFamily family = is_ff2
        ? Stream1TransformerGemmFamily::Ff2
        : Stream1TransformerGemmFamily::AttentionOut;
    const char* epilogue_env = is_ff2
        ? "BEAM_STREAM1_TRANSFORMER_FF2_EPILOGUE"
        : "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_EPILOGUE";
    const Stream1TransformerResidualEpiloguePolicy epilogue_policy =
        parse_stream1_transformer_residual_epilogue_policy(std::getenv(epilogue_env));
    const char* swizzle_env = is_ff2
        ? "BEAM_STREAM1_TRANSFORMER_FF2_SWIZZLE"
        : "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_SWIZZLE";
    const Stream1TransformerGemmSwizzlePolicy swizzle_policy =
        parse_stream1_transformer_gemm_swizzle_policy(std::getenv(swizzle_env));
    if (epilogue_policy == Stream1TransformerResidualEpiloguePolicy::Separate) {

        if (swizzle_policy != Stream1TransformerGemmSwizzlePolicy::Identity1) {
            throw std::invalid_argument("residual GEMM swizzle greater than 1 requires fused epilogue");
        }
        stream1_transformer_linear_residual_cuda(
            input, weight, residual_inout, rows, input_cols, output_cols, dtype, stream);
        stream1_transformer_bias_round_layernorm_copy_launch(
            residual_inout, normalized_output, bias, gamma, beta, rows, output_cols, dtype, stream);
        return;
    }
    if (dtype != STREAM1_DTYPE_FP16) {
        throw std::invalid_argument("fused residual+bias-round epilogue requires fp16 SM75+");
    }
    if (output_cols != 256U) {
        throw std::invalid_argument("fused residual+bias-round epilogue requires output_cols=256");
    }

    const char* gemm_env = is_ff2
        ? "BEAM_STREAM1_TRANSFORMER_FF2_POLICY"
        : "BEAM_STREAM1_TRANSFORMER_ATTN_OUT_POLICY";
    const Stream1TransformerGemmPolicy gemm_policy =
        parse_stream1_transformer_gemm_policy(family, std::getenv(gemm_env));
    if (gemm_policy != Stream1TransformerGemmPolicy::M128N128) {
        throw std::invalid_argument("fused residual+bias-round epilogue is compiled only for m128n128");
    }
    if (!stream1_transformer_gemm_swizzle_allowed(
            family,
            gemm_policy,
            Stream1TransformerGemmStagePolicy::Stages3,
            swizzle_policy)) {
        throw std::invalid_argument("residual GEMM swizzle is not compiled for this family and policy");
    }
    if (stream1_transformer_current_device_sm80_or_newer()) {
        if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity2) {
            stream1_transformer_linear_residual_bias_round_typed<
                cutlass::half_t,
                cutlass::arch::Sm80,
                cutlass::gemm::GemmShape<16, 8, 16>,
                cutlass::gemm::GemmShape<128, 128, 32>,
                cutlass::gemm::GemmShape<64, 64, 32>,
                2>(input, weight, residual_inout, bias, rows, input_cols, output_cols, stream);
        } else {
            stream1_transformer_linear_residual_bias_round_typed<
                cutlass::half_t,
                cutlass::arch::Sm80,
                cutlass::gemm::GemmShape<16, 8, 16>,
                cutlass::gemm::GemmShape<128, 128, 32>,
                cutlass::gemm::GemmShape<64, 64, 32>>(
                    input, weight, residual_inout, bias, rows, input_cols, output_cols, stream);
        }
    } else if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity2) {
        stream1_transformer_linear_residual_bias_round_typed<
            cutlass::half_t,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>,
            2, 2>(input, weight, residual_inout, bias, rows, input_cols, output_cols, stream);
    } else {
        stream1_transformer_linear_residual_bias_round_typed<
            cutlass::half_t,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::gemm::GemmShape<128, 128, 32>,
            cutlass::gemm::GemmShape<64, 64, 32>,
            1, 2>(
                input, weight, residual_inout, bias, rows, input_cols, output_cols, stream);
    }
    stream1_transformer_layernorm_copy_launch(
        residual_inout,
        normalized_output,
        gamma,
        beta,
        rows,
        output_cols,
        dtype,
        stream);
}

template <
    typename Element,
    typename ArchTag,
    typename InstructionShape,
    typename ThreadblockShape,
    typename WarpShape,
    int Swizzle = 1>
void stream1_transformer_linear_bias_typed(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream) {
    constexpr int elements_per_access = 128 / cutlass::sizeof_bits<Element>::value;
    using Epilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        Element,
        float,
        float,
        Element,
        Element,
        elements_per_access,
        cutlass::epilogue::thread::Identity<float>,
        cutlass::plus<float>,
        false,
        Element>;
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>>;

    const cutlass::gemm::GemmCoord problem(
        static_cast<int>(rows),
        static_cast<int>(output_cols),
        static_cast<int>(input_cols));
    typename Epilogue::Params epilogue_params{1.0f, 0.0f};
    const std::int64_t batch_stride_a = static_cast<std::int64_t>(rows) * input_cols;
    const std::int64_t batch_stride_b = static_cast<std::int64_t>(input_cols) * output_cols;
    const std::int64_t batch_stride_output = static_cast<std::int64_t>(rows) * output_cols;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        1,
        epilogue_params,
        reinterpret_cast<Element const*>(input),
        reinterpret_cast<Element const*>(weight),
        reinterpret_cast<Element const*>(output),
        reinterpret_cast<Element*>(output),
        const_cast<Element*>(reinterpret_cast<Element const*>(bias)),
        nullptr,
        batch_stride_a,
        batch_stride_b,
        batch_stride_output,
        batch_stride_output,
        0,
        0,
        static_cast<int>(input_cols),
        static_cast<int>(output_cols),
        static_cast<int>(output_cols),
        static_cast<int>(output_cols),
        0,
        static_cast<int>(output_cols));

    Gemm gemm;
    const cutlass::Status status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Stream1 piece_transformer linear+bias GEMM launch failed");
    }
}

void stream1_transformer_linear_bias_cuda(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    if (dtype == STREAM1_DTYPE_BF16) {
        int device = 0;
        cudaDeviceProp prop{};
        BEAM_CUDA_CHECK(cudaGetDevice(&device));
        BEAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 8) {
            throw std::invalid_argument("Stream1 piece_transformer bf16 linear+bias GEMM requires SM80+");
        }
        stream1_transformer_linear_bias_typed<
            cutlass::bfloat16_t,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>>(input, weight, bias, output, rows, input_cols, output_cols, stream);
        return;
    }
    if (dtype == STREAM1_DTYPE_FP16) {
        if (stream1_transformer_current_device_sm80_or_newer()) {
            const Stream1TransformerGemmPolicy policy = parse_stream1_transformer_gemm_policy(
                Stream1TransformerGemmFamily::Qkv,
                std::getenv("BEAM_STREAM1_TRANSFORMER_QKV_POLICY"));

            const Stream1TransformerGemmSwizzlePolicy swizzle_policy =
                parse_stream1_transformer_gemm_swizzle_policy(
                    std::getenv("BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE"));
            if (!stream1_transformer_gemm_swizzle_allowed(
                    Stream1TransformerGemmFamily::Qkv,
                    policy,
                    Stream1TransformerGemmStagePolicy::Stages3,
                    swizzle_policy)) {
                throw std::invalid_argument("QKV swizzle is not compiled for this policy and stage count");
            }
            if (policy == Stream1TransformerGemmPolicy::M64N128) {
                stream1_transformer_linear_bias_typed<
                    cutlass::half_t, cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<64, 128, 32>,
                    cutlass::gemm::GemmShape<32, 64, 32>>(input, weight, bias, output, rows, input_cols, output_cols, stream);
                return;
            }
            if (policy == Stream1TransformerGemmPolicy::M128N128) {
                if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity4) {
                    stream1_transformer_linear_bias_typed<
                        cutlass::half_t, cutlass::arch::Sm80,
                        cutlass::gemm::GemmShape<16, 8, 16>,
                        cutlass::gemm::GemmShape<128, 128, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>,
                        4>(input, weight, bias, output, rows, input_cols, output_cols, stream);
                    return;
                }
                if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity8) {
                    stream1_transformer_linear_bias_typed<
                        cutlass::half_t, cutlass::arch::Sm80,
                        cutlass::gemm::GemmShape<16, 8, 16>,
                        cutlass::gemm::GemmShape<128, 128, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>,
                        8>(input, weight, bias, output, rows, input_cols, output_cols, stream);
                    return;
                }
                stream1_transformer_linear_bias_typed<
                    cutlass::half_t, cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<64, 64, 32>>(input, weight, bias, output, rows, input_cols, output_cols, stream);
                return;
            }
            if (policy == Stream1TransformerGemmPolicy::M256N128) {
                stream1_transformer_linear_bias_typed<
                    cutlass::half_t, cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<256, 128, 32>,
                    cutlass::gemm::GemmShape<64, 64, 32>>(input, weight, bias, output, rows, input_cols, output_cols, stream);
                return;
            }
            stream1_transformer_linear_bias_typed<
                cutlass::half_t,
                cutlass::arch::Sm80,
                cutlass::gemm::GemmShape<16, 8, 16>,
                cutlass::gemm::GemmShape<128, 64, 32>,
                cutlass::gemm::GemmShape<64, 32, 32>>(input, weight, bias, output, rows, input_cols, output_cols, stream);
            return;
        }
        const Stream1TransformerGemmPolicy policy = parse_stream1_transformer_gemm_policy(
            Stream1TransformerGemmFamily::Qkv,
            std::getenv("BEAM_STREAM1_TRANSFORMER_QKV_POLICY"));
        const Stream1TransformerGemmSwizzlePolicy swizzle_policy =
            parse_stream1_transformer_gemm_swizzle_policy(
                std::getenv("BEAM_STREAM1_TRANSFORMER_QKV_SWIZZLE"));
        if (!stream1_transformer_gemm_policy_supported_on_sm(
                Stream1TransformerGemmFamily::Qkv, policy, 75) ||
            !stream1_transformer_gemm_swizzle_allowed(
                Stream1TransformerGemmFamily::Qkv,
                policy,
                Stream1TransformerGemmStagePolicy::Stages2,
                swizzle_policy)) {
            throw std::invalid_argument("QKV policy or swizzle is not compiled for SM75");
        }
        if (policy == Stream1TransformerGemmPolicy::M64N128) {
            stream1_transformer_linear_bias_typed<
                cutlass::half_t, cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<64, 128, 32>,
                cutlass::gemm::GemmShape<32, 64, 32>>(
                    input, weight, bias, output, rows, input_cols, output_cols, stream);
            return;
        }
        if (policy == Stream1TransformerGemmPolicy::M128N128) {
            if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity4) {
                stream1_transformer_linear_bias_typed<
                    cutlass::half_t, cutlass::arch::Sm75,
                    cutlass::gemm::GemmShape<16, 8, 8>,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<64, 64, 32>, 4>(
                        input, weight, bias, output, rows, input_cols, output_cols, stream);
                return;
            }
            if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity8) {
                stream1_transformer_linear_bias_typed<
                    cutlass::half_t, cutlass::arch::Sm75,
                    cutlass::gemm::GemmShape<16, 8, 8>,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<64, 64, 32>, 8>(
                        input, weight, bias, output, rows, input_cols, output_cols, stream);
                return;
            }
            stream1_transformer_linear_bias_typed<
                cutlass::half_t, cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<128, 128, 32>,
                cutlass::gemm::GemmShape<64, 64, 32>>(
                    input, weight, bias, output, rows, input_cols, output_cols, stream);
            return;
        }
        if (policy == Stream1TransformerGemmPolicy::M256N128) {
            stream1_transformer_linear_bias_typed<
                cutlass::half_t, cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<256, 128, 32>,
                cutlass::gemm::GemmShape<64, 64, 32>>(
                    input, weight, bias, output, rows, input_cols, output_cols, stream);
            return;
        }
        stream1_transformer_linear_bias_typed<
            cutlass::half_t,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>>(
                input, weight, bias, output, rows, input_cols, output_cols, stream);
        return;
    }
    throw std::invalid_argument("Stream1 piece_transformer linear+bias GEMM dtype must be fp16 or bf16");
}
template <

    typename Element,
    typename ArchTag,
    typename InstructionShape,
    typename ThreadblockShape,
    typename WarpShape,
    int Stages,
    int Swizzle = 1>
void stream1_transformer_ff1_linear_bias_silu_typed(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream) {
    constexpr int elements_per_access = 128 / cutlass::sizeof_bits<Element>::value;
    using Epilogue = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
        Element,
        float,
        float,
        Element,
        Element,
        elements_per_access,
        cutlass::epilogue::thread::SiLu<float>,
        cutlass::plus<float>,
        false,
        Element>;
    using Gemm = cutlass::gemm::device::GemmUniversalWithBroadcast<
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        Element,
        cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        ArchTag,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
        Stages>;

    const cutlass::gemm::GemmCoord problem(
        static_cast<int>(rows),
        static_cast<int>(output_cols),
        static_cast<int>(input_cols));
    typename Epilogue::Params epilogue_params{1.0f, 0.0f};
    const std::int64_t batch_stride_a = static_cast<std::int64_t>(rows) * input_cols;
    const std::int64_t batch_stride_b = static_cast<std::int64_t>(input_cols) * output_cols;
    const std::int64_t batch_stride_output = static_cast<std::int64_t>(rows) * output_cols;

    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        problem,
        1,
        epilogue_params,
        reinterpret_cast<Element const*>(input),
        reinterpret_cast<Element const*>(weight),
        reinterpret_cast<Element const*>(output),
        reinterpret_cast<Element*>(output),
        const_cast<Element*>(reinterpret_cast<Element const*>(bias)),
        nullptr,
        batch_stride_a,
        batch_stride_b,
        batch_stride_output,
        batch_stride_output,
        0,
        0,
        static_cast<int>(input_cols),
        static_cast<int>(output_cols),
        static_cast<int>(output_cols),
        static_cast<int>(output_cols),
        0,
        static_cast<int>(output_cols));

    Gemm gemm;
    const cutlass::Status status = gemm(args, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("Stream1 piece_transformer FF1 fused GEMM launch failed");
    }
}

template <
    typename Element,
    typename ArchTag,
    typename InstructionShape,
    typename ThreadblockShape,
    typename WarpShape>
void stream1_transformer_ff1_linear_bias_silu_policy(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    cudaStream_t stream,
    Stream1TransformerGemmStagePolicy stage_policy) {

    if (stage_policy == Stream1TransformerGemmStagePolicy::Stages2) {
        stream1_transformer_ff1_linear_bias_silu_typed<
            Element, ArchTag, InstructionShape, ThreadblockShape, WarpShape, 2>(
                input, weight, bias, output, rows, input_cols, output_cols, stream);
        return;
    }
    stream1_transformer_ff1_linear_bias_silu_typed<
        Element, ArchTag, InstructionShape, ThreadblockShape, WarpShape, 3>(
            input, weight, bias, output, rows, input_cols, output_cols, stream);
}

void stream1_transformer_ff1_linear_bias_silu_cuda(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    std::uint32_t rows,
    std::uint32_t input_cols,
    std::uint32_t output_cols,
    std::uint32_t dtype,
    cudaStream_t stream) {
    if (dtype == STREAM1_DTYPE_BF16) {
        int device = 0;
        cudaDeviceProp prop{};
        BEAM_CUDA_CHECK(cudaGetDevice(&device));
        BEAM_CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
        if (prop.major < 8) {
            throw std::invalid_argument("Stream1 piece_transformer bf16 FF1 fused GEMM requires SM80+");
        }
        stream1_transformer_ff1_linear_bias_silu_typed<
            cutlass::bfloat16_t,
            cutlass::arch::Sm80,
            cutlass::gemm::GemmShape<16, 8, 16>,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>, 3>(input, weight, bias, output, rows, input_cols, output_cols, stream);
        return;
    }
    if (dtype == STREAM1_DTYPE_FP16) {
        if (stream1_transformer_current_device_sm80_or_newer()) {
            const Stream1TransformerGemmPolicy policy = parse_stream1_transformer_gemm_policy(
                Stream1TransformerGemmFamily::Ff1,
                std::getenv("BEAM_STREAM1_TRANSFORMER_FF1_POLICY"));
            const Stream1TransformerGemmStagePolicy stage_policy = parse_stream1_transformer_gemm_stage_policy(
                std::getenv("BEAM_STREAM1_TRANSFORMER_FF1_STAGES"));
            const Stream1TransformerGemmSwizzlePolicy swizzle_policy =
                parse_stream1_transformer_gemm_swizzle_policy(
                    std::getenv("BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE"));
            if (!stream1_transformer_gemm_swizzle_allowed(
                    Stream1TransformerGemmFamily::Ff1, policy, stage_policy, swizzle_policy)) {
                throw std::invalid_argument("FF1 swizzle is not compiled for this policy and stage count");
            }

            if (policy == Stream1TransformerGemmPolicy::M64N128) {
                stream1_transformer_ff1_linear_bias_silu_policy<
                    cutlass::half_t,
                    cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<64, 128, 32>,
                    cutlass::gemm::GemmShape<32, 64, 32>>(
                        input, weight, bias, output, rows, input_cols, output_cols, stream, stage_policy);
                return;
            }
            if (policy == Stream1TransformerGemmPolicy::M128N128W64N32) {
                stream1_transformer_ff1_linear_bias_silu_policy<
                    cutlass::half_t,
                    cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<64, 32, 32>>(
                        input, weight, bias, output, rows, input_cols, output_cols, stream, stage_policy);
                return;
            }
            if (policy == Stream1TransformerGemmPolicy::M128N128) {
                if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity4) {
                    stream1_transformer_ff1_linear_bias_silu_typed<
                        cutlass::half_t,
                        cutlass::arch::Sm80,
                        cutlass::gemm::GemmShape<16, 8, 16>,
                        cutlass::gemm::GemmShape<128, 128, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>,
                        3,
                        4>(input, weight, bias, output, rows, input_cols, output_cols, stream);
                    return;
                }
                if (swizzle_policy == Stream1TransformerGemmSwizzlePolicy::Identity8) {
                    stream1_transformer_ff1_linear_bias_silu_typed<
                        cutlass::half_t,
                        cutlass::arch::Sm80,
                        cutlass::gemm::GemmShape<16, 8, 16>,
                        cutlass::gemm::GemmShape<128, 128, 32>,
                        cutlass::gemm::GemmShape<64, 64, 32>,
                        3,
                        8>(input, weight, bias, output, rows, input_cols, output_cols, stream);
                    return;
                }
                stream1_transformer_ff1_linear_bias_silu_policy<
                    cutlass::half_t,
                    cutlass::arch::Sm80,
                    cutlass::gemm::GemmShape<16, 8, 16>,
                    cutlass::gemm::GemmShape<128, 128, 32>,
                    cutlass::gemm::GemmShape<64, 64, 32>>(
                        input, weight, bias, output, rows, input_cols, output_cols, stream, stage_policy);
                return;
            }
            stream1_transformer_ff1_linear_bias_silu_policy<
                cutlass::half_t,
                cutlass::arch::Sm80,
                cutlass::gemm::GemmShape<16, 8, 16>,
                cutlass::gemm::GemmShape<128, 64, 32>,
                cutlass::gemm::GemmShape<64, 32, 32>>(
                    input, weight, bias, output, rows, input_cols, output_cols, stream, stage_policy);
            return;
        }
        const Stream1TransformerGemmPolicy policy = parse_stream1_transformer_gemm_policy(
            Stream1TransformerGemmFamily::Ff1,
            std::getenv("BEAM_STREAM1_TRANSFORMER_FF1_POLICY"));
        const char* stage_env = std::getenv("BEAM_STREAM1_TRANSFORMER_FF1_STAGES");
        const Stream1TransformerGemmStagePolicy stage_policy =
            stage_env == nullptr || stage_env[0] == '\0'
                ? Stream1TransformerGemmStagePolicy::Stages2
                : parse_stream1_transformer_gemm_stage_policy(stage_env);
        const Stream1TransformerGemmSwizzlePolicy swizzle_policy =
            parse_stream1_transformer_gemm_swizzle_policy(
                std::getenv("BEAM_STREAM1_TRANSFORMER_FF1_SWIZZLE"));
        if (!stream1_transformer_gemm_policy_supported_on_sm(
                Stream1TransformerGemmFamily::Ff1, policy, 75) ||
            !stream1_transformer_gemm_stage_supported_on_sm(
                Stream1TransformerGemmFamily::Ff1, stage_policy, 75) ||
            swizzle_policy != Stream1TransformerGemmSwizzlePolicy::Identity1) {
            throw std::invalid_argument("FF1 policy requires stages=2 and swizzle=1 on SM75");
        }
        if (policy == Stream1TransformerGemmPolicy::M64N128) {
            stream1_transformer_ff1_linear_bias_silu_typed<
                cutlass::half_t,
                cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<64, 128, 32>,
                cutlass::gemm::GemmShape<32, 64, 32>,
                2>(input, weight, bias, output, rows, input_cols, output_cols, stream);
            return;
        }
        if (policy == Stream1TransformerGemmPolicy::M128N128W64N32) {
            stream1_transformer_ff1_linear_bias_silu_typed<
                cutlass::half_t,
                cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<128, 128, 32>,
                cutlass::gemm::GemmShape<64, 32, 32>,
                2>(input, weight, bias, output, rows, input_cols, output_cols, stream);
            return;
        }
        if (policy == Stream1TransformerGemmPolicy::M128N128) {
            stream1_transformer_ff1_linear_bias_silu_typed<
                cutlass::half_t,
                cutlass::arch::Sm75,
                cutlass::gemm::GemmShape<16, 8, 8>,
                cutlass::gemm::GemmShape<128, 128, 32>,
                cutlass::gemm::GemmShape<64, 64, 32>,
                2>(input, weight, bias, output, rows, input_cols, output_cols, stream);
            return;
        }
        stream1_transformer_ff1_linear_bias_silu_typed<
            cutlass::half_t,
            cutlass::arch::Sm75,
            cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::gemm::GemmShape<128, 64, 32>,
            cutlass::gemm::GemmShape<64, 32, 32>,
            2>(input, weight, bias, output, rows, input_cols, output_cols, stream);
        return;
    }
    throw std::invalid_argument("Stream1 piece_transformer FF1 fused GEMM dtype must be fp16 or bf16");
}

template <typename Element, typename ArchTag, typename InstructionShape>
void stream1_transformer_qk_batched_launch_typed(
    const half* qkv,
    half* scores_probs,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    const cutlass::gemm::GemmCoord problem(
        static_cast<int>(STREAM1_TRANSFORMER_SEQ51),
        static_cast<int>(STREAM1_TRANSFORMER_SEQ51),
        static_cast<int>(STREAM1_TRANSFORMER_HEAD_DIM32));
    constexpr float scale = 0.1767766952966369f;
    const std::int64_t qkv_batch_stride = static_cast<std::int64_t>(
        STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_QKV_STRIDE51);
    const std::int64_t score_batch_stride = static_cast<std::int64_t>(STREAM1_TRANSFORMER_SCORE_STRIDE51);
    cutlass::Status status = cutlass::Status::kSuccess;
    for (std::uint32_t head = 0; head < STREAM1_TRANSFORMER_NHEAD8; ++head) {
        const std::uint64_t head_offset = static_cast<std::uint64_t>(head) * STREAM1_TRANSFORMER_HEAD_DIM32;
        const half* q_base = qkv + head_offset;
        const half* k_base = qkv + STREAM1_TRANSFORMER_DMODEL256 + head_offset;
        half* score_base = scores_probs +
            static_cast<std::uint64_t>(head) * b_micro * STREAM1_TRANSFORMER_SCORE_STRIDE51;

        using Gemm = cutlass::gemm::device::GemmBatched<
            Element,
            cutlass::layout::RowMajor,
            Element,
            cutlass::layout::ColumnMajor,
            Element,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            ArchTag,
            cutlass::gemm::GemmShape<64, 64, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>,
            InstructionShape>;
        Gemm gemm;
        typename Gemm::Arguments args{
            problem,
            {reinterpret_cast<Element const*>(q_base), static_cast<int>(STREAM1_TRANSFORMER_QKV_STRIDE51)},
            qkv_batch_stride,
            {reinterpret_cast<Element const*>(k_base), static_cast<int>(STREAM1_TRANSFORMER_QKV_STRIDE51)},
            qkv_batch_stride,
            {reinterpret_cast<Element const*>(score_base), static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)},
            score_batch_stride,
            {reinterpret_cast<Element*>(score_base), static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)},
            score_batch_stride,
            {scale, 0.0f},
            static_cast<int>(b_micro)};
        status = gemm(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("Stream1 piece_transformer QK GemmBatched launch failed");
        }
    }
}

template <typename Element, typename ArchTag, typename InstructionShape>
void stream1_transformer_pv_batched_launch_typed(
    const half* qkv,
    half* scores_probs,
    half* context,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    const cutlass::gemm::GemmCoord problem(
        static_cast<int>(STREAM1_TRANSFORMER_SEQ51),
        static_cast<int>(STREAM1_TRANSFORMER_HEAD_DIM32),
        static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51));
    const std::int64_t score_batch_stride = static_cast<std::int64_t>(STREAM1_TRANSFORMER_SCORE_STRIDE51);
    const std::int64_t context_batch_stride = static_cast<std::int64_t>(
        STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_DMODEL256);
    cutlass::Status status = cutlass::Status::kSuccess;
    for (std::uint32_t head = 0; head < STREAM1_TRANSFORMER_NHEAD8; ++head) {
        const std::uint64_t head_offset = static_cast<std::uint64_t>(head) * STREAM1_TRANSFORMER_HEAD_DIM32;
        const half* prob_base = scores_probs +
            static_cast<std::uint64_t>(head) * b_micro * STREAM1_TRANSFORMER_SCORE_STRIDE51;
        const half* value_base = prob_base + STREAM1_TRANSFORMER_PROB_STRIDE51;
        half* context_base = context + head_offset;

        using Gemm = cutlass::gemm::device::GemmBatched<
            Element,
            cutlass::layout::RowMajor,
            Element,
            cutlass::layout::RowMajor,
            Element,
            cutlass::layout::RowMajor,
            float,
            cutlass::arch::OpClassTensorOp,
            ArchTag,
            cutlass::gemm::GemmShape<64, 32, 32>,
            cutlass::gemm::GemmShape<32, 32, 32>,
            InstructionShape>;
        Gemm gemm;
        typename Gemm::Arguments args{
            problem,
            {reinterpret_cast<Element const*>(prob_base), static_cast<int>(STREAM1_TRANSFORMER_SCORE_ROW_STRIDE51)},
            score_batch_stride,
            {reinterpret_cast<Element const*>(value_base), static_cast<int>(STREAM1_TRANSFORMER_HEAD_DIM32)},
            score_batch_stride,
            {reinterpret_cast<Element const*>(context_base), static_cast<int>(STREAM1_TRANSFORMER_DMODEL256)},
            context_batch_stride,
            {reinterpret_cast<Element*>(context_base), static_cast<int>(STREAM1_TRANSFORMER_DMODEL256)},
            context_batch_stride,
            {1.0f, 0.0f},
            static_cast<int>(b_micro)};
        status = gemm(args, nullptr, stream);
        if (status != cutlass::Status::kSuccess) {
            throw std::runtime_error("Stream1 piece_transformer PV GemmBatched launch failed");
        }
    }
}

#endif
void stream1_transformer_attention_launch(
    half* qkv,
    const Stream1TransformerScratchView& scratch,
    Stream1TransformerDims dims,
    std::uint32_t b_micro,
    Stream1TransformerAttentionBackend attention_backend,
    cudaStream_t stream) {
    if (dims.seq_len != STREAM1_TRANSFORMER_SEQ51 ||
        dims.d_model != STREAM1_TRANSFORMER_DMODEL256 ||
        dims.nhead != STREAM1_TRANSFORMER_NHEAD8 ||
        dims.head_dim != STREAM1_TRANSFORMER_HEAD_DIM32) {
        throw std::invalid_argument("Stream1 piece_transformer tensor attention requires seq_len=51 d_model=256 nhead=8 head_dim=32");
    }
    if (qkv == nullptr || scratch.attention_scores_probs == nullptr ||
        scratch.attention_context == nullptr) {
        throw std::invalid_argument("Stream1 piece_transformer tensor attention requires qkv, score scratch, and attention_context");
    }
#if BEAM_HAS_CUTLASS
    stream1_transformer_fmha_attention_cuda(
        qkv,
        scratch.attention_scores_probs,
        scratch.attention_context,
        dims,
        attention_backend == Stream1TransformerAttentionBackend::Sm75Fp16Fmha,
        b_micro,
        stream);
    return;

#else
    (void)qkv;
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
    const std::uint64_t in_base = static_cast<std::uint64_t>(row) * dims.padded_seq_len * dims.d_model;
    const std::uint64_t out_base = static_cast<std::uint64_t>(row) * dims.d_model;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < dims.d_model; col += blockDim.x) {
        sum += stream1_transformer_load_scalar_device(tokens, in_base + col, dims.dtype);
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
    __syncthreads();
    float var_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < dims.d_model; col += blockDim.x) {
        const float x = stream1_transformer_load_scalar_device(tokens, in_base + col, dims.dtype);
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
        const float x = stream1_transformer_load_scalar_device(tokens, in_base + col, dims.dtype);
        const float y = (x - mean) * inv_std *
            stream1_transformer_load_scalar_device(gamma, col, dims.dtype) +
            stream1_transformer_load_scalar_device(beta, col, dims.dtype);
        stream1_transformer_store_scalar_device(cls, out_base + col, y, dims.dtype);
    }
}

__global__ void stream1_transformer_cls_bias_layernorm_kernel(
    const half* __restrict__ tokens,
    half* __restrict__ cls,
    const half* __restrict__ bias,
    const half* __restrict__ gamma,
    const half* __restrict__ beta,
    Stream1TransformerDims dims,
    std::uint32_t b_micro) {
    const std::uint32_t row = blockIdx.x;
    if (row >= b_micro) {
        return;
    }
    extern __shared__ float reduce[];
    const std::uint64_t in_base = static_cast<std::uint64_t>(row) * dims.padded_seq_len * dims.d_model;
    const std::uint64_t out_base = static_cast<std::uint64_t>(row) * dims.d_model;
    float sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < dims.d_model; col += blockDim.x) {
        sum += stream1_transformer_load_scalar_device(tokens, in_base + col, dims.dtype) +
            stream1_transformer_load_scalar_device(bias, col, dims.dtype);
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
    __syncthreads();
    float var_sum = 0.0f;
    for (std::uint32_t col = threadIdx.x; col < dims.d_model; col += blockDim.x) {
        const float x = stream1_transformer_load_scalar_device(tokens, in_base + col, dims.dtype) +
            stream1_transformer_load_scalar_device(bias, col, dims.dtype);
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
        const float x = stream1_transformer_load_scalar_device(tokens, in_base + col, dims.dtype) +
            stream1_transformer_load_scalar_device(bias, col, dims.dtype);
        const float y = (x - mean) * inv_std *
            stream1_transformer_load_scalar_device(gamma, col, dims.dtype) +
            stream1_transformer_load_scalar_device(beta, col, dims.dtype);
        stream1_transformer_store_scalar_device(cls, out_base + col, y, dims.dtype);
    }
}
__global__ void stream1_transformer_gather_cls256_kernel(
    const half* __restrict__ token_rows,
    half* __restrict__ cls_rows,
    Stream1TransformerDims dims,
    std::uint32_t b_micro) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t tid = threadIdx.x;
    if (row >= b_micro || tid >= 128U) {
        return;
    }
    const std::uint64_t src_base = static_cast<std::uint64_t>(row) * STREAM1_TRANSFORMER_SEQ51 * STREAM1_TRANSFORMER_DMODEL256;
    const std::uint64_t dst_base = static_cast<std::uint64_t>(row) * STREAM1_TRANSFORMER_DMODEL256;
    const std::uint32_t col0 = tid;
    const std::uint32_t col1 = tid + 128U;
    const float x0 = stream1_transformer_load_scalar_device(token_rows, src_base + col0, dims.dtype);
    const float x1 = stream1_transformer_load_scalar_device(token_rows, src_base + col1, dims.dtype);
    stream1_transformer_store_scalar_device(cls_rows, dst_base + col0, x0, dims.dtype);
    stream1_transformer_store_scalar_device(cls_rows, dst_base + col1, x1, dims.dtype);
}
__global__ void stream1_transformer_score_quantize_kernel(
    const half* __restrict__ logits,
    const half* __restrict__ output_bias,
    const std::uint32_t* __restrict__ count,
    std::uint32_t* __restrict__ score_ring,
    std::uint32_t b_micro,
    std::uint32_t parent_offset,
    Stream1TransformerDims dims) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (i >= total) {
        return;
    }
    const std::uint32_t row = i / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint32_t move = i % static_cast<std::uint32_t>(MOVE_COUNT);
    if (parent_offset + row >= *count) {
        return;
    }
    const float q = stream1_transformer_load_scalar_device(logits, static_cast<std::uint64_t>(row) * dims.output_dim + move, dims.dtype) +
        stream1_transformer_load_scalar_device(output_bias, move, dims.dtype);
    score_ring[i] = stream1_transformer_score_key_from_float_device(q);
}

__global__ void stream1_transformer_score_quantize_graph_job_kernel(
    const half* __restrict__ logits,
    const half* __restrict__ output_bias,
    const std::uint32_t* __restrict__ count,
    const std::uint32_t* __restrict__ graph_job_index,
    std::uint32_t* __restrict__ score_ring,
    std::uint32_t b_micro,
    std::uint32_t slot_b_micro,
    std::uint32_t parent_offset,
    Stream1TransformerDims dims) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    if (i >= total) {
        return;
    }
    const std::uint32_t job = *graph_job_index;
    const std::uint32_t row = i / static_cast<std::uint32_t>(MOVE_COUNT);
    const std::uint32_t move = i % static_cast<std::uint32_t>(MOVE_COUNT);
    if (parent_offset + row >= count[job]) {
        return;
    }
    const float q = stream1_transformer_load_scalar_device(logits, static_cast<std::uint64_t>(row) * dims.output_dim + move, dims.dtype) +
        stream1_transformer_load_scalar_device(output_bias, move, dims.dtype);
    const std::uint64_t out_idx = static_cast<std::uint64_t>(job) * slot_b_micro * MOVE_COUNT +
        static_cast<std::uint64_t>(parent_offset) * MOVE_COUNT + i;
    score_ring[out_idx] = stream1_transformer_score_key_from_float_device(q);
}

bool stream1_transformer_env_flag(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0' || std::strcmp(value, "0") == 0) {
        return false;
    }
    if (std::strcmp(value, "1") == 0) {
        return true;
    }
    throw std::invalid_argument(std::string(name) + " must be unset, 0, or 1");
}

bool stream1_transformer_block51_requested() {
    return stream1_transformer_env_flag("BEAM_STREAM1_TRANSFORMER_BLOCK51");
}

bool stream1_transformer_final_cls_only_requested() {
    return stream1_transformer_env_flag("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ONLY");
}

bool stream1_transformer_final_cls_attention_requested() {
    return stream1_transformer_env_flag("BEAM_STREAM1_TRANSFORMER_FINAL_CLS_ATTENTION");
}


bool stream1_transformer_is_block51_shape(Stream1TransformerDims dims) {
    return dims.state_len == 120U &&
        dims.num_classes == 120U &&
        dims.num_pieces == 50U &&
        dims.max_piece_size == 3U &&
        dims.seq_len == 51U &&
        dims.d_model == 256U &&
        dims.nhead == 8U &&
        dims.head_dim == 32U &&
        dims.transformer_layers == 4U &&
        dims.ff_dim == 1024U &&
        dims.output_dim == MOVE_COUNT;
}

void stream1_transformer_final_layer_cls_only_block51_from_ln1_cuda(
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t b_micro,
    Stream1TransformerAttentionBackend attention_backend,
    cudaStream_t stream) {
    const Stream1TransformerDims dims = network.dims;
    const Stream1TransformerBlockView block = network.blocks[3];
    const std::uint32_t token_rows = b_micro * STREAM1_TRANSFORMER_SEQ51;

    stream1_transformer_linear_bias_cuda(
        scratch.attention_context,
        block.attn_qkv_weight,
        block.attn_qkv_bias,
        scratch.qkv,
        token_rows,
        STREAM1_TRANSFORMER_DMODEL256,
        3U * STREAM1_TRANSFORMER_DMODEL256,
        dims.dtype,
        stream);
    if (stream1_transformer_final_cls_attention_requested()) {
        stream1_transformer_fmha_cls_attention_cuda(
            scratch.qkv,
            scratch.attention_scores_probs,
            dims,
            attention_backend == Stream1TransformerAttentionBackend::Sm75Fp16Fmha,
            b_micro,
            stream);
    } else {
        stream1_transformer_attention_launch(
            scratch.qkv,
            scratch,
            dims,
            b_micro,
            attention_backend,
            stream);
        stream1_transformer_gather_cls256_kernel<<<b_micro, 128, 0, stream>>>(
            scratch.attention_context,
            scratch.attention_scores_probs,
            dims,
            b_micro);
    }
    stream1_transformer_gather_cls256_kernel<<<b_micro, 128, 0, stream>>>(
        scratch.tokens,
        scratch.qkv,
        dims,
        b_micro);
    stream1_transformer_residual_bias_round_layernorm_cuda(
        scratch.attention_scores_probs,
        block.attn_out_weight,
        scratch.qkv,
        block.attn_out_bias,
        scratch.attention_context,
        block.ln2_gamma,
        block.ln2_beta,
        b_micro,
        STREAM1_TRANSFORMER_DMODEL256,
        STREAM1_TRANSFORMER_DMODEL256,
        dims.dtype,
        stream);
    stream1_transformer_ff1_linear_bias_silu_cuda(
        scratch.attention_context,
        block.ff1_weight,
        block.ff1_bias,
        scratch.ff_hidden,
        b_micro,
        STREAM1_TRANSFORMER_DMODEL256,
        1024U,
        dims.dtype,
        stream);
    stream1_transformer_residual_bias_round_layernorm_cuda(
        scratch.ff_hidden,
        block.ff2_weight,
        scratch.qkv,
        block.ff2_bias,
        scratch.attention_context,
        network.output_ln_gamma,
        network.output_ln_beta,
        b_micro,
        1024U,
        STREAM1_TRANSFORMER_DMODEL256,
        dims.dtype,
        stream);
}

void stream1_transformer_final_layer_cls_only_block51_cuda(
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t b_micro,
    Stream1TransformerAttentionBackend attention_backend,
    cudaStream_t stream) {
    const Stream1TransformerDims dims = network.dims;
    const Stream1TransformerBlockView block = network.blocks[3];
    const std::uint32_t token_rows = b_micro * STREAM1_TRANSFORMER_SEQ51;

    stream1_transformer_layernorm_copy_launch(
        scratch.tokens,
        scratch.attention_context,
        block.ln1_gamma,
        block.ln1_beta,
        token_rows,
        STREAM1_TRANSFORMER_DMODEL256,
        dims.dtype,
        stream);
    stream1_transformer_final_layer_cls_only_block51_from_ln1_cuda(
        network,
        scratch,
        b_micro,
        attention_backend,
        stream);
}

void stream1_transformer_block51_run_layers_cuda(
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t b_micro,
    Stream1TransformerAttentionBackend attention_backend,
    cudaStream_t stream) {
    const Stream1TransformerDims dims = network.dims;
    const std::uint32_t token_rows = b_micro * STREAM1_TRANSFORMER_SEQ51;
    const bool final_cls_only = stream1_transformer_final_cls_only_requested();

    const std::uint32_t full_token_layer_count = final_cls_only ? 3U : 4U;
    bool ln1_precomputed = false;
    bool final_ln1_precomputed = false;

    for (std::uint32_t layer = 0; layer < full_token_layer_count; ++layer) {
        const Stream1TransformerBlockView block = network.blocks[layer];
        if (!ln1_precomputed) {
            stream1_transformer_layernorm_copy_launch(
                scratch.tokens,
                scratch.attention_context,
                block.ln1_gamma,
                block.ln1_beta,
                token_rows,
                STREAM1_TRANSFORMER_DMODEL256,
                dims.dtype,
                stream);
        }
        ln1_precomputed = false;

        stream1_transformer_linear_bias_cuda(
            scratch.attention_context,
            block.attn_qkv_weight,
            block.attn_qkv_bias,
            scratch.qkv,
            token_rows,
            STREAM1_TRANSFORMER_DMODEL256,
            3U * STREAM1_TRANSFORMER_DMODEL256,
            dims.dtype,
            stream);
        stream1_transformer_attention_launch(
            scratch.qkv,
            scratch,
            dims,
            b_micro,
            attention_backend,
            stream);
        stream1_transformer_residual_bias_round_layernorm_cuda(
            scratch.attention_context,
            block.attn_out_weight,
            scratch.tokens,
            block.attn_out_bias,
            scratch.attention_context,
            block.ln2_gamma,
            block.ln2_beta,
            token_rows,
            STREAM1_TRANSFORMER_DMODEL256,
            STREAM1_TRANSFORMER_DMODEL256,
            dims.dtype,
            stream);
        stream1_transformer_ff1_linear_bias_silu_cuda(
            scratch.attention_context,
            block.ff1_weight,
            block.ff1_bias,
            scratch.ff_hidden,
            token_rows,
            STREAM1_TRANSFORMER_DMODEL256,
            1024U,
            dims.dtype,
            stream);
        if (layer + 1U < full_token_layer_count) {
            const Stream1TransformerBlockView next_block = network.blocks[layer + 1U];
            stream1_transformer_residual_bias_round_layernorm_cuda(
                scratch.ff_hidden,
                block.ff2_weight,
                scratch.tokens,
                block.ff2_bias,
                scratch.attention_context,
                next_block.ln1_gamma,
                next_block.ln1_beta,
                token_rows,
                1024U,
                STREAM1_TRANSFORMER_DMODEL256,
                dims.dtype,
                stream);
            ln1_precomputed = true;
        } else if (final_cls_only) {
            const Stream1TransformerBlockView final_block = network.blocks[3];
            stream1_transformer_residual_bias_round_layernorm_cuda(
                scratch.ff_hidden,
                block.ff2_weight,
                scratch.tokens,
                block.ff2_bias,
                scratch.attention_context,
                final_block.ln1_gamma,
                final_block.ln1_beta,
                token_rows,
                1024U,
                STREAM1_TRANSFORMER_DMODEL256,
                dims.dtype,
                stream);
            final_ln1_precomputed = true;
        } else {
            stream1_transformer_linear_residual_cuda(
                scratch.ff_hidden,
                block.ff2_weight,
                scratch.tokens,
                token_rows,
                1024U,
                STREAM1_TRANSFORMER_DMODEL256,
                dims.dtype,
                stream);
            stream1_transformer_bias_add_launch(
                scratch.tokens,
                block.ff2_bias,
                token_rows,
                STREAM1_TRANSFORMER_DMODEL256,
                dims.dtype,
                stream);
        }
    }

    if (final_cls_only) {
        if (final_ln1_precomputed) {
            stream1_transformer_final_layer_cls_only_block51_from_ln1_cuda(
                network,
                scratch,
                b_micro,
                attention_backend,
                stream);
        } else {
            stream1_transformer_final_layer_cls_only_block51_cuda(
                network,
                scratch,
                b_micro,
                attention_backend,
                stream);
        }
    } else {
        stream1_transformer_cls_layernorm_kernel<<<b_micro, 256, 256 * sizeof(float), stream>>>(
            scratch.tokens,
            scratch.attention_context,
            network.output_ln_gamma,
            network.output_ln_beta,
            dims,
            b_micro);
    }
}

void stream1_transformer_inference_block51_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    std::uint32_t parent_offset,
    Stream1TransformerAttentionBackend attention_backend,
    cudaStream_t stream) {
    NvtxRange range("Stream1_transformer_block51_launch");
    const Stream1TransformerDims dims = network.dims;
    if (!stream1_transformer_is_block51_shape(dims)) {
        throw std::invalid_argument("Stream1 piece_transformer block51 backend requires exact p900 seq=51 d=256 h=8 ff=1024 layers=4 shape");
    }
    if (b_micro == 0U) {
        return;
    }
    stream1_transformer_build_input_layernorm51x256_launch(
        current_frontier_states,
        parent_base,
        count,
        network,
        scratch.tokens,
        b_micro,
        parent_offset,
        stream);
    stream1_transformer_block51_run_layers_cuda(
        network,
        scratch,
        b_micro,
        attention_backend,
        stream);
    stream1_cutlass_linear_cuda(
        scratch.attention_context,
        network.output_weight,
        scratch.logits,
        b_micro,
        STREAM1_TRANSFORMER_DMODEL256,
        dims.output_dim,
        dims.dtype,
        stream);
    stream1_transformer_score_quantize_kernel<<<(b_micro * MOVE_COUNT + 255U) / 256U, 256, 0, stream>>>(
        scratch.logits,
        network.output_bias,
        count,
        score_ring,
        b_micro,
        parent_offset,
        dims);
}

void stream1_transformer_inference_block51_graph_job_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const std::uint32_t* graph_job_index,
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    std::uint32_t slot_b_micro,
    std::uint32_t parent_offset,
    Stream1TransformerAttentionBackend attention_backend,
    cudaStream_t stream) {
    NvtxRange range("Stream1_transformer_block51_graph_job_launch");
    const Stream1TransformerDims dims = network.dims;
    if (!stream1_transformer_is_block51_shape(dims)) {
        throw std::invalid_argument("Stream1 piece_transformer block51 backend requires exact p900 seq=51 d=256 h=8 ff=1024 layers=4 shape");
    }
    if (b_micro == 0U) {
        return;
    }
    stream1_transformer_build_input_layernorm51x256_graph_job_launch(
        current_frontier_states,
        parent_base,
        count,
        graph_job_index,
        network,
        scratch.tokens,
        b_micro,
        parent_offset,
        stream);
    stream1_transformer_block51_run_layers_cuda(
        network,
        scratch,
        b_micro,
        attention_backend,
        stream);
    stream1_cutlass_linear_cuda(
        scratch.attention_context,
        network.output_weight,
        scratch.logits,
        b_micro,
        STREAM1_TRANSFORMER_DMODEL256,
        dims.output_dim,
        dims.dtype,
        stream);
    stream1_transformer_score_quantize_graph_job_kernel<<<(b_micro * MOVE_COUNT + 255U) / 256U, 256, 0, stream>>>(
        scratch.logits,
        network.output_bias,
        count,
        graph_job_index,
        score_ring,
        b_micro,
        slot_b_micro,
        parent_offset,
        dims);
}
void stream1_transformer_inference_graph_job_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const std::uint32_t* graph_job_index,
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    std::uint32_t slot_b_micro,
    std::uint32_t parent_offset,
    cudaStream_t stream) {
    NvtxRange range("Stream1_transformer_graph_job_inference_launch");
    const Stream1TransformerDims dims = network.dims;
    if (dims.output_dim != MOVE_COUNT) {
        throw std::invalid_argument("Stream1 piece_transformer output_dim must equal MOVE_COUNT");
    }
    if (dims.d_model == 0U || dims.nhead == 0U || dims.head_dim == 0U ||
        dims.d_model != dims.nhead * dims.head_dim) {
        throw std::invalid_argument("Stream1 piece_transformer d_model must equal nhead * head_dim");
    }
    if (dims.seq_len != dims.num_pieces + 1U || dims.max_piece_size == 0U ||
        dims.padded_seq_len < dims.seq_len || dims.sequence_alignment == 0U ||
        dims.padded_seq_len % dims.sequence_alignment != 0U) {
        throw std::invalid_argument("Stream1 piece_transformer logical/padded sequence dimensions are inconsistent");
    }
    if (dims.padded_seq_len > 64U || dims.head_dim > 64U) {
        throw std::invalid_argument("Stream1 piece_transformer fused attention tile requires seq_len<=64 and head_dim<=64");
    }
    const Stream1TransformerAttentionBackend attention_backend =
        stream1_transformer_select_attention_backend(dims.dtype);
    if (b_micro == 0U) {
        return;
    }
    if (stream1_transformer_block51_requested() && dims.seq_len == 51U && dims.padded_seq_len >= 51U) {
        stream1_transformer_inference_block51_graph_job_cuda(
            current_frontier_states,
            parent_base,
            count,
            graph_job_index,
            network,
            scratch,
            score_ring,
            b_micro,
            slot_b_micro,
            parent_offset,
            attention_backend,
            stream);
        return;
    }
#if BEAM_HAS_CUTLASS
    const std::uint32_t token_rows = b_micro * dims.padded_seq_len;
    const dim3 token_block(128);
    const dim3 token_grid(token_rows, (dims.d_model + token_block.x - 1U) / token_block.x);
    stream1_transformer_build_input_kernel_graph_job<<<token_grid, token_block, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        graph_job_index,
        network,
        scratch.tokens,
        b_micro,
        parent_offset);

    stream1_transformer_layernorm_copy_launch(
        scratch.tokens,
        scratch.tokens,
        network.input_ln_gamma,
        network.input_ln_beta,
        token_rows,
        dims.d_model,
        dims.dtype,
        stream);
    stream1_transformer_zero_padded_rows_launch(scratch.tokens, dims, b_micro, stream);

    for (std::uint32_t layer = 0; layer < dims.transformer_layers; ++layer) {
        const Stream1TransformerBlockView block = network.blocks[layer];
        if (layer == 0U) {
            stream1_transformer_layernorm_copy_launch(
                scratch.tokens,
                scratch.attention_context,
                block.ln1_gamma,
                block.ln1_beta,
                token_rows,
                dims.d_model,
                dims.dtype,
                stream);
        } else {
            const Stream1TransformerBlockView prev_block = network.blocks[layer - 1U];
            stream1_transformer_bias_layernorm_copy_launch(
                scratch.tokens,
                scratch.attention_context,
                prev_block.ff2_bias,
                block.ln1_gamma,
                block.ln1_beta,
                token_rows,
                dims.d_model,
                dims.dtype,
                stream);
        }
        stream1_transformer_linear_bias_cuda(
            scratch.attention_context,
            block.attn_qkv_weight,
            block.attn_qkv_bias,
            scratch.qkv,
            token_rows,
            dims.d_model,
            3U * dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_attention_launch(
            scratch.qkv,
            scratch,
            dims,
            b_micro,
            attention_backend,
            stream);
        stream1_transformer_linear_residual_cuda(
            scratch.attention_context,
            block.attn_out_weight,
            scratch.tokens,
            token_rows,
            dims.d_model,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_bias_layernorm_copy_launch(
            scratch.tokens,
            scratch.attention_context,
            block.attn_out_bias,
            block.ln2_gamma,
            block.ln2_beta,
            token_rows,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_ff1_linear_bias_silu_cuda(
            scratch.attention_context,
            block.ff1_weight,
            block.ff1_bias,
            scratch.ff_hidden,
            token_rows,
            dims.d_model,
            dims.ff_dim,
            dims.dtype,
            stream);
        stream1_transformer_linear_residual_cuda(
            scratch.ff_hidden,
            block.ff2_weight,
            scratch.tokens,
            token_rows,
            dims.ff_dim,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_zero_padded_rows_launch(scratch.tokens, dims, b_micro, stream);
    }

    if (dims.transformer_layers > 0U) {
        const Stream1TransformerBlockView last_block = network.blocks[dims.transformer_layers - 1U];
        stream1_transformer_cls_bias_layernorm_kernel<<<b_micro, 256, 256 * sizeof(float), stream>>>(
            scratch.tokens,
            scratch.attention_context,
            last_block.ff2_bias,
            network.output_ln_gamma,
            network.output_ln_beta,
            dims,
            b_micro);
    } else {
        stream1_transformer_cls_layernorm_kernel<<<b_micro, 256, 256 * sizeof(float), stream>>>(
            scratch.tokens,
            scratch.attention_context,
            network.output_ln_gamma,
            network.output_ln_beta,
            dims,
            b_micro);
    }
    stream1_cutlass_linear_cuda(
        scratch.attention_context,
        network.output_weight,
        scratch.logits,
        b_micro,
        dims.d_model,
        dims.output_dim,
        dims.dtype,
        stream);
    stream1_transformer_score_quantize_graph_job_kernel<<<(b_micro * MOVE_COUNT + 255U) / 256U, 256, 0, stream>>>(
        scratch.logits,
        network.output_bias,
        count,
        graph_job_index,
        score_ring,
        b_micro,
        slot_b_micro,
        parent_offset,
        dims);
#else
    (void)current_frontier_states;
    (void)parent_base;
    (void)count;
    (void)graph_job_index;
    (void)network;
    (void)scratch;
    (void)score_ring;
    (void)b_micro;
    (void)stream;
    throw std::invalid_argument("Stream1 piece_transformer CUDA forward requires CUTLASS-enabled build");
#endif
}
void stream1_transformer_inference_cuda(
    const State128* current_frontier_states,
    const std::uint64_t* parent_base,
    const std::uint32_t* count,
    const Stream1TransformerNetworkView& network,
    const Stream1TransformerScratchView& scratch,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    std::uint32_t parent_offset,
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
    if (dims.seq_len != dims.num_pieces + 1U || dims.max_piece_size == 0U ||
        dims.padded_seq_len < dims.seq_len || dims.sequence_alignment == 0U ||
        dims.padded_seq_len % dims.sequence_alignment != 0U) {
        throw std::invalid_argument("Stream1 piece_transformer logical/padded sequence dimensions are inconsistent");
    }
    if (dims.padded_seq_len > 64U || dims.head_dim > 64U) {
        throw std::invalid_argument("Stream1 piece_transformer fused attention tile requires seq_len<=64 and head_dim<=64");
    }
    const Stream1TransformerAttentionBackend attention_backend =
        stream1_transformer_select_attention_backend(dims.dtype);
    if (b_micro == 0U) {
        return;
    }
    if (stream1_transformer_block51_requested() && dims.seq_len == 51U && dims.padded_seq_len >= 51U) {
        stream1_transformer_inference_block51_cuda(
            current_frontier_states,
            parent_base,
            count,
            network,
            scratch,
            score_ring,
            b_micro,
            parent_offset,
            attention_backend,
            stream);
        return;
    }
#if BEAM_HAS_CUTLASS
    const std::uint32_t token_rows = b_micro * dims.padded_seq_len;
    const dim3 token_block(128);
    const dim3 token_grid(token_rows, (dims.d_model + token_block.x - 1U) / token_block.x);
    stream1_transformer_build_input_kernel<<<token_grid, token_block, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        network,
        scratch.tokens,
        b_micro,
        parent_offset);

    stream1_transformer_layernorm_copy_launch(
        scratch.tokens,
        scratch.tokens,
        network.input_ln_gamma,
        network.input_ln_beta,
        token_rows,
        dims.d_model,
        dims.dtype,
        stream);
    stream1_transformer_zero_padded_rows_launch(scratch.tokens, dims, b_micro, stream);


    for (std::uint32_t layer = 0; layer < dims.transformer_layers; ++layer) {
        const Stream1TransformerBlockView block = network.blocks[layer];
        if (layer == 0U) {
            stream1_transformer_layernorm_copy_launch(
                scratch.tokens,
                scratch.attention_context,
                block.ln1_gamma,
                block.ln1_beta,
                token_rows,
                dims.d_model,
                dims.dtype,
                stream);
        } else {
            const Stream1TransformerBlockView prev_block = network.blocks[layer - 1U];
            stream1_transformer_bias_layernorm_copy_launch(
                scratch.tokens,
                scratch.attention_context,
                prev_block.ff2_bias,
                block.ln1_gamma,
                block.ln1_beta,
                token_rows,
                dims.d_model,
                dims.dtype,
                stream);
        }
        stream1_transformer_linear_bias_cuda(
            scratch.attention_context,
            block.attn_qkv_weight,
            block.attn_qkv_bias,
            scratch.qkv,
            token_rows,
            dims.d_model,
            3U * dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_attention_launch(
            scratch.qkv,
            scratch,
            dims,
            b_micro,
            attention_backend,
            stream);
        stream1_transformer_linear_residual_cuda(
            scratch.attention_context,
            block.attn_out_weight,
            scratch.tokens,
            token_rows,
            dims.d_model,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_bias_layernorm_copy_launch(
            scratch.tokens,
            scratch.attention_context,
            block.attn_out_bias,
            block.ln2_gamma,
            block.ln2_beta,
            token_rows,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_ff1_linear_bias_silu_cuda(
            scratch.attention_context,
            block.ff1_weight,
            block.ff1_bias,
            scratch.ff_hidden,
            token_rows,
            dims.d_model,
            dims.ff_dim,
            dims.dtype,
            stream);
        stream1_transformer_linear_residual_cuda(
            scratch.ff_hidden,
            block.ff2_weight,
            scratch.tokens,
            token_rows,
            dims.ff_dim,
            dims.d_model,
            dims.dtype,
            stream);
        stream1_transformer_zero_padded_rows_launch(scratch.tokens, dims, b_micro, stream);

    }

    if (dims.transformer_layers > 0U) {
        const Stream1TransformerBlockView last_block = network.blocks[dims.transformer_layers - 1U];
        stream1_transformer_cls_bias_layernorm_kernel<<<b_micro, 256, 256 * sizeof(float), stream>>>(
            scratch.tokens,
            scratch.attention_context,
            last_block.ff2_bias,
            network.output_ln_gamma,
            network.output_ln_beta,
            dims,
            b_micro);
    } else {
        stream1_transformer_cls_layernorm_kernel<<<b_micro, 256, 256 * sizeof(float), stream>>>(
            scratch.tokens,
            scratch.attention_context,
            network.output_ln_gamma,
            network.output_ln_beta,
            dims,
            b_micro);
    }
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
        parent_offset,
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

} // namespace beam
