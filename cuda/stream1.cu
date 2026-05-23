#include "stream1.hpp"

#include "config.hpp"
#include "nvtx_ranges.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#if BEAM_HAS_CUTLASS
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/layout/matrix.h>
#endif

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

__device__ std::uint32_t score_key_from_float_device(float q) {
    q = fminf(fmaxf(q, 0.0f), SCORE_MAX_Q);
    return static_cast<std::uint32_t>(rintf(q * static_cast<float>(SCORE_SCALE)));
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
    const half* __restrict__ input_weight,
    const half* __restrict__ input_bias,
    half* __restrict__ hidden1,
    Stream1NetworkDims dims,
    std::uint32_t b_micro) {
    const std::uint32_t row = blockIdx.x;
    const std::uint32_t h = (blockIdx.y * blockDim.x + threadIdx.x) * 4U;
    const std::uint32_t count_value = *count;
    const bool row_valid = row < b_micro && row < count_value;
    __shared__ std::uint8_t state_shared[STATE_STORAGE_LEN];
    if (row_valid && threadIdx.x < STATE_STORAGE_LEN) {
        const std::uint64_t parent_idx = *parent_base + row;
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

    for (std::uint32_t p = 0; p < dims.state_len; ++p) {
        const std::uint32_t value = static_cast<std::uint32_t>(state_shared[p]);
        const std::uint32_t idx = (p * dims.num_classes + value) * dims.hidden1 + h;
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
    }

    half* out = hidden1 + row * dims.hidden1 + h;
    acc0 = relu_device(acc0);
    if (has01) {
        acc1 = relu_device(acc1);
        *reinterpret_cast<half2*>(out) = __floats2half2_rn(acc0, acc1);
    } else {
        out[0] = __float2half(acc0);
    }
    if (has23) {
        acc2 = relu_device(acc2);
        acc3 = relu_device(acc3);
        *reinterpret_cast<half2*>(out + 2U) = __floats2half2_rn(acc2, acc3);
    } else if (has2) {
        out[2] = __float2half(relu_device(acc2));
    }
}

__global__ void stream1_bias_relu_kernel(
    half* matrix,
    const half* bias,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * cols;
    if (i >= total) {
        return;
    }
    const std::uint32_t col = i % cols;
    const float x = __half2float(matrix[i]) + __half2float(bias[col]);
    matrix[i] = __float2half(relu_device(x));
}

__global__ void stream1_residual_add_bias_relu_kernel(
    half* matrix,
    const half* residual,
    const half* bias,
    std::uint32_t rows,
    std::uint32_t cols) {
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t total = rows * cols;
    if (i >= total) {
        return;
    }
    const std::uint32_t col = i % cols;
    const float x = __half2float(matrix[i]) + __half2float(residual[i]) + __half2float(bias[col]);
    matrix[i] = __float2half(relu_device(x));
}

__global__ void stream1_score_epilogue_quantize_kernel(
    const half* output,
    const half* output_bias,
    const std::uint32_t* count,
    std::uint32_t* score_ring,
    std::uint32_t b_micro) {
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
    const float q = __half2float(output[i]) + __half2float(output_bias[move]);
    score_ring[i] = score_key_from_float_device(q);
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
    const Stream1NetworkView& network,
    const Stream1CutlassScratch& scratch,
    std::uint32_t* score_ring,
    std::uint32_t b_micro,
    cudaStream_t stream) {
    NvtxRange range("Stream1_CUTLASS_inference_launch");
#if BEAM_HAS_CUTLASS
    const dim3 input_block(128);
    const dim3 input_grid(b_micro, (network.dims.hidden1 + input_block.x * 4U - 1U) / (input_block.x * 4U));
    stream1_folded_input_half4_tiled_kernel<<<input_grid, input_block, 0, stream>>>(
        current_frontier_states,
        parent_base,
        count,
        network.input_weight,
        network.input_bias,
        scratch.hidden1,
        network.dims,
        b_micro);

    stream1_cutlass_linear_cuda(
        scratch.hidden1,
        network.hidden_weight,
        scratch.hidden2,
        b_micro,
        network.dims.hidden1,
        network.dims.hidden2,
        stream);

    const std::uint32_t hidden2_total = b_micro * network.dims.hidden2;
    stream1_bias_relu_kernel<<<(hidden2_total + 255U) / 256U, 256, 0, stream>>>(
        scratch.hidden2,
        network.hidden_bias,
        b_micro,
        network.dims.hidden2);

    half* residual_in = scratch.hidden2;
    half* residual_tmp = scratch.residual_tmp;
    for (std::uint32_t block = 0; block < network.dims.residual_count; ++block) {
        stream1_cutlass_linear_cuda(
            residual_in,
            network.residual_fc1_weight[block],
            residual_tmp,
            b_micro,
            network.dims.hidden2,
            network.dims.hidden2,
            stream);

        stream1_bias_relu_kernel<<<(hidden2_total + 255U) / 256U, 256, 0, stream>>>(
            residual_tmp,
            network.residual_fc1_bias[block],
            b_micro,
            network.dims.hidden2);

        stream1_cutlass_linear_cuda(
            residual_tmp,
            network.residual_fc2_weight[block],
            residual_tmp,
            b_micro,
            network.dims.hidden2,
            network.dims.hidden2,
            stream);

        stream1_residual_add_bias_relu_kernel<<<(hidden2_total + 255U) / 256U, 256, 0, stream>>>(
            residual_tmp,
            residual_in,
            network.residual_fc2_bias[block],
            b_micro,
            network.dims.hidden2);

        half* swap = residual_in;
        residual_in = residual_tmp;
        residual_tmp = swap;
    }

    stream1_cutlass_linear_cuda(
        residual_in,
        network.output_weight,
        scratch.output,
        b_micro,
        network.dims.hidden2,
        static_cast<std::uint32_t>(MOVE_COUNT),
        stream);

    const std::uint32_t output_total = b_micro * static_cast<std::uint32_t>(MOVE_COUNT);
    stream1_score_epilogue_quantize_kernel<<<(output_total + 255U) / 256U, 256, 0, stream>>>(
        scratch.output,
        network.output_bias,
        count,
        score_ring,
        b_micro);
#else
    (void)current_frontier_states;
    (void)parent_base;
    (void)count;
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
    cudaStream_t stream) {
    NvtxRange range("Stream1_CUTLASS_linear_launch");
#if BEAM_HAS_CUTLASS
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
