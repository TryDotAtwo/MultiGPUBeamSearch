#pragma once

#include "stream1_transformer_sm120_nvfp4_ffn_policy.hpp"
#include "stream1_transformer_sm120_nvfp4_fused_ffn.cuh"

#include <cuda/barrier>
#include <cuda_runtime.h>

#include <cstdint>

namespace stream1_sm120_nvfp4_persistent {

#ifndef STREAM1_SM120_M16_CONSUMER_WARPS
#define STREAM1_SM120_M16_CONSUMER_WARPS 4
#endif

#ifndef STREAM1_SM120_M16_PRODUCER_WARPS
#define STREAM1_SM120_M16_PRODUCER_WARPS 4
#endif

#ifndef STREAM1_SM120_HOIST_FF2_A
#define STREAM1_SM120_HOIST_FF2_A 0
#endif

#ifndef STREAM1_SM120_DIRECT_FF2_FRAGMENT_PACK
#define STREAM1_SM120_DIRECT_FF2_FRAGMENT_PACK 1
#endif

struct Pipeline : Stream1Sm120Nvfp4PersistentFfnPipeline {
    static constexpr std::uint32_t kProducerThreads = kProducerWarps * 32U;
    static constexpr std::uint32_t kConsumerThreads = kConsumerWarps * 32U;
    static constexpr std::uint32_t kBytesPerBuffer =
        kValueBytesPerBuffer + kScaleBytesPerBuffer;
    static constexpr std::uint32_t kWordsPerBuffer = kBytesPerBuffer / 4U;
    static_assert(kProducerThreads + kConsumerThreads == kThreads);
    static_assert(kBytesPerBuffer * kRingBuffers == kHandoffBytes);
    static_assert(kBytesPerBuffer % sizeof(std::uint32_t) == 0U);
};

using BlockBarrier = cuda::barrier<cuda::thread_scope_block>;

struct alignas(128) HandoffStorage {
    std::uint32_t words[Pipeline::kRingBuffers][Pipeline::kWordsPerBuffer];
    BlockBarrier ready[Pipeline::kRingBuffers];
    BlockBarrier free[Pipeline::kRingBuffers];
    std::uint32_t generation[Pipeline::kRingBuffers];
};

// Executable ownership oracle used before wiring real MMA fragments.  The
// producer and consumer warp groups progress independently; ready/free parity
// prevents both early reads and overwrite of a generation still in use.
__global__ void persistent_handoff_contract_kernel(
    const std::uint32_t* source,
    std::uint64_t* checksums,
    std::uint32_t* violations) {
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& storage = *reinterpret_cast<HandoffStorage*>(shared_bytes);
    if (threadIdx.x == 0U) {
        for (std::uint32_t buffer = 0; buffer < Pipeline::kRingBuffers; ++buffer) {
            init(&storage.ready[buffer], Pipeline::kProducerThreads);
            init(&storage.free[buffer], Pipeline::kConsumerThreads);
            storage.generation[buffer] = 0xffffffffU;
        }
    }
    __syncthreads();

    if (threadIdx.x < Pipeline::kProducerThreads) {
        const std::uint32_t producer_thread = threadIdx.x;
        for (std::uint32_t slice = 0; slice < Pipeline::kHiddenSlices; ++slice) {
            const std::uint32_t buffer = slice % Pipeline::kRingBuffers;
            const std::uint32_t generation = slice / Pipeline::kRingBuffers;
            if (generation != 0U) {
                storage.free[buffer].wait_parity((generation - 1U) & 1U);
            }
            for (std::uint32_t word = producer_thread;
                 word < Pipeline::kWordsPerBuffer;
                 word += Pipeline::kProducerThreads) {
                storage.words[buffer][word] =
                    source[static_cast<std::size_t>(slice) *
                               Pipeline::kWordsPerBuffer +
                           word];
            }
            if (producer_thread == 0U) {
                storage.generation[buffer] = generation;
            }
            (void)storage.ready[buffer].arrive();
        }
    } else {
        const std::uint32_t consumer_thread =
            threadIdx.x - Pipeline::kProducerThreads;
        for (std::uint32_t slice = 0; slice < Pipeline::kHiddenSlices; ++slice) {
            const std::uint32_t buffer = slice % Pipeline::kRingBuffers;
            const std::uint32_t generation = slice / Pipeline::kRingBuffers;
            storage.ready[buffer].wait_parity(generation & 1U);
            if (storage.generation[buffer] != generation) {
                atomicAdd(violations, 1U);
            }
            std::uint64_t local = 0U;
            for (std::uint32_t word = consumer_thread;
                 word < Pipeline::kWordsPerBuffer;
                 word += Pipeline::kConsumerThreads) {
                local += storage.words[buffer][word];
            }
            atomicAdd(
                reinterpret_cast<unsigned long long*>(checksums + slice),
                static_cast<unsigned long long>(local));
            (void)storage.free[buffer].arrive();
        }
    }
}

struct PersistentFfnM16Pipeline {
    static constexpr std::uint32_t kRows = 16U;
    static constexpr std::uint32_t kDModel = 256U;
    static constexpr std::uint32_t kFfDim = 1024U;
    static constexpr std::uint32_t kSliceColumns = 128U;
    static constexpr std::uint32_t kHiddenSlices = kFfDim / kSliceColumns;
    static constexpr std::uint32_t kRingBuffers = 2U;
    static constexpr std::uint32_t kProducerWarps =
        STREAM1_SM120_M16_PRODUCER_WARPS;
    static constexpr std::uint32_t kConsumerWarps =
        STREAM1_SM120_M16_CONSUMER_WARPS;
    static constexpr std::uint32_t kProducerThreads = kProducerWarps * 32U;
    static constexpr std::uint32_t kConsumerThreads = kConsumerWarps * 32U;
    static constexpr std::uint32_t kThreads =
        kProducerThreads + kConsumerThreads;
    static_assert(kProducerWarps == 4U || kProducerWarps == 8U);
    static_assert(
        kConsumerWarps == 2U || kConsumerWarps == 4U ||
        kConsumerWarps == 8U);
    static_assert((kDModel / 8U) % kConsumerWarps == 0U);
    static constexpr std::uint32_t kValueBytesPerBuffer =
        kRows * kSliceColumns / 2U;
    static constexpr std::uint32_t kScaleBytesPerBuffer =
        kRows * (kSliceColumns / 16U);
    static constexpr std::uint32_t kFragmentWordsPerBuffer =
        kRows * kSliceColumns / 8U;
};

// Production-scale SM120 topology.  SM120 NVFP4 uses warp-level mma.sync
// accumulators (not SM100 UMMA/TMEM), so one CTA owns four independent M16
// row tiles.  Four producer warps compute FF1 for all 64 rows while sixteen
// consumer warps split each row tile's 32 output N8 tiles, keeping the FF2
// accumulator at 32 FP32 registers per lane.  Input and the active immutable weight slice
// are staged once per CTA; the quantized hidden slice stays in a two-slot
// fragment-order ring.
struct PersistentFfnM32Pipeline {
    static constexpr std::uint32_t kRows = 32U;
    static constexpr std::uint32_t kRowsPerWarp = 16U;
    static constexpr std::uint32_t kDModel = 256U;
    static constexpr std::uint32_t kFfDim = 1024U;
    static constexpr std::uint32_t kSliceColumns = 128U;
    static constexpr std::uint32_t kHiddenSlices = kFfDim / kSliceColumns;
    static constexpr std::uint32_t kRingBuffers = 2U;
    static constexpr std::uint32_t kProducerWarps =
        4U * (kRows / kRowsPerWarp);
    static constexpr std::uint32_t kConsumerWarps =
        4U * (kRows / kRowsPerWarp);
    static constexpr std::uint32_t kThreads =
        (kProducerWarps + kConsumerWarps) * 32U;
    static constexpr std::uint32_t kInputValueBytes = kRows * kDModel / 2U;
    static constexpr std::uint32_t kInputScaleBytes =
        kRows * (kDModel / 16U);
    static constexpr std::uint32_t kWeightValueBytes =
        kDModel * kSliceColumns / 2U;
    static constexpr std::uint32_t kWeightScaleBytes =
        kSliceColumns * (kDModel / 16U);
    static constexpr std::uint32_t kRingValueBytes =
        kRows * kSliceColumns / 2U;
    static constexpr std::uint32_t kRingScaleBytes =
        kRows * (kSliceColumns / 16U);
    static constexpr std::uint32_t kRingScratchBytes = kRingValueBytes;
    static constexpr std::uint32_t kInputStageBytes =
        kInputValueBytes + kInputScaleBytes;
    static constexpr std::uint32_t kWeightStageBytes =
        kWeightValueBytes + kWeightScaleBytes;
    static constexpr std::uint32_t kRingBytes =
        kRingBuffers *
        (kRingValueBytes + kRingScaleBytes + kRingScratchBytes);
};

struct alignas(128) PersistentFfnM32Storage {
    using Config = PersistentFfnM32Pipeline;
    struct alignas(128) InputStage {
        std::uint32_t fragments[Config::kInputValueBytes / 4U];
        std::uint8_t scales[Config::kInputScaleBytes];
    } input;
    struct alignas(128) WeightStage {
        std::uint32_t fragments[Config::kWeightValueBytes / 4U];
        std::uint8_t scales[Config::kWeightScaleBytes];
    } weights;
    struct alignas(128) RingBuffer {
        // Temporary row-major quantization scratch.  This is shared-memory
        // only and will be removed once direct fragment-order packing has a
        // numerical oracle.
        std::uint8_t values[Config::kRingScratchBytes];
        std::uint32_t fragments[Config::kRingValueBytes / 4U];
        std::uint8_t scales[Config::kRingScaleBytes];
    } ring[Config::kRingBuffers];
    BlockBarrier ready[Config::kRingBuffers];
    BlockBarrier free[Config::kRingBuffers];
};

static_assert(PersistentFfnM32Pipeline::kProducerWarps == 8U);
static_assert(PersistentFfnM32Pipeline::kConsumerWarps == 8U);
static_assert(PersistentFfnM32Pipeline::kThreads == 512U);
static_assert(PersistentFfnM32Pipeline::kInputStageBytes == 4608U);
static_assert(PersistentFfnM32Pipeline::kWeightStageBytes == 18432U);
static_assert(PersistentFfnM32Pipeline::kRingBytes == 8704U);
static_assert(sizeof(PersistentFfnM32Storage) <= 32U * 1024U);

struct alignas(128) PersistentFfnM16Storage {
    struct alignas(128) InputStage {
        std::uint32_t values[
            (PersistentFfnM16Pipeline::kDModel / 64U) * 32U * 4U];
        std::uint32_t scales[
            (PersistentFfnM16Pipeline::kRows *
             (PersistentFfnM16Pipeline::kDModel / 16U)) /
            4U];
    } input;
    struct alignas(128) RingBuffer {
#if !STREAM1_SM120_DIRECT_FF2_FRAGMENT_PACK
        std::uint8_t values[PersistentFfnM16Pipeline::kValueBytesPerBuffer];
#endif
        std::uint8_t scales[PersistentFfnM16Pipeline::kScaleBytesPerBuffer];
        std::uint32_t fragments[
            PersistentFfnM16Pipeline::kFragmentWordsPerBuffer];
    } ring[PersistentFfnM16Pipeline::kRingBuffers];
    BlockBarrier ready[PersistentFfnM16Pipeline::kRingBuffers];
    BlockBarrier free[PersistentFfnM16Pipeline::kRingBuffers];
};

// Inverse of the SM120 A-fragment prepack mapping used by load_a_prepacked:
// [M16, K64, lane32, register4, nibble8].  FF1 already assigns each lane a
// unique adjacent FP4 pair, so the packed byte can be written directly into
// the physical FF2 operand without a row-major scratch tensor or atomics.
__device__ inline void store_ff2_fragment_pair(
    std::uint32_t* fragments, std::uint32_t row, std::uint32_t column,
    std::uint8_t packed) {
    const std::uint32_t tile_k = column / 64U;
    const std::uint32_t local_column = column % 64U;
    const std::uint32_t thread_0 = (local_column % 32U) / 8U;
    const std::uint32_t thread_1 = row % 8U;
    const std::uint32_t lane = thread_0 + thread_1 * 4U;
    const std::uint32_t reg = (row / 8U) + 2U * (local_column / 32U);
    const std::uint32_t item = local_column % 8U;
    const std::size_t nibble =
        (((static_cast<std::size_t>(tile_k) * 32U + lane) * 4U + reg) * 8U) +
        item;
    reinterpret_cast<std::uint8_t*>(fragments)[nibble / 2U] = packed;
}

__global__ void persistent_fused_ffn_kernel(
    const std::uint8_t* input_values,
    const std::uint8_t* input_scales_compensated,
    const std::uint8_t* ff1_values,
    const std::uint8_t* ff1_scales,
    const float* ff1_bias,
    const std::uint8_t* ff2_values,
    const std::uint8_t* ff2_scales,
    const float* ff2_bias,
    float* output,
    std::uint32_t rows) {
    using namespace stream1_sm120_nvfp4_raw;
    using Config = PersistentFfnM16Pipeline;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<PersistentFfnM16Storage*>(shared_bytes);
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const bool producer = warp < Config::kProducerWarps;
    const std::uint32_t global_row_begin = blockIdx.x * Config::kRows;

    if (threadIdx.x == 0U) {
        for (std::uint32_t buffer = 0; buffer < Config::kRingBuffers; ++buffer) {
            init(&shared.ready[buffer], Config::kProducerThreads);
            init(&shared.free[buffer], Config::kConsumerThreads);
        }
    }
    constexpr std::uint32_t input_fragment_words =
        (Config::kDModel / 64U) * 32U * 4U;
    const auto* input_words = reinterpret_cast<const std::uint32_t*>(input_values) +
        static_cast<std::size_t>(blockIdx.x) * input_fragment_words;
    for (std::uint32_t word = threadIdx.x; word < input_fragment_words;
         word += blockDim.x) {
        shared.input.values[word] = input_words[word];
    }
    constexpr std::uint32_t input_scale_words =
        (Config::kRows * (Config::kDModel / 16U)) / 4U;
    const auto* input_scale_words_global =
        reinterpret_cast<const std::uint32_t*>(input_scales_compensated) +
        static_cast<std::size_t>(global_row_begin) *
            (Config::kDModel / 16U) / 4U;
    for (std::uint32_t word = threadIdx.x; word < input_scale_words;
         word += blockDim.x) {
        shared.input.scales[word] = input_scale_words_global[word];
    }
    __syncthreads();

    if (producer) {
        constexpr std::uint32_t groups_per_warp = 8U / Config::kProducerWarps;
        static_assert(groups_per_warp * Config::kProducerWarps == 8U);
        for (std::uint32_t slice = 0; slice < Config::kHiddenSlices; ++slice) {
            const std::uint32_t buffer = slice % Config::kRingBuffers;
            const std::uint32_t generation = slice / Config::kRingBuffers;
            if (generation != 0U) {
                shared.free[buffer].wait_parity((generation - 1U) & 1U);
            }
            float acc0[groups_per_warp][4] = {};
            float acc1[groups_per_warp][4] = {};
            for (std::uint32_t k_begin = 0; k_begin < Config::kDModel;
                 k_begin += 64U) {
                std::uint32_t a[4];
                load_a_prepacked(shared.input.values, Config::kDModel, 0U,
                                 k_begin, lane, a);
                const auto sfa = load_sfa_pair(
                    reinterpret_cast<const std::uint8_t*>(shared.input.scales),
                    Config::kDModel / 16U, 0U, k_begin, lane);
#pragma unroll
                for (std::uint32_t slot = 0; slot < groups_per_warp; ++slot) {
                    const std::uint32_t group =
                        slice * 8U + warp * groups_per_warp + slot;
                    std::uint32_t b0[2];
                    std::uint32_t b1[2];
                    load_b_prepacked(
                        reinterpret_cast<const std::uint32_t*>(ff1_values),
                        Config::kFfDim, Config::kDModel, group * 16U,
                        k_begin, lane, b0);
                    load_b_prepacked(
                        reinterpret_cast<const std::uint32_t*>(ff1_values),
                        Config::kFfDim, Config::kDModel, group * 16U + 8U,
                        k_begin, lane, b1);
                    const auto sfb0 = load_sfb_pair(
                        ff1_scales, Config::kDModel / 16U,
                        group * 16U, k_begin, lane);
                    const auto sfb1 = load_sfb_pair(
                        ff1_scales, Config::kDModel / 16U,
                        group * 16U + 8U, k_begin, lane);
                    Mma::fma(
                        acc0[slot][0], acc0[slot][1], acc0[slot][2], acc0[slot][3],
                        a[0], a[1], a[2], a[3], b0[0], b0[1],
                        acc0[slot][0], acc0[slot][1], acc0[slot][2], acc0[slot][3],
                        sfa, sfb0);
                    Mma::fma(
                        acc1[slot][0], acc1[slot][1], acc1[slot][2], acc1[slot][3],
                        a[0], a[1], a[2], a[3], b1[0], b1[1],
                        acc1[slot][0], acc1[slot][1], acc1[slot][2], acc1[slot][3],
                        sfa, sfb1);
                }
            }
#pragma unroll
            for (std::uint32_t slot = 0; slot < groups_per_warp; ++slot) {
                const std::uint32_t group_local = warp * groups_per_warp + slot;
                const std::uint32_t group_global = slice * 8U + group_local;
                const std::uint32_t lane_in_group = lane & 3U;
                const std::uint32_t row_low = lane >> 2U;
                const std::uint32_t row_high = row_low + 8U;
                const std::uint32_t global_col_pair =
                    group_global * 16U + lane_in_group * 2U;
                float low[4] = {
                    fmaxf(acc0[slot][0] + ff1_bias[global_col_pair], 0.0F),
                    fmaxf(acc0[slot][1] + ff1_bias[global_col_pair + 1U], 0.0F),
                    fmaxf(acc1[slot][0] + ff1_bias[global_col_pair + 8U], 0.0F),
                    fmaxf(acc1[slot][1] + ff1_bias[global_col_pair + 9U], 0.0F)};
                float high[4] = {
                    fmaxf(acc0[slot][2] + ff1_bias[global_col_pair], 0.0F),
                    fmaxf(acc0[slot][3] + ff1_bias[global_col_pair + 1U], 0.0F),
                    fmaxf(acc1[slot][2] + ff1_bias[global_col_pair + 8U], 0.0F),
                    fmaxf(acc1[slot][3] + ff1_bias[global_col_pair + 9U], 0.0F)};
                float max_low = fmaxf(fmaxf(low[0], low[1]), fmaxf(low[2], low[3]));
                float max_high =
                    fmaxf(fmaxf(high[0], high[1]), fmaxf(high[2], high[3]));
                max_low = fmaxf(max_low, __shfl_xor_sync(0xffffffffU, max_low, 1, 4));
                max_low = fmaxf(max_low, __shfl_xor_sync(0xffffffffU, max_low, 2, 4));
                max_high = fmaxf(max_high, __shfl_xor_sync(0xffffffffU, max_high, 1, 4));
                max_high = fmaxf(max_high, __shfl_xor_sync(0xffffffffU, max_high, 2, 4));
                const cutlass::float_ue4m3_t scale_low(
                    max_low == 0.0F ? 1.0F : max_low / 6.0F);
                const cutlass::float_ue4m3_t scale_high(
                    max_high == 0.0F ? 1.0F : max_high / 6.0F);
                const float reciprocal_low = __frcp_rn(static_cast<float>(scale_low));
                const float reciprocal_high = __frcp_rn(static_cast<float>(scale_high));
                if (lane_in_group == 0U) {
                    shared.ring[buffer].scales[row_low * 8U + group_local] =
                        scale_low.raw();
                    shared.ring[buffer].scales[row_high * 8U + group_local] =
                        scale_high.raw();
                }
                const std::size_t byte_low =
                    (static_cast<std::size_t>(row_low) * Config::kSliceColumns +
                     group_local * 16U) /
                        2U +
                    lane_in_group;
                const std::size_t byte_high =
                    (static_cast<std::size_t>(row_high) * Config::kSliceColumns +
                     group_local * 16U) /
                        2U +
                    lane_in_group;
                const std::uint8_t packed_low_0 =
                    stream1_sm120_nvfp4_fused::pack_e2m1_pair_scaled(
                        low[0], low[1], reciprocal_low);
                const std::uint8_t packed_low_1 =
                    stream1_sm120_nvfp4_fused::pack_e2m1_pair_scaled(
                        low[2], low[3], reciprocal_low);
                const std::uint8_t packed_high_0 =
                    stream1_sm120_nvfp4_fused::pack_e2m1_pair_scaled(
                        high[0], high[1], reciprocal_high);
                const std::uint8_t packed_high_1 =
                    stream1_sm120_nvfp4_fused::pack_e2m1_pair_scaled(
                        high[2], high[3], reciprocal_high);
#if STREAM1_SM120_DIRECT_FF2_FRAGMENT_PACK
                store_ff2_fragment_pair(
                    shared.ring[buffer].fragments, row_low,
                    group_local * 16U + lane_in_group * 2U, packed_low_0);
                store_ff2_fragment_pair(
                    shared.ring[buffer].fragments, row_low,
                    group_local * 16U + lane_in_group * 2U + 8U,
                    packed_low_1);
                store_ff2_fragment_pair(
                    shared.ring[buffer].fragments, row_high,
                    group_local * 16U + lane_in_group * 2U, packed_high_0);
                store_ff2_fragment_pair(
                    shared.ring[buffer].fragments, row_high,
                    group_local * 16U + lane_in_group * 2U + 8U,
                    packed_high_1);
#else
                shared.ring[buffer].values[byte_low] = packed_low_0;
                shared.ring[buffer].values[byte_low + 4U] = packed_low_1;
                shared.ring[buffer].values[byte_high] = packed_high_0;
                shared.ring[buffer].values[byte_high + 4U] = packed_high_1;
#endif
            }
#if !STREAM1_SM120_DIRECT_FF2_FRAGMENT_PACK
            asm volatile("bar.sync 1, 128;" : : : "memory");
            for (std::uint32_t word_index = threadIdx.x;
                 word_index < Config::kFragmentWordsPerBuffer;
                 word_index += Config::kProducerThreads) {
                const std::uint32_t reg = word_index & 3U;
                const std::uint32_t lane_index = (word_index >> 2U) & 31U;
                const std::uint32_t tile_k = word_index / (32U * 4U);
                const std::uint32_t thread_0 = lane_index & 3U;
                const std::uint32_t thread_1 = lane_index >> 2U;
                std::uint32_t word = 0U;
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    const std::uint32_t value =
                        reg * 8U + static_cast<std::uint32_t>(item);
                    const std::uint32_t value_0 = value & 7U;
                    const std::uint32_t value_1 = (value >> 3U) & 1U;
                    const std::uint32_t value_2 = value >> 4U;
                    const std::uint32_t linear = thread_0 * 128U + thread_1 +
                        value_0 * 16U + value_1 * 8U + value_2 * 512U;
                    const std::uint32_t row = linear % 16U;
                    const std::uint32_t col = tile_k * 64U + linear / 16U;
                    word |= static_cast<std::uint32_t>(
                                packed_nibble(
                                    shared.ring[buffer].values,
                                    static_cast<std::size_t>(row) *
                                            Config::kSliceColumns +
                                        col))
                        << (item * 4U);
                }
                shared.ring[buffer].fragments[word_index] = word;
            }
#endif
            (void)shared.ready[buffer].arrive();
        }
        return;
    }

    constexpr std::uint32_t output_tiles_per_warp =
        (Config::kDModel / 8U) / Config::kConsumerWarps;
    float output_acc[output_tiles_per_warp][4] = {};
    const std::uint32_t consumer_warp = warp - Config::kProducerWarps;
    for (std::uint32_t slice = 0; slice < Config::kHiddenSlices; ++slice) {
        const std::uint32_t buffer = slice % Config::kRingBuffers;
        const std::uint32_t generation = slice / Config::kRingBuffers;
        shared.ready[buffer].wait_parity(generation & 1U);
#if STREAM1_SM120_HOIST_FF2_A
#pragma unroll
        for (std::uint32_t local_k = 0; local_k < Config::kSliceColumns;
             local_k += 64U) {
            // A/SFA are identical for every output N tile.  Keep the packed
            // hidden fragment in registers across all eight independent B
            // tiles instead of re-reading shared memory eight times.
            std::uint32_t a[4];
            load_a_prepacked(shared.ring[buffer].fragments,
                             Config::kSliceColumns, 0U, local_k, lane, a);
            const auto sfa = load_sfa_pair(
                shared.ring[buffer].scales,
                Config::kSliceColumns / 16U, 0U, local_k, lane);
#pragma unroll
            for (std::uint32_t local_tile = 0;
                 local_tile < output_tiles_per_warp; ++local_tile) {
                const std::uint32_t tile_n =
                    consumer_warp + local_tile * Config::kConsumerWarps;
                std::uint32_t b[2];
                load_b_prepacked(
                    reinterpret_cast<const std::uint32_t*>(ff2_values),
                    Config::kDModel, Config::kFfDim, tile_n * 8U,
                    slice * Config::kSliceColumns + local_k, lane, b);
                const auto sfb = load_sfb_pair(
                    ff2_scales, Config::kFfDim / 16U, tile_n * 8U,
                    slice * Config::kSliceColumns + local_k, lane);
                Mma::fma(
                    output_acc[local_tile][0], output_acc[local_tile][1],
                    output_acc[local_tile][2], output_acc[local_tile][3],
                    a[0], a[1], a[2], a[3], b[0], b[1],
                    output_acc[local_tile][0], output_acc[local_tile][1],
                    output_acc[local_tile][2], output_acc[local_tile][3],
                    sfa, sfb);
            }
        }
#else
#pragma unroll
        for (std::uint32_t local_tile = 0;
             local_tile < output_tiles_per_warp; ++local_tile) {
            const std::uint32_t tile_n =
                consumer_warp + local_tile * Config::kConsumerWarps;
#pragma unroll
            for (std::uint32_t local_k = 0; local_k < Config::kSliceColumns;
                 local_k += 64U) {
                std::uint32_t a[4];
                std::uint32_t b[2];
                load_a_prepacked(shared.ring[buffer].fragments,
                                 Config::kSliceColumns, 0U, local_k, lane, a);
                load_b_prepacked(
                    reinterpret_cast<const std::uint32_t*>(ff2_values),
                    Config::kDModel, Config::kFfDim, tile_n * 8U,
                    slice * Config::kSliceColumns + local_k, lane, b);
                const auto sfa = load_sfa_pair(
                    shared.ring[buffer].scales,
                    Config::kSliceColumns / 16U, 0U, local_k, lane);
                const auto sfb = load_sfb_pair(
                    ff2_scales, Config::kFfDim / 16U, tile_n * 8U,
                    slice * Config::kSliceColumns + local_k, lane);
                Mma::fma(
                    output_acc[local_tile][0], output_acc[local_tile][1],
                    output_acc[local_tile][2], output_acc[local_tile][3],
                    a[0], a[1], a[2], a[3], b[0], b[1],
                    output_acc[local_tile][0], output_acc[local_tile][1],
                    output_acc[local_tile][2], output_acc[local_tile][3],
                    sfa, sfb);
            }
        }
#endif
        (void)shared.free[buffer].arrive();
    }

    const std::uint32_t thread_0 = lane & 3U;
    const std::uint32_t thread_1 = lane >> 2U;
#pragma unroll
    for (std::uint32_t local_tile = 0; local_tile < output_tiles_per_warp;
         ++local_tile) {
        const std::uint32_t tile_n =
            consumer_warp + local_tile * Config::kConsumerWarps;
#pragma unroll
        for (int value = 0; value < 4; ++value) {
            const std::uint32_t linear = thread_0 * 32U + thread_1 +
                (static_cast<std::uint32_t>(value) & 1U) * 16U +
                (static_cast<std::uint32_t>(value) >> 1U) * 8U;
            const std::uint32_t row = linear % 16U;
            const std::uint32_t col = tile_n * 8U + linear / 16U;
            const std::uint32_t global_row = global_row_begin + row;
            if (global_row < rows) {
                output[static_cast<std::size_t>(global_row) * Config::kDModel + col] =
                    output_acc[local_tile][value] + ff2_bias[col];
            }
        }
    }
}

// Fixed-row executable used to validate the production CTA topology before
// TMA replaces the cooperative global loads. Four producer warps cooperate on
// each M16 row tile and four consumer warps split its 32 output N8 tiles. The
// hidden activation never leaves shared memory.
__global__ void persistent_fused_ffn_m32_kernel(
    const std::uint8_t* input_values,
    const std::uint8_t* input_scales_compensated,
    const std::uint8_t* ff1_values,
    const std::uint8_t* ff1_scales,
    const float* ff1_bias,
    const std::uint8_t* ff2_values,
    const std::uint8_t* ff2_scales,
    const float* ff2_bias,
    float* output,
    std::uint32_t rows) {
    using namespace stream1_sm120_nvfp4_raw;
    using Config = PersistentFfnM32Pipeline;
    extern __shared__ __align__(128) unsigned char shared_bytes[];
    auto& shared = *reinterpret_cast<PersistentFfnM32Storage*>(shared_bytes);
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    const bool producer = warp < Config::kProducerWarps;
    const std::uint32_t global_row_begin = blockIdx.x * Config::kRows;

    if (threadIdx.x == 0U) {
        for (std::uint32_t buffer = 0; buffer < Config::kRingBuffers; ++buffer) {
            init(&shared.ready[buffer], Config::kProducerWarps * 32U);
            init(&shared.free[buffer], Config::kConsumerWarps * 32U);
        }
    }
    const auto* input_words = reinterpret_cast<const std::uint32_t*>(input_values) +
        static_cast<std::size_t>(blockIdx.x) *
            (Config::kInputValueBytes / sizeof(std::uint32_t));
    for (std::uint32_t word = threadIdx.x;
         word < Config::kInputValueBytes / sizeof(std::uint32_t);
         word += blockDim.x) {
        shared.input.fragments[word] = input_words[word];
    }
    const auto* input_scales = input_scales_compensated +
        static_cast<std::size_t>(global_row_begin) * (Config::kDModel / 16U);
    for (std::uint32_t byte = threadIdx.x; byte < Config::kInputScaleBytes;
         byte += blockDim.x) {
        shared.input.scales[byte] = input_scales[byte];
    }
    __syncthreads();

    if (producer) {
        constexpr std::uint32_t kInputFragmentWordsPerTile =
            (Config::kDModel / 64U) * 32U * 4U;
        constexpr std::uint32_t kFragmentWordsPerTile =
            Config::kRowsPerWarp * Config::kSliceColumns / 8U;
        constexpr std::uint32_t kValueBytesPerTile =
            Config::kRowsPerWarp * Config::kSliceColumns / 2U;
        constexpr std::uint32_t kScaleBytesPerTile =
            Config::kRowsPerWarp * (Config::kSliceColumns / 16U);
        const std::uint32_t row_tile = warp >> 2U;
        const std::uint32_t producer_in_tile = warp & 3U;
        constexpr std::uint32_t kGroupsPerProducer = 2U;
        const auto* input_fragments = shared.input.fragments +
            row_tile * kInputFragmentWordsPerTile;
        const auto* input_scale_tile = shared.input.scales +
            row_tile * Config::kRowsPerWarp * (Config::kDModel / 16U);

        for (std::uint32_t slice = 0; slice < Config::kHiddenSlices; ++slice) {
            const std::uint32_t buffer = slice % Config::kRingBuffers;
            const std::uint32_t generation = slice / Config::kRingBuffers;
            if (generation != 0U) {
                shared.free[buffer].wait_parity((generation - 1U) & 1U);
            }
            float acc0[kGroupsPerProducer][4] = {};
            float acc1[kGroupsPerProducer][4] = {};
            for (std::uint32_t k_begin = 0; k_begin < Config::kDModel;
                 k_begin += 64U) {
                std::uint32_t a[4];
                load_a_prepacked(input_fragments, Config::kDModel, 0U,
                                 k_begin, lane, a);
                const auto sfa = load_sfa_pair(
                    input_scale_tile, Config::kDModel / 16U, 0U, k_begin, lane);
#pragma unroll
                for (std::uint32_t slot = 0; slot < kGroupsPerProducer;
                     ++slot) {
                    const std::uint32_t group_local =
                        producer_in_tile * kGroupsPerProducer + slot;
                    const std::uint32_t group = slice * 8U + group_local;
                    std::uint32_t b0[2];
                    std::uint32_t b1[2];
                    load_b_prepacked(
                        reinterpret_cast<const std::uint32_t*>(ff1_values),
                        Config::kFfDim, Config::kDModel, group * 16U,
                        k_begin, lane, b0);
                    load_b_prepacked(
                        reinterpret_cast<const std::uint32_t*>(ff1_values),
                        Config::kFfDim, Config::kDModel, group * 16U + 8U,
                        k_begin, lane, b1);
                    const auto sfb0 = load_sfb_pair(
                        ff1_scales, Config::kDModel / 16U,
                        group * 16U, k_begin, lane);
                    const auto sfb1 = load_sfb_pair(
                        ff1_scales, Config::kDModel / 16U,
                        group * 16U + 8U, k_begin, lane);
                    Mma::fma(acc0[slot][0], acc0[slot][1],
                             acc0[slot][2], acc0[slot][3],
                             a[0], a[1], a[2], a[3], b0[0], b0[1],
                             acc0[slot][0], acc0[slot][1],
                             acc0[slot][2], acc0[slot][3],
                             sfa, sfb0);
                    Mma::fma(acc1[slot][0], acc1[slot][1],
                             acc1[slot][2], acc1[slot][3],
                             a[0], a[1], a[2], a[3], b1[0], b1[1],
                             acc1[slot][0], acc1[slot][1],
                             acc1[slot][2], acc1[slot][3],
                             sfa, sfb1);
                }
            }
            auto* values = shared.ring[buffer].values +
                row_tile * kValueBytesPerTile;
            auto* scales = shared.ring[buffer].scales +
                row_tile * kScaleBytesPerTile;
#pragma unroll
            for (std::uint32_t slot = 0; slot < kGroupsPerProducer; ++slot) {
                const std::uint32_t group_local =
                    producer_in_tile * kGroupsPerProducer + slot;
                const std::uint32_t group_global = slice * 8U + group_local;
                const std::uint32_t lane_in_group = lane & 3U;
                const std::uint32_t row_low = lane >> 2U;
                const std::uint32_t row_high = row_low + 8U;
                const std::uint32_t col_pair =
                    group_global * 16U + lane_in_group * 2U;
                float low[4] = {
                    fmaxf(acc0[slot][0] + ff1_bias[col_pair], 0.0F),
                    fmaxf(acc0[slot][1] + ff1_bias[col_pair + 1U], 0.0F),
                    fmaxf(acc1[slot][0] + ff1_bias[col_pair + 8U], 0.0F),
                    fmaxf(acc1[slot][1] + ff1_bias[col_pair + 9U], 0.0F)};
                float high[4] = {
                    fmaxf(acc0[slot][2] + ff1_bias[col_pair], 0.0F),
                    fmaxf(acc0[slot][3] + ff1_bias[col_pair + 1U], 0.0F),
                    fmaxf(acc1[slot][2] + ff1_bias[col_pair + 8U], 0.0F),
                    fmaxf(acc1[slot][3] + ff1_bias[col_pair + 9U], 0.0F)};
                float max_low = fmaxf(fmaxf(low[0], low[1]), fmaxf(low[2], low[3]));
                float max_high =
                    fmaxf(fmaxf(high[0], high[1]), fmaxf(high[2], high[3]));
                max_low = fmaxf(max_low,
                    __shfl_xor_sync(0xffffffffU, max_low, 1, 4));
                max_low = fmaxf(max_low,
                    __shfl_xor_sync(0xffffffffU, max_low, 2, 4));
                max_high = fmaxf(max_high,
                    __shfl_xor_sync(0xffffffffU, max_high, 1, 4));
                max_high = fmaxf(max_high,
                    __shfl_xor_sync(0xffffffffU, max_high, 2, 4));
                const cutlass::float_ue4m3_t scale_low(
                    max_low == 0.0F ? 1.0F : max_low / 6.0F);
                const cutlass::float_ue4m3_t scale_high(
                    max_high == 0.0F ? 1.0F : max_high / 6.0F);
                const float inv_low = __frcp_rn(static_cast<float>(scale_low));
                const float inv_high = __frcp_rn(static_cast<float>(scale_high));
                if (lane_in_group == 0U) {
                    scales[row_low * 8U + group_local] = scale_low.raw();
                    scales[row_high * 8U + group_local] = scale_high.raw();
                }
                const std::size_t byte_low =
                    (static_cast<std::size_t>(row_low) * Config::kSliceColumns +
                     group_local * 16U) / 2U + lane_in_group;
                const std::size_t byte_high =
                    (static_cast<std::size_t>(row_high) * Config::kSliceColumns +
                     group_local * 16U) / 2U + lane_in_group;
                values[byte_low] = stream1_sm120_nvfp4_fused::
                    pack_e2m1_pair_scaled(low[0], low[1], inv_low);
                values[byte_low + 4U] = stream1_sm120_nvfp4_fused::
                    pack_e2m1_pair_scaled(low[2], low[3], inv_low);
                values[byte_high] = stream1_sm120_nvfp4_fused::
                    pack_e2m1_pair_scaled(high[0], high[1], inv_high);
                values[byte_high + 4U] = stream1_sm120_nvfp4_fused::
                    pack_e2m1_pair_scaled(high[2], high[3], inv_high);
            }
            asm volatile("bar.sync 1, 256;" : : : "memory");
            constexpr std::uint32_t kFragmentWords =
                Config::kRingValueBytes / sizeof(std::uint32_t);
            for (std::uint32_t word_index = threadIdx.x;
                 word_index < kFragmentWords;
                 word_index += Config::kProducerWarps * 32U) {
                const std::uint32_t fragment_row_tile =
                    word_index / kFragmentWordsPerTile;
                const std::uint32_t local_word =
                    word_index % kFragmentWordsPerTile;
                const auto* tile_values = shared.ring[buffer].values +
                    fragment_row_tile * kValueBytesPerTile;
                const std::uint32_t reg = word_index & 3U;
                const std::uint32_t lane_index = (local_word >> 2U) & 31U;
                const std::uint32_t tile_k = local_word / (32U * 4U);
                const std::uint32_t thread_0 = lane_index & 3U;
                const std::uint32_t thread_1 = lane_index >> 2U;
                std::uint32_t word = 0U;
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    const std::uint32_t value = reg * 8U + item;
                    const std::uint32_t linear = thread_0 * 128U + thread_1 +
                        (value & 7U) * 16U + ((value >> 3U) & 1U) * 8U +
                        (value >> 4U) * 512U;
                    const std::uint32_t row = linear % 16U;
                    const std::uint32_t col = tile_k * 64U + linear / 16U;
                    word |= static_cast<std::uint32_t>(packed_nibble(
                        tile_values, static_cast<std::size_t>(row) *
                            Config::kSliceColumns + col)) << (item * 4U);
                }
                shared.ring[buffer].fragments[word_index] = word;
            }
            (void)shared.ready[buffer].arrive();
        }
        return;
    }

    constexpr std::uint32_t kOutputTilesPerWarp = 8U;
    constexpr std::uint32_t kFragmentWordsPerTile =
        Config::kRowsPerWarp * Config::kSliceColumns / 8U;
    constexpr std::uint32_t kScaleBytesPerTile =
        Config::kRowsPerWarp * (Config::kSliceColumns / 16U);
    float output_acc[kOutputTilesPerWarp][4] = {};
    const std::uint32_t consumer_warp = warp - Config::kProducerWarps;
    const std::uint32_t row_tile = consumer_warp >> 2U;
    const std::uint32_t output_quarter = consumer_warp & 3U;
    for (std::uint32_t slice = 0; slice < Config::kHiddenSlices; ++slice) {
        const std::uint32_t buffer = slice % Config::kRingBuffers;
        const std::uint32_t generation = slice / Config::kRingBuffers;
        shared.ready[buffer].wait_parity(generation & 1U);
        const auto* fragments = shared.ring[buffer].fragments +
            row_tile * kFragmentWordsPerTile;
        const auto* scales = shared.ring[buffer].scales +
            row_tile * kScaleBytesPerTile;
#pragma unroll
        for (std::uint32_t local_tile = 0; local_tile < kOutputTilesPerWarp;
             ++local_tile) {
            const std::uint32_t tile_n = output_quarter * 8U + local_tile;
#pragma unroll
            for (std::uint32_t local_k = 0; local_k < Config::kSliceColumns;
                 local_k += 64U) {
                std::uint32_t a[4];
                std::uint32_t b[2];
                load_a_prepacked(fragments, Config::kSliceColumns,
                                 0U, local_k, lane, a);
                load_b_prepacked(
                    reinterpret_cast<const std::uint32_t*>(ff2_values),
                    Config::kDModel, Config::kFfDim, tile_n * 8U,
                    slice * Config::kSliceColumns + local_k, lane, b);
                const auto sfa = load_sfa_pair(
                    scales, Config::kSliceColumns / 16U, 0U, local_k, lane);
                const auto sfb = load_sfb_pair(
                    ff2_scales, Config::kFfDim / 16U, tile_n * 8U,
                    slice * Config::kSliceColumns + local_k, lane);
                Mma::fma(output_acc[local_tile][0], output_acc[local_tile][1],
                         output_acc[local_tile][2], output_acc[local_tile][3],
                         a[0], a[1], a[2], a[3], b[0], b[1],
                         output_acc[local_tile][0], output_acc[local_tile][1],
                         output_acc[local_tile][2], output_acc[local_tile][3],
                         sfa, sfb);
            }
        }
        (void)shared.free[buffer].arrive();
    }

    const std::uint32_t thread_0 = lane & 3U;
    const std::uint32_t thread_1 = lane >> 2U;
#pragma unroll
    for (std::uint32_t local_tile = 0; local_tile < kOutputTilesPerWarp;
         ++local_tile) {
        const std::uint32_t tile_n = output_quarter * 8U + local_tile;
#pragma unroll
        for (int value = 0; value < 4; ++value) {
            const std::uint32_t linear = thread_0 * 32U + thread_1 +
                (static_cast<std::uint32_t>(value) & 1U) * 16U +
                (static_cast<std::uint32_t>(value) >> 1U) * 8U;
            const std::uint32_t row = linear % 16U;
            const std::uint32_t col = tile_n * 8U + linear / 16U;
            const std::uint32_t global_row = global_row_begin +
                row_tile * Config::kRowsPerWarp + row;
            if (global_row < rows) {
                output[static_cast<std::size_t>(global_row) *
                    Config::kDModel + col] =
                    output_acc[local_tile][value] + ff2_bias[col];
            }
        }
    }
}

}  // namespace stream1_sm120_nvfp4_persistent
