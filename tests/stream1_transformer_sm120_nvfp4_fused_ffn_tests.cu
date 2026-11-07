#include "stream1_transformer_sm120_nvfp4_fused_ffn.cuh"
#include "stream1_transformer_sm120_nvfp4_persistent_pipeline.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

namespace {
using cutlass::float_e2m1_t;
using cutlass::float_ue4m3_t;
constexpr std::uint32_t M = 16, D = 256, F = 1024;

void require(bool ok, const char* message) { if (!ok) { std::cerr << message << '\n'; std::exit(1); } }
float fp4(std::uint8_t raw) { return static_cast<float>(float_e2m1_t::bitcast(raw)); }

struct Packed {
    std::vector<std::uint8_t> values;
    std::vector<std::uint8_t> scales;
};

Packed pack(const std::vector<float>& source, std::uint32_t rows, std::uint32_t cols,
            bool compensate) {
    Packed result{{}, {}};
    result.values.assign(static_cast<std::size_t>(rows) * cols / 2U, 0U);
    result.scales.resize(static_cast<std::size_t>(rows) * (cols / 16U));
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t begin = 0; begin < cols; begin += 16U) {
            float maximum = 0.0F;
            for (std::uint32_t col = begin; col < begin + 16U; ++col)
                maximum = std::max(maximum, std::abs(source[static_cast<std::size_t>(row) * cols + col]));
            const float_ue4m3_t encoded((compensate ? 2.0F : 1.0F) *
                (maximum == 0.0F ? 1.0F : maximum / 6.0F));
            const float logical = static_cast<float>(encoded) / (compensate ? 2.0F : 1.0F);
            result.scales[static_cast<std::size_t>(row) * (cols / 16U) + begin / 16U] = encoded.raw();
            for (std::uint32_t col = begin; col < begin + 16U; ++col) {
                const std::size_t index = static_cast<std::size_t>(row) * cols + col;
                const std::uint8_t raw = float_e2m1_t(source[index] / logical).raw();
                result.values[index / 2U] |= static_cast<std::uint8_t>(raw << ((index & 1U) * 4U));
            }
        }
    }
    return result;
}

std::vector<float> unpack(const Packed& packed, std::uint32_t rows, std::uint32_t cols,
                          bool compensated) {
    std::vector<float> result(static_cast<std::size_t>(rows) * cols);
    for (std::uint32_t row = 0; row < rows; ++row) for (std::uint32_t col = 0; col < cols; ++col) {
        const std::size_t index = static_cast<std::size_t>(row) * cols + col;
        const auto raw = static_cast<std::uint8_t>((packed.values[index / 2U] >> ((index & 1U) * 4U)) & 15U);
        const float scale = static_cast<float>(float_ue4m3_t::bitcast(
            packed.scales[static_cast<std::size_t>(row) * (cols / 16U) + col / 16U])) /
            (compensated ? 2.0F : 1.0F);
        result[index] = fp4(raw) * scale;
    }
    return result;
}

[[maybe_unused]] std::vector<std::uint8_t> transpose_value_payload(
    const Packed& source, std::uint32_t n, std::uint32_t k) {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(n) * k / 2U, 0U);
    for (std::uint32_t column = 0; column < n; ++column) {
        for (std::uint32_t inner = 0; inner < k; ++inner) {
            const std::size_t from = static_cast<std::size_t>(column) * k + inner;
            const std::size_t to = static_cast<std::size_t>(inner) * n + column;
            const std::uint8_t raw = static_cast<std::uint8_t>(
                (source.values[from / 2U] >> ((from & 1U) * 4U)) & 15U);
            result[to / 2U] |= static_cast<std::uint8_t>(raw << ((to & 1U) * 4U));
        }
    }
    return result;
}

std::vector<std::uint8_t> prepack_b_fragments(
    const Packed& source, std::uint32_t n, std::uint32_t k) {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(n) * k / 2U, 0U);
    auto* words = reinterpret_cast<std::uint32_t*>(result.data());
    for (std::uint32_t tile_n = 0; tile_n < n / 8U; ++tile_n) {
        for (std::uint32_t tile_k = 0; tile_k < k / 64U; ++tile_k) {
            for (std::uint32_t lane = 0; lane < 32U; ++lane) {
                const std::uint32_t thread_0 = lane & 3U;
                const std::uint32_t thread_1 = lane >> 2U;
                for (std::uint32_t reg = 0; reg < 2U; ++reg) {
                    std::uint32_t word = 0U;
                    for (std::uint32_t item = 0; item < 8U; ++item) {
                        const std::uint32_t value = reg * 8U + item;
                        const std::uint32_t linear = thread_0 * 64U + thread_1 +
                            (value & 7U) * 8U + (value >> 3U) * 256U;
                        const std::uint32_t column = tile_n * 8U + linear % 8U;
                        const std::uint32_t inner = tile_k * 64U + linear / 8U;
                        const std::size_t logical = static_cast<std::size_t>(column) * k + inner;
                        const std::uint8_t raw = static_cast<std::uint8_t>(
                            (source.values[logical / 2U] >> ((logical & 1U) * 4U)) & 15U);
                        word |= static_cast<std::uint32_t>(raw) << (item * 4U);
                    }
                    const std::size_t target =
                        ((static_cast<std::size_t>(tile_n) * (k / 64U) + tile_k) * 32U + lane) * 2U + reg;
                    words[target] = word;
                }
            }
        }
    }
    return result;
}

std::vector<std::uint8_t> prepack_a_fragments(
    const Packed& source, std::uint32_t m, std::uint32_t k) {
    std::vector<std::uint8_t> result(static_cast<std::size_t>(m) * k / 2U, 0U);
    auto* words = reinterpret_cast<std::uint32_t*>(result.data());
    for (std::uint32_t tile_m = 0; tile_m < m / 16U; ++tile_m) {
        for (std::uint32_t tile_k = 0; tile_k < k / 64U; ++tile_k) {
            for (std::uint32_t lane = 0; lane < 32U; ++lane) {
                const std::uint32_t thread_0 = lane & 3U;
                const std::uint32_t thread_1 = lane >> 2U;
                for (std::uint32_t reg = 0; reg < 4U; ++reg) {
                    std::uint32_t word = 0U;
                    for (std::uint32_t item = 0; item < 8U; ++item) {
                        const std::uint32_t value = reg * 8U + item;
                        const std::uint32_t linear = thread_0 * 128U + thread_1 +
                            (value & 7U) * 16U + ((value >> 3U) & 1U) * 8U +
                            (value >> 4U) * 512U;
                        const std::uint32_t row = tile_m * 16U + linear % 16U;
                        const std::uint32_t inner = tile_k * 64U + linear / 16U;
                        const std::size_t logical = static_cast<std::size_t>(row) * k + inner;
                        const std::uint8_t raw = static_cast<std::uint8_t>(
                            (source.values[logical / 2U] >> ((logical & 1U) * 4U)) & 15U);
                        word |= static_cast<std::uint32_t>(raw) << (item * 4U);
                    }
                    const std::size_t target =
                        ((static_cast<std::size_t>(tile_m) * (k / 64U) + tile_k) * 32U + lane) * 4U + reg;
                    words[target] = word;
                }
            }
        }
    }
    return result;
}

template<class T> T* device_copy(const std::vector<T>& host) {
    T* pointer = nullptr; cudaMalloc(&pointer, host.size() * sizeof(T));
    cudaMemcpy(pointer, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice); return pointer;
}

__global__ void ff1_k64_probe_kernel(
    const std::uint8_t* input_values,
    const std::uint8_t* input_scales,
    const std::uint8_t* weight_values,
    const std::uint8_t* weight_scales,
    float* partials) {
    using namespace stream1_sm120_nvfp4_raw;
    const std::uint32_t lane = threadIdx.x & 31U;
    const std::uint32_t warp = threadIdx.x >> 5U;
    for (std::uint32_t tile_n = warp; tile_n < F / 8U; tile_n += blockDim.x / 32U) {
        for (std::uint32_t chunk = 0; chunk < D / 64U; ++chunk) {
            float accumulators[4] = {0.0F, 0.0F, 0.0F, 0.0F};
            std::uint32_t a[4];
            std::uint32_t b[2];
            const std::uint32_t k_begin = chunk * 64U;
            load_a_prepacked(
                reinterpret_cast<const std::uint32_t*>(input_values),
                D, 0U, k_begin, lane, a);
            load_b_prepacked(
                reinterpret_cast<const std::uint32_t*>(weight_values),
                F, D, tile_n * 8U, k_begin, lane, b);
            const auto sfa = load_sfa_pair(input_scales, D / 16U, 0U, k_begin, lane);
            const auto sfb = load_sfb_pair(weight_scales, D / 16U, tile_n * 8U, k_begin, lane);
            Mma::fma(
                accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                a[0], a[1], a[2], a[3], b[0], b[1],
                accumulators[0], accumulators[1], accumulators[2], accumulators[3],
                sfa, sfb);
            store_c(partials + static_cast<std::size_t>(chunk) * M * F,
                    F, 0U, tile_n * 8U, lane, accumulators);
        }
    }
}
}  // namespace

int main() {
    std::vector<float> input(M * D), w1(F * D), w2(D * F), b1(F), b2(D);
    for (std::size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 7) - 3) * 0.25F;
    for (std::size_t i = 0; i < w1.size(); ++i) w1[i] = float(int(i % 5) - 2) * 0.125F;
    for (std::size_t i = 0; i < w2.size(); ++i) w2[i] = float(int(i % 7) - 3) * 0.0625F;
    for (std::size_t i = 0; i < b1.size(); ++i) b1[i] = float(int(i % 3) - 1) * 0.125F;
    for (std::size_t i = 0; i < b2.size(); ++i) b2[i] = float(int(i % 3) - 1) * 0.25F;
    // Kernel weights are K-major/N-contiguous.
    std::vector<float> w1_kn(D * F), w2_kn(F * D);
    for (std::uint32_t n=0;n<F;++n) for(std::uint32_t k=0;k<D;++k) w1_kn[k*F+n]=w1[n*D+k];
    for (std::uint32_t n=0;n<D;++n) for(std::uint32_t k=0;k<F;++k) w2_kn[k*D+n]=w2[n*F+k];
    const auto pi=pack(input,M,D,false), p1=pack(w1,F,D,false), p2=pack(w2,D,F,false);
    const auto qi=unpack(pi,M,D,false), q1=unpack(p1,F,D,false), q2=unpack(p2,D,F,false);
    // Transpose already-quantized value nibbles to K-major while retaining
    // the original N-major K32 scale matrix.
    const auto p1_values_kn = prepack_b_fragments(p1, F, D);
    const auto p2_values_kn = prepack_b_fragments(p2, D, F);
    float transpose_error = 0.0F;
    for (std::uint32_t n = 0; n < F; ++n) for (std::uint32_t k = 0; k < D; ++k) {
        const std::uint32_t tile_n = n / 8U, tile_k = k / 64U;
        const std::uint32_t local_n = n % 8U, local_k = k % 64U;
        std::uint8_t raw = 0xffU;
        for (std::uint32_t lane = 0; lane < 32U && raw == 0xffU; ++lane) {
            const std::uint32_t thread_0 = lane & 3U, thread_1 = lane >> 2U;
            for (std::uint32_t value = 0; value < 16U; ++value) {
                const std::uint32_t linear = thread_0 * 64U + thread_1 +
                    (value & 7U) * 8U + (value >> 3U) * 256U;
                if (linear % 8U == local_n && linear / 8U == local_k) {
                    const auto* words = reinterpret_cast<const std::uint32_t*>(p1_values_kn.data());
                    const std::size_t word_index =
                        ((static_cast<std::size_t>(tile_n) * (D / 64U) + tile_k) * 32U + lane) * 2U + value / 8U;
                    raw = static_cast<std::uint8_t>((words[word_index] >> ((value & 7U) * 4U)) & 15U);
                }
            }
        }
        const float scale = static_cast<float>(float_ue4m3_t::bitcast(
            p1.scales[static_cast<std::size_t>(n) * (D / 16U) + k / 16U]));
        transpose_error = std::max(
            transpose_error, std::abs(fp4(raw) * scale - q1[static_cast<std::size_t>(n) * D + k]));
    }
    require(transpose_error == 0.0F, "K-major weight payload transpose mismatch");
    std::vector<float> raw_hidden(M*F), hidden(M*F), reference(M*D);
    for(std::uint32_t m=0;m<M;++m) for(std::uint32_t n=0;n<F;++n){float s=0.0F;for(std::uint32_t k=0;k<D;++k)s+=qi[m*D+k]*q1[n*D+k];raw_hidden[m*F+n]=s;hidden[m*F+n]=std::max(s+b1[n],0.0F);}
    const auto ph=pack(hidden,M,F,false); const auto qh=unpack(ph,M,F,false);
    std::cerr << "ff1_parts_m0_n0";
    for (std::uint32_t begin = 0; begin < D; begin += 64U) {
        float part = 0.0F;
        for (std::uint32_t k = begin; k < begin + 64U; ++k)
            part += qi[k] * q1[k];
        std::cerr << ' ' << part;
    }
    std::cerr << '\n';
    for(std::uint32_t m=0;m<M;++m) for(std::uint32_t n=0;n<D;++n){float s=b2[n];for(std::uint32_t k=0;k<F;++k)s+=qh[m*F+k]*q2[n*F+k];reference[m*D+n]=s;}
    const auto pi_values_fragments = prepack_a_fragments(pi, M, D);
    auto* di = device_copy(pi_values_fragments);
    auto* dsi = device_copy(pi.scales);
    auto* d1 = device_copy(p1_values_kn);
    auto* ds1 = device_copy(p1.scales);
    auto* db1 = device_copy(b1);
    auto* d2 = device_copy(p2_values_kn);
    auto* ds2 = device_copy(p2.scales);
    auto* db2 = device_copy(b2);
    float* dout=nullptr; cudaMalloc(&dout,reference.size()*sizeof(float));
    float* ddebug=nullptr; cudaMalloc(&ddebug,raw_hidden.size()*sizeof(float));
    float* dpartials=nullptr; cudaMalloc(&dpartials,4U*raw_hidden.size()*sizeof(float));
    ff1_k64_probe_kernel<<<1,256>>>(di,dsi,d1,ds1,dpartials);
    require(cudaDeviceSynchronize()==cudaSuccess,"FF1 K64 probe launch failed");
    std::vector<float> gpu_partials(4U*raw_hidden.size());
    cudaMemcpy(gpu_partials.data(),dpartials,gpu_partials.size()*sizeof(float),cudaMemcpyDeviceToHost);
    for (std::uint32_t n = 0; n < 5U; ++n) {
        std::cerr << "ff1_probe_m0_n" << n;
        for (std::uint32_t chunk = 0; chunk < 4U; ++chunk) {
            float cpu = 0.0F;
            for (std::uint32_t k = chunk * 64U; k < (chunk + 1U) * 64U; ++k)
                cpu += qi[k] * q1[static_cast<std::size_t>(n) * D + k];
            std::cerr << " c" << chunk << "="
                      << gpu_partials[static_cast<std::size_t>(chunk) * M * F + n]
                      << "/" << cpu;
        }
        std::cerr << '\n';
    }
    cudaFuncSetAttribute(
        stream1_sm120_nvfp4_fused::fused_ffn_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(stream1_sm120_nvfp4_fused::SharedStorage));
    cudaFuncAttributes attributes{};
    cudaFuncGetAttributes(
        &attributes, stream1_sm120_nvfp4_fused::fused_ffn_kernel);
    std::cout << "fused_ffn_resources registers=" << attributes.numRegs
              << " static_shared=" << attributes.sharedSizeBytes
              << " dynamic_shared=" << sizeof(stream1_sm120_nvfp4_fused::SharedStorage)
              << " local_bytes=" << attributes.localSizeBytes
              << " max_threads=" << attributes.maxThreadsPerBlock << '\n';
    for (const int threads : {128, 256}) {
        int active_blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks, stream1_sm120_nvfp4_fused::fused_ffn_kernel,
            threads, sizeof(stream1_sm120_nvfp4_fused::SharedStorage));
        std::cout << "fused_ffn_occupancy threads=" << threads
                  << " active_blocks_per_sm=" << active_blocks << '\n';
    }
    stream1_sm120_nvfp4_fused::fused_ffn_kernel<<<1,256,sizeof(stream1_sm120_nvfp4_fused::SharedStorage)>>>(di,dsi,d1,ds1,db1,d2,ds2,db2,dout,ddebug,M);
    require(cudaDeviceSynchronize()==cudaSuccess,"fused kernel launch failed");
    std::vector<float> actual(reference.size()), actual_hidden(raw_hidden.size()); cudaMemcpy(actual.data(),dout,actual.size()*sizeof(float),cudaMemcpyDeviceToHost); cudaMemcpy(actual_hidden.data(),ddebug,actual_hidden.size()*sizeof(float),cudaMemcpyDeviceToHost);
    float hidden_error=0; for(std::size_t i=0;i<actual_hidden.size();++i) hidden_error=std::max(hidden_error,std::abs(actual_hidden[i]-raw_hidden[i]));
    for (std::size_t i = 0, shown = 0; i < actual_hidden.size() && shown < 16; ++i) {
        if (actual_hidden[i] != raw_hidden[i]) {
            std::cerr << "hidden_mismatch index=" << i << " actual=" << actual_hidden[i]
                      << " reference=" << raw_hidden[i] << '\n';
            ++shown;
        }
    }
    float error=0; for(std::size_t i=0;i<actual.size();++i) error=std::max(error,std::abs(actual[i]-reference[i]));
    std::cout<<"stream1_transformer_sm120_nvfp4_fused_ffn_tests hidden_max_abs_error="<<hidden_error<<" max_abs_error="<<error<<'\n';
    require(error < 1.0e-4F,"fused FFN numerical mismatch");
    float* dpersistent = nullptr;
    cudaMalloc(&dpersistent, reference.size() * sizeof(float));
    stream1_sm120_nvfp4_persistent::persistent_fused_ffn_kernel<<<
        1, stream1_sm120_nvfp4_persistent::PersistentFfnM16Pipeline::kThreads,
        sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM16Storage)>>>(
        di, dsi, d1, ds1, db1, d2, ds2, db2, dpersistent, M);
    require(cudaDeviceSynchronize() == cudaSuccess,
            "persistent fused FFN launch failed");
    std::vector<float> persistent_actual(reference.size());
    cudaMemcpy(persistent_actual.data(), dpersistent,
               persistent_actual.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float persistent_error = 0.0F;
    for (std::size_t i = 0; i < persistent_actual.size(); ++i) {
        persistent_error = std::max(
            persistent_error, std::abs(persistent_actual[i] - reference[i]));
    }
    std::cout << "persistent_fused_ffn_m16 max_abs_error="
              << persistent_error << '\n';
    require(persistent_error < 1.0e-4F,
            "persistent fused FFN numerical mismatch");

    constexpr std::uint32_t m128_rows = 128U;
    std::vector<float> m128_input(static_cast<std::size_t>(m128_rows) * D);
    for (std::uint32_t row = 0; row < m128_rows; ++row) {
        const std::uint32_t source_row = row % M;
        std::copy(input.begin() + static_cast<std::size_t>(source_row) * D,
                  input.begin() + static_cast<std::size_t>(source_row + 1U) * D,
                  m128_input.begin() + static_cast<std::size_t>(row) * D);
    }
    const auto m128_packed = pack(m128_input, m128_rows, D, false);
    const auto m128_fragments = prepack_a_fragments(m128_packed, m128_rows, D);
    auto* m128_values = device_copy(m128_fragments);
    auto* m128_scales = device_copy(m128_packed.scales);
    float* m128_output = nullptr;
    cudaMalloc(&m128_output,
               static_cast<std::size_t>(m128_rows) * D * sizeof(float));
    cudaMemset(m128_output, 0x7f,
               static_cast<std::size_t>(m128_rows) * D * sizeof(float));
    const cudaError_t m128_attribute_status = cudaFuncSetAttribute(
        stream1_sm120_nvfp4_persistent::persistent_fused_ffn_m32_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM32Storage));
    cudaFuncAttributes m128_attributes{};
    const cudaError_t m128_attributes_status = cudaFuncGetAttributes(
        &m128_attributes,
        stream1_sm120_nvfp4_persistent::persistent_fused_ffn_m32_kernel);
    std::cout << "persistent_fused_ffn_m128_resources attribute_status="
              << cudaGetErrorString(m128_attribute_status)
              << " attributes_status=" << cudaGetErrorString(m128_attributes_status)
              << " registers=" << m128_attributes.numRegs
              << " local_bytes=" << m128_attributes.localSizeBytes
              << " max_threads=" << m128_attributes.maxThreadsPerBlock
              << " dynamic_shared="
              << sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM32Storage)
              << '\n';
    stream1_sm120_nvfp4_persistent::persistent_fused_ffn_m32_kernel<<<
        4, stream1_sm120_nvfp4_persistent::PersistentFfnM32Pipeline::kThreads,
        sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM32Storage)>>>(
        m128_values, m128_scales, d1, ds1, db1, d2, ds2, db2,
        m128_output, m128_rows);
    const cudaError_t m128_launch_status = cudaPeekAtLastError();
    const cudaError_t m128_sync_status = cudaDeviceSynchronize();
    std::cout << "persistent_fused_ffn_m128_launch launch_status="
              << cudaGetErrorString(m128_launch_status)
              << " sync_status=" << cudaGetErrorString(m128_sync_status) << '\n';
    require(m128_launch_status == cudaSuccess && m128_sync_status == cudaSuccess,
            "M128 persistent fused FFN launch failed");
    std::vector<float> m128_actual(static_cast<std::size_t>(m128_rows) * D);
    cudaMemcpy(m128_actual.data(), m128_output,
               m128_actual.size() * sizeof(float), cudaMemcpyDeviceToHost);
    float m128_error = 0.0F;
    std::size_t m128_shown = 0U;
    for (std::uint32_t row = 0; row < m128_rows; ++row) {
        const std::uint32_t source_row = row % M;
        for (std::uint32_t col = 0; col < D; ++col) {
            const float actual_value =
                m128_actual[static_cast<std::size_t>(row) * D + col];
            const float expected_value =
                persistent_actual[static_cast<std::size_t>(source_row) * D + col];
            const float difference = std::abs(actual_value - expected_value);
            m128_error = std::max(m128_error, difference);
            if (difference > 1.0e-4F && m128_shown < 24U) {
                std::cerr << "m128_mismatch row=" << row << " col=" << col
                          << " actual=" << actual_value
                          << " expected=" << expected_value
                          << " error=" << difference << '\n';
                ++m128_shown;
            }
        }
    }
    std::cout << "persistent_fused_ffn_m128 max_abs_error="
              << m128_error << '\n';
    require(m128_error < 1.0e-4F,
            "M128 persistent fused FFN numerical mismatch");
    cudaFree(m128_values);
    cudaFree(m128_scales);
    cudaFree(m128_output);
    cudaFree(dpersistent);
    cudaFuncAttributes persistent_attributes{};
    cudaFuncGetAttributes(
        &persistent_attributes,
        stream1_sm120_nvfp4_persistent::persistent_fused_ffn_kernel);
    int persistent_active_blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &persistent_active_blocks,
        stream1_sm120_nvfp4_persistent::persistent_fused_ffn_kernel,
        stream1_sm120_nvfp4_persistent::PersistentFfnM16Pipeline::kThreads,
        sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM16Storage));
    std::cout << "persistent_fused_ffn_resources registers="
              << persistent_attributes.numRegs
              << " dynamic_shared="
              << sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM16Storage)
              << " local_bytes=" << persistent_attributes.localSizeBytes
              << " active_blocks_per_sm=" << persistent_active_blocks << '\n';
    for (const std::uint32_t benchmark_rows :
         {8192U, 16384U, 32768U, 51072U}) {
        std::vector<float> benchmark_input(static_cast<std::size_t>(benchmark_rows) * D);
        for (std::uint32_t row = 0; row < benchmark_rows; ++row) {
            std::copy(input.begin(), input.begin() + D,
                      benchmark_input.begin() + static_cast<std::size_t>(row) * D);
        }
        const auto benchmark_packed = pack(benchmark_input, benchmark_rows, D, false);
        const auto benchmark_fragments =
            prepack_a_fragments(benchmark_packed, benchmark_rows, D);
        auto* benchmark_values = device_copy(benchmark_fragments);
        auto* benchmark_scales = device_copy(benchmark_packed.scales);
        float* benchmark_output = nullptr;
        cudaMalloc(&benchmark_output,
                   static_cast<std::size_t>(benchmark_rows) * D * sizeof(float));
        const dim3 grid((benchmark_rows +
            stream1_sm120_nvfp4_fused::kTotalRowsPerCta - 1U) /
            stream1_sm120_nvfp4_fused::kTotalRowsPerCta);
      for (const std::uint32_t threads : {256U}) {
        for (int warmup = 0; warmup < 5; ++warmup) {
            stream1_sm120_nvfp4_fused::fused_ffn_kernel<<<
                grid, threads, sizeof(stream1_sm120_nvfp4_fused::SharedStorage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output, nullptr, benchmark_rows);
        }
        require(cudaDeviceSynchronize() == cudaSuccess, "fused benchmark warmup failed");
        cudaEvent_t begin_event{}, end_event{};
        cudaEventCreate(&begin_event);
        cudaEventCreate(&end_event);
        constexpr int iterations = 100;
        cudaEventRecord(begin_event);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stream1_sm120_nvfp4_fused::fused_ffn_kernel<<<
                grid, threads, sizeof(stream1_sm120_nvfp4_fused::SharedStorage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output, nullptr, benchmark_rows);
        }
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        float total_ms = 0.0F;
        cudaEventElapsedTime(&total_ms, begin_event, end_event);
        const double per_iteration_ms = total_ms / iterations;
        const double flops = 4.0 * benchmark_rows * D * F;
        const double tflops = flops / (per_iteration_ms * 1.0e9);
        std::cout << "fused_ffn_benchmark rows=" << benchmark_rows
                  << " threads=" << threads
                  << " ms=" << per_iteration_ms
                  << " tflops=" << tflops << '\n';
        cudaEventDestroy(begin_event);
        cudaEventDestroy(end_event);
        cudaEventCreate(&begin_event);
        cudaEventCreate(&end_event);
        for (int warmup = 0; warmup < 5; ++warmup) {
            stream1_sm120_nvfp4_persistent::persistent_fused_ffn_kernel<<<
                grid, stream1_sm120_nvfp4_persistent::PersistentFfnM16Pipeline::kThreads,
                sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM16Storage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output, benchmark_rows);
        }
        require(cudaDeviceSynchronize() == cudaSuccess,
                "persistent fused benchmark warmup failed");
        cudaEventRecord(begin_event);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stream1_sm120_nvfp4_persistent::persistent_fused_ffn_kernel<<<
                grid, stream1_sm120_nvfp4_persistent::PersistentFfnM16Pipeline::kThreads,
                sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM16Storage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output, benchmark_rows);
        }
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        cudaEventElapsedTime(&total_ms, begin_event, end_event);
        const double persistent_ms = total_ms / iterations;
        std::cout << "persistent_fused_ffn_benchmark rows=" << benchmark_rows
                  << " threads="
                  << stream1_sm120_nvfp4_persistent::PersistentFfnM16Pipeline::kThreads
                  << " ms=" << persistent_ms
                  << " tflops=" << flops / (persistent_ms * 1.0e9) << '\n';
        cudaEventDestroy(begin_event);
        cudaEventDestroy(end_event);

        float* benchmark_output_peer = nullptr;
        cudaMalloc(&benchmark_output_peer,
                   static_cast<std::size_t>(benchmark_rows) * D * sizeof(float));
        cudaStream_t worker_streams[2]{};
        cudaStream_t timing_stream{};
        cudaEvent_t start_event{}, worker_done[2]{}, stop_event{};
        cudaStreamCreate(&worker_streams[0]);
        cudaStreamCreate(&worker_streams[1]);
        cudaStreamCreate(&timing_stream);
        cudaEventCreate(&start_event);
        cudaEventCreate(&worker_done[0]);
        cudaEventCreate(&worker_done[1]);
        cudaEventCreate(&stop_event);
        cudaEventRecord(start_event, timing_stream);
        for (int slot = 0; slot < 2; ++slot) {
            cudaStreamWaitEvent(worker_streams[slot], start_event);
        }
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stream1_sm120_nvfp4_persistent::persistent_fused_ffn_kernel<<<
                grid, stream1_sm120_nvfp4_persistent::PersistentFfnM16Pipeline::kThreads,
                sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM16Storage),
                worker_streams[0]>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output, benchmark_rows);
            stream1_sm120_nvfp4_persistent::persistent_fused_ffn_kernel<<<
                grid, stream1_sm120_nvfp4_persistent::PersistentFfnM16Pipeline::kThreads,
                sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM16Storage),
                worker_streams[1]>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output_peer, benchmark_rows);
        }
        cudaEventRecord(worker_done[0], worker_streams[0]);
        cudaEventRecord(worker_done[1], worker_streams[1]);
        cudaStreamWaitEvent(timing_stream, worker_done[0]);
        cudaStreamWaitEvent(timing_stream, worker_done[1]);
        cudaEventRecord(stop_event, timing_stream);
        cudaEventSynchronize(stop_event);
        cudaEventElapsedTime(&total_ms, start_event, stop_event);
        const double persistent_two_slot_ms = total_ms / iterations;
        std::cout << "persistent_fused_ffn_two_slot_benchmark rows="
                  << benchmark_rows << " slots=2 ms=" << persistent_two_slot_ms
                  << " aggregate_tflops="
                  << (2.0 * flops) / (persistent_two_slot_ms * 1.0e9) << '\n';
        cudaEventDestroy(start_event);
        cudaEventDestroy(worker_done[0]);
        cudaEventDestroy(worker_done[1]);
        cudaEventDestroy(stop_event);
        cudaStreamDestroy(worker_streams[0]);
        cudaStreamDestroy(worker_streams[1]);
        cudaStreamDestroy(timing_stream);
        cudaFree(benchmark_output_peer);

        const dim3 m128_grid((benchmark_rows +
            stream1_sm120_nvfp4_persistent::PersistentFfnM32Pipeline::kRows - 1U) /
            stream1_sm120_nvfp4_persistent::PersistentFfnM32Pipeline::kRows);
        cudaEventCreate(&begin_event);
        cudaEventCreate(&end_event);
        for (int warmup = 0; warmup < 5; ++warmup) {
            stream1_sm120_nvfp4_persistent::persistent_fused_ffn_m32_kernel<<<
                m128_grid,
                stream1_sm120_nvfp4_persistent::PersistentFfnM32Pipeline::kThreads,
                sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM32Storage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output, benchmark_rows);
        }
        require(cudaDeviceSynchronize() == cudaSuccess,
                "M128 persistent fused benchmark warmup failed");
        cudaEventRecord(begin_event);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stream1_sm120_nvfp4_persistent::persistent_fused_ffn_m32_kernel<<<
                m128_grid,
                stream1_sm120_nvfp4_persistent::PersistentFfnM32Pipeline::kThreads,
                sizeof(stream1_sm120_nvfp4_persistent::PersistentFfnM32Storage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, benchmark_output, benchmark_rows);
        }
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        cudaEventElapsedTime(&total_ms, begin_event, end_event);
        const double m128_ms = total_ms / iterations;
        std::cout << "persistent_fused_ffn_m128_benchmark rows="
                  << benchmark_rows << " threads="
                  << stream1_sm120_nvfp4_persistent::PersistentFfnM32Pipeline::kThreads
                  << " ms=" << m128_ms
                  << " tflops=" << flops / (m128_ms * 1.0e9) << '\n';
        cudaEventDestroy(begin_event);
        cudaEventDestroy(end_event);
        cudaEventCreate(&begin_event);
        cudaEventCreate(&end_event);
        cudaEventRecord(begin_event);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stream1_sm120_nvfp4_fused::fused_ffn_kernel<<<
                grid, threads, sizeof(stream1_sm120_nvfp4_fused::SharedStorage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, db2, nullptr, nullptr, benchmark_rows);
        }
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        cudaEventElapsedTime(&total_ms, begin_event, end_event);
        std::cout << "fused_ffn_stage_benchmark stage=ff1_pack rows=" << benchmark_rows
                  << " threads=" << threads
                  << " ms=" << total_ms / iterations << '\n';
        cudaEventDestroy(begin_event);
        cudaEventDestroy(end_event);
        cudaEventCreate(&begin_event);
        cudaEventCreate(&end_event);
        cudaEventRecord(begin_event);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stream1_sm120_nvfp4_fused::fused_ffn_kernel<<<
                grid, threads, sizeof(stream1_sm120_nvfp4_fused::SharedStorage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                nullptr, nullptr, nullptr, nullptr, nullptr, benchmark_rows);
        }
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        cudaEventElapsedTime(&total_ms, begin_event, end_event);
        std::cout << "fused_ffn_stage_benchmark stage=ff1_mma rows=" << benchmark_rows
                  << " threads=" << threads
                  << " ms=" << total_ms / iterations << '\n';
        cudaEventDestroy(begin_event);
        cudaEventDestroy(end_event);
        cudaEventCreate(&begin_event);
        cudaEventCreate(&end_event);
        cudaEventRecord(begin_event);
        for (int iteration = 0; iteration < iterations; ++iteration) {
            stream1_sm120_nvfp4_fused::fused_ffn_kernel<<<
                grid, threads, sizeof(stream1_sm120_nvfp4_fused::SharedStorage)>>>(
                benchmark_values, benchmark_scales, d1, ds1, db1,
                d2, ds2, nullptr, nullptr, nullptr, benchmark_rows);
        }
        cudaEventRecord(end_event);
        cudaEventSynchronize(end_event);
        cudaEventElapsedTime(&total_ms, begin_event, end_event);
        std::cout << "fused_ffn_stage_benchmark stage=ff1_quant rows=" << benchmark_rows
                  << " threads=" << threads
                  << " ms=" << total_ms / iterations << '\n';
        cudaEventDestroy(begin_event);
        cudaEventDestroy(end_event);
      }
        cudaFree(benchmark_values);
        cudaFree(benchmark_scales);
        cudaFree(benchmark_output);
    }
    return 0;
}
