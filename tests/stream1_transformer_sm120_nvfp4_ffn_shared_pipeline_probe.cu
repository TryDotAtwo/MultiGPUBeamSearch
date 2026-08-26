#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int kRows = 64;
constexpr int kFfDim = 1024;
constexpr int kScaleVector = 16;
constexpr int kProducerTiles = 8;
constexpr int kThreads = 9 * 32;
constexpr int kValueBytes = kRows * kFfDim / 2;
constexpr int kScaleBytes = kRows * (kFfDim / kScaleVector);
constexpr int kSharedBytes = kValueBytes + kScaleBytes;

[[noreturn]] void fail(const char* what, cudaError_t error) {
    std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
    std::exit(1);
}

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        fail(what, error);
    }
}

// Structural probe for the intended fused FFN ownership model.  Eight producer
// tiles cooperatively populate one CTA-local packed NVFP4 hidden tensor.  The
// same CTA then consumes it in two logical FF2 output groups.  No hidden byte
// is written to global memory.
__global__ void shared_pipeline_probe(std::uint32_t* output, int iterations) {
    extern __shared__ std::uint8_t storage[];
    auto* values = storage;
    auto* scales = storage + kValueBytes;

    std::uint32_t checksum = 0;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        for (int producer_tile = 0; producer_tile < kProducerTiles; ++producer_tile) {
            const int value_begin = producer_tile * (kValueBytes / kProducerTiles);
            const int value_end = value_begin + (kValueBytes / kProducerTiles);
            for (int index = value_begin + threadIdx.x; index < value_end;
                 index += blockDim.x) {
                values[index] = static_cast<std::uint8_t>(
                    (index + producer_tile + iteration + blockIdx.x) & 0xff);
            }

            const int scale_begin = producer_tile * (kScaleBytes / kProducerTiles);
            const int scale_end = scale_begin + (kScaleBytes / kProducerTiles);
            for (int index = scale_begin + threadIdx.x; index < scale_end;
                 index += blockDim.x) {
                scales[index] = static_cast<std::uint8_t>(
                    (3 * index + producer_tile + iteration + blockIdx.x) & 0xff);
            }
        }
        __syncthreads();

        // Two sequential N=128 consumer groups mirror the fixed FF2 topology.
        for (int consumer_group = 0; consumer_group < 2; ++consumer_group) {
            std::uint32_t local = 0;
            for (int index = threadIdx.x + consumer_group; index < kValueBytes;
                 index += blockDim.x * 2) {
                local += values[index];
            }
            for (int index = threadIdx.x + consumer_group; index < kScaleBytes;
                 index += blockDim.x * 2) {
                local += scales[index];
            }
            checksum += local;
        }
        __syncthreads();
    }

    atomicAdd(output + blockIdx.x, checksum);
}

}  // namespace

int main(int argc, char** argv) {
    const int blocks = argc > 1 ? std::max(1, std::atoi(argv[1])) : 1024;
    const int iterations = argc > 2 ? std::max(1, std::atoi(argv[2])) : 100;

    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "cudaGetDeviceProperties");
    check(cudaFuncSetAttribute(
              shared_pipeline_probe,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              kSharedBytes),
          "cudaFuncSetAttribute");

    int active_blocks_per_sm = 0;
    check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &active_blocks_per_sm,
              shared_pipeline_probe,
              kThreads,
              kSharedBytes),
          "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

    std::uint32_t* output = nullptr;
    check(cudaMalloc(&output, static_cast<std::size_t>(blocks) * sizeof(*output)),
          "cudaMalloc");
    check(cudaMemset(output, 0, static_cast<std::size_t>(blocks) * sizeof(*output)),
          "cudaMemset");

    cudaEvent_t start{};
    cudaEvent_t stop{};
    check(cudaEventCreate(&start), "cudaEventCreate(start)");
    check(cudaEventCreate(&stop), "cudaEventCreate(stop)");
    check(cudaEventRecord(start), "cudaEventRecord(start)");
    shared_pipeline_probe<<<blocks, kThreads, kSharedBytes>>>(output, iterations);
    check(cudaGetLastError(), "shared_pipeline_probe launch");
    check(cudaEventRecord(stop), "cudaEventRecord(stop)");
    check(cudaEventSynchronize(stop), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0F;
    check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
    std::vector<std::uint32_t> host_output(static_cast<std::size_t>(blocks));
    check(cudaMemcpy(host_output.data(), output,
                     host_output.size() * sizeof(host_output.front()),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy");

    const bool nonzero = std::all_of(host_output.begin(), host_output.end(),
                                     [](std::uint32_t value) { return value != 0U; });
    const double bytes = static_cast<double>(blocks) * iterations * kSharedBytes * 2.0;
    const double effective_gib_s = bytes / (elapsed_ms * 1.0e-3) / (1024.0 * 1024.0 * 1024.0);
    std::printf(
        "sm120_nvfp4_ffn_shared_probe gpu=%s cc=%d.%d sm_count=%d shared_bytes=%d "
        "threads=%d active_blocks_per_sm=%d blocks=%d iterations=%d elapsed_ms=%.6f "
        "effective_gib_s=%.3f nonzero=%d\n",
        properties.name,
        properties.major,
        properties.minor,
        properties.multiProcessorCount,
        kSharedBytes,
        kThreads,
        active_blocks_per_sm,
        blocks,
        iterations,
        elapsed_ms,
        effective_gib_s,
        nonzero ? 1 : 0);

    cudaEventDestroy(stop);
    cudaEventDestroy(start);
    cudaFree(output);
    return nonzero ? 0 : 2;
}
