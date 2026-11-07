#include "stream1_transformer_sm120_nvfp4_persistent_pipeline.cuh"

#include <cstdlib>
#include <iostream>

int main() {
    using Config =
        stream1_sm120_nvfp4_persistent::PersistentFfnM32Pipeline;
    using Storage =
        stream1_sm120_nvfp4_persistent::PersistentFfnM32Storage;
    static_assert(Config::kRows == 32U);
    static_assert(Config::kHiddenSlices == 8U);
    static_assert(Config::kRingBuffers == 2U);
    static_assert(Config::kProducerWarps == 8U);
    static_assert(Config::kConsumerWarps == 8U);
    static_assert(Config::kThreads == 512U);
    static_assert(Config::kInputStageBytes == 4608U);
    static_assert(Config::kWeightStageBytes == 18432U);
    static_assert(Config::kRingBytes == 8704U);
    if (sizeof(Storage) > 32U * 1024U) {
        return EXIT_FAILURE;
    }
    std::cout
        << "stream1_transformer_sm120_nvfp4_m32_pipeline_contract_tests=pass"
        << " threads=" << Config::kThreads
        << " shared_bytes=" << sizeof(Storage)
        << " input_stage_bytes=" << Config::kInputStageBytes
        << " weight_stage_bytes=" << Config::kWeightStageBytes
        << " ring_bytes=" << Config::kRingBytes << '\n';
    return EXIT_SUCCESS;
}
