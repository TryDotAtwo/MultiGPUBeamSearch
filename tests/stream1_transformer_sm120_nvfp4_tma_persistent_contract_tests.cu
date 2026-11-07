#include "stream1_transformer_sm120_nvfp4_tma_persistent.cuh"

#include <cstdlib>
#include <iostream>
#include <type_traits>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(EXIT_FAILURE);
    }
}

}  // namespace

int main() {
    using Plan = stream1_sm120_nvfp4_tma_persistent::Plan;
    using Storage = stream1_sm120_nvfp4_tma_persistent::SharedStorage;

    static_assert(Plan::kRowsPerCta == 128U);
    static_assert(Plan::kHiddenSliceColumns == 128U);
    static_assert(Plan::kHiddenSlices == 8U);
    static_assert(Plan::kRingBuffers == 2U);
    static_assert(Plan::kGlobalHiddenBytes == 0U);
    static_assert(Plan::kHandoffBytes == 18432U);
    static_assert(alignof(Storage) >= 128U);
    static_assert(std::is_trivially_destructible_v<Storage>);

    require(
        Plan::kSharedStorageBytes <= Plan::kSm120SharedCapacityBytes,
        "M128 TMA persistent FFN exceeds SM120 shared-memory capacity");
    require(
        Plan::kFf1MainloopStages >= 2U && Plan::kFf2MainloopStages >= 2U,
        "both TMA mainloops need at least double buffering");
    require(
        sizeof(Storage) == Plan::kSharedStorageBytes,
        "reported M128 persistent shared-storage size mismatch");

    std::cout
        << "stream1_transformer_sm120_nvfp4_tma_persistent_contract_tests=pass"
        << " shared_bytes=" << sizeof(Storage)
        << " handoff_bytes=" << Plan::kHandoffBytes
        << " ff1_stages=" << Plan::kFf1MainloopStages
        << " ff2_stages=" << Plan::kFf2MainloopStages
        << '\n';
    return EXIT_SUCCESS;
}
