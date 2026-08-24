#include "stream1_transformer_cublaslt_fp16.hpp"

#include <cublasLt.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace {

constexpr std::size_t kWorkspaceBytes = 64U * 1024U * 1024U;

void check(cublasStatus_t status, const char* what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(what);
    }
}

struct Plan {
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t input = nullptr;
    cublasLtMatrixLayout_t weight = nullptr;
    cublasLtMatrixLayout_t output = nullptr;
    cublasLtMatmulAlgo_t algorithm{};

    Plan() = default;
    Plan(const Plan&) = delete;
    Plan& operator=(const Plan&) = delete;
    Plan(Plan&& other) noexcept
        : operation(other.operation), input(other.input), weight(other.weight),
          output(other.output), algorithm(other.algorithm) {
        other.operation = nullptr;
        other.input = nullptr;
        other.weight = nullptr;
        other.output = nullptr;
    }

    ~Plan() {
        if (output) cublasLtMatrixLayoutDestroy(output);
        if (weight) cublasLtMatrixLayoutDestroy(weight);
        if (input) cublasLtMatrixLayoutDestroy(input);
        if (operation) cublasLtMatmulDescDestroy(operation);
    }
};

struct Cache {
    cublasLtHandle_t handle = nullptr;
    std::mutex mutex;
    std::unordered_map<std::uint64_t, Plan> plans;

    Cache() { check(cublasLtCreate(&handle), "cublasLtCreate failed"); }
    ~Cache() { cublasLtDestroy(handle); }
};

Cache& cache() {
    static Cache value;
    return value;
}

std::uint64_t key(std::uint32_t m, std::uint32_t k, std::uint32_t n) {
    return (static_cast<std::uint64_t>(m) << 32U) ^
        (static_cast<std::uint64_t>(k) << 16U) ^ n;
}

Plan make_plan(Cache& owner, std::uint32_t m, std::uint32_t k, std::uint32_t n) {
    Plan plan;
    check(cublasLtMatmulDescCreate(&plan.operation, CUBLAS_COMPUTE_32F, CUDA_R_32F),
          "cublasLtMatmulDescCreate failed");
    check(cublasLtMatrixLayoutCreate(&plan.input, CUDA_R_16F, m, k, k),
          "cublasLt input layout failed");
    check(cublasLtMatrixLayoutCreate(&plan.weight, CUDA_R_16F, k, n, n),
          "cublasLt weight layout failed");
    check(cublasLtMatrixLayoutCreate(&plan.output, CUDA_R_16F, m, n, n),
          "cublasLt output layout failed");
    const cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    check(cublasLtMatrixLayoutSetAttribute(plan.input, CUBLASLT_MATRIX_LAYOUT_ORDER,
          &row_major, sizeof(row_major)), "cublasLt input order failed");
    check(cublasLtMatrixLayoutSetAttribute(plan.weight, CUBLASLT_MATRIX_LAYOUT_ORDER,
          &row_major, sizeof(row_major)), "cublasLt weight order failed");
    check(cublasLtMatrixLayoutSetAttribute(plan.output, CUBLASLT_MATRIX_LAYOUT_ORDER,
          &row_major, sizeof(row_major)), "cublasLt output order failed");

    cublasLtMatmulPreference_t preference = nullptr;
    check(cublasLtMatmulPreferenceCreate(&preference), "cublasLt preference create failed");
    check(cublasLtMatmulPreferenceSetAttribute(preference,
          CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceBytes, sizeof(kWorkspaceBytes)),
          "cublasLt workspace preference failed");
    cublasLtMatmulHeuristicResult_t result{};
    int returned = 0;
    const auto status = cublasLtMatmulAlgoGetHeuristic(
        owner.handle, plan.operation, plan.input, plan.weight, plan.output, plan.output,
        preference, 1, &result, &returned);
    cublasLtMatmulPreferenceDestroy(preference);
    check(status, "cublasLt heuristic query failed");
    if (returned != 1 || result.state != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cublasLt found no FP16 algorithm for transformer shape");
    }
    plan.algorithm = result.algo;
    return plan;
}

}  // namespace

std::size_t stream1_transformer_cublaslt_fp16_workspace_bytes() { return kWorkspaceBytes; }

void stream1_transformer_cublaslt_fp16_linear_cuda(
    const half* input, const half* weight, half* output,
    std::uint32_t rows, std::uint32_t input_cols, std::uint32_t output_cols,
    void* workspace, std::size_t workspace_bytes, cudaStream_t stream) {
    if (!input || !weight || !output || rows == 0U || input_cols == 0U || output_cols == 0U) {
        throw std::invalid_argument("cublasLt FP16 GEMM received invalid arguments");
    }
    if (!workspace || workspace_bytes < kWorkspaceBytes) {
        throw std::invalid_argument("cublasLt FP16 GEMM workspace is smaller than 64 MiB");
    }
    auto& owner = cache();
    std::lock_guard<std::mutex> lock(owner.mutex);
    auto it = owner.plans.find(key(rows, input_cols, output_cols));
    if (it == owner.plans.end()) {
        it = owner.plans.emplace(key(rows, input_cols, output_cols),
                                 make_plan(owner, rows, input_cols, output_cols)).first;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    check(cublasLtMatmul(owner.handle, it->second.operation, &alpha,
          input, it->second.input, weight, it->second.weight, &beta,
          output, it->second.output, output, it->second.output,
          &it->second.algorithm, workspace, workspace_bytes, stream),
          "cublasLt FP16 matmul failed");
}
