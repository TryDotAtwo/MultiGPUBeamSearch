// Preserve the existing stable owner grouping for the NCCL CNOT wire records.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <tuple>

namespace {
std::tuple<at::Tensor, at::Tensor> exchange_packet(const at::Tensor& rows,
                                                    const at::Tensor& owners,
                                                    std::int64_t world_size) {
    TORCH_CHECK(rows.is_cuda() && rows.scalar_type() == at::kLong && rows.dim() == 2 &&
                owners.device() == rows.device() && owners.scalar_type() == at::kLong &&
                owners.dim() == 1 && owners.size(0) == rows.size(0) && world_size >= 2,
                "invalid CNOT exchange packet");
    const c10::cuda::CUDAGuard guard(rows.device());
    auto counts = at::bincount(owners, {}, world_size);
    auto order = at::argsort(owners, true, 0, false);
    auto send = at::index_select(rows, 0, order).contiguous();
    return {send, counts};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_cnot_domain, m) {
    m.def("exchange_packet(Tensor rows, Tensor owners, int world_size) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("exchange_packet", TORCH_FN(exchange_packet));
}
