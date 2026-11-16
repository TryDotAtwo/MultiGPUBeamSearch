// CNOT winner packing through the same ATen operations as the reference.
// The caller owns the tensors and the process-local CUDA stream.
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <algorithm>
#include <tuple>

namespace {
using at::indexing::Slice;

std::tuple<at::Tensor, at::Tensor, at::Tensor> winner_packet(
    const at::Tensor& ranks, const at::Tensor& local, const at::Tensor& new_lex,
    const at::Tensor& meta, const at::Tensor& children, const at::Tensor& closed,
    std::int64_t rank, std::int64_t world) {
    TORCH_CHECK(world >= 1 && world <= 65536 && rank >= 0 && rank < world,
                "invalid CNOT world geometry");
    TORCH_CHECK(children.is_cuda() && children.scalar_type() == at::kLong && children.dim() == 3,
                "CNOT children must be CUDA int64 matrices");
    TORCH_CHECK(closed.device() == children.device() && closed.scalar_type() == at::kLong
                && closed.dim() == 3 && closed.size(1) == children.size(1)
                && closed.size(2) == children.size(2), "CNOT closed matrices disagree with children");
    TORCH_CHECK(ranks.device() == children.device() && local.device() == children.device()
                && new_lex.device() == children.device() && meta.device() == children.device(),
                "CNOT winner metadata must be rank-local");
    TORCH_CHECK(ranks.scalar_type() == at::kLong && local.scalar_type() == at::kLong
                && new_lex.scalar_type() == at::kLong && meta.scalar_type() == at::kLong
                && ranks.dim() == 1 && local.dim() == 1 && new_lex.dim() == 1
                && ranks.size(0) == local.size(0) && ranks.size(0) == new_lex.size(0)
                && meta.dim() == 2 && meta.size(0) == children.size(0) && meta.size(1) == 4,
                "invalid CNOT winner metadata");
    const c10::cuda::CUDAGuard guard(children.device());
    const auto count = local.size(0);
    auto slots = at::nonzero(ranks.eq(rank)).flatten();
    auto kept = local.index({slots});
    auto owned_rows = children.index({kept});
    auto next_closed = at::cat({closed, owned_rows}, 0);
    auto records = at::stack({slots, new_lex.index({slots}), meta.index({kept, 0}),
                              meta.index({kept, 3}), ranks.index({slots})}, 1);
    auto wire = at::cat({records, owned_rows.reshape({kept.size(0), children.size(1) * children.size(2)})}, 1);
    auto targets = at::floor_divide(slots.mul(world), std::max<std::int64_t>(1, count));
    return {next_closed, wire, targets};
}

at::Tensor sort_received(const at::Tensor& received) {
    TORCH_CHECK(received.is_cuda() && received.scalar_type() == at::kLong && received.dim() == 2
                && received.size(1) >= 5, "invalid CNOT received records");
    const c10::cuda::CUDAGuard guard(received.device());
    return received.index({at::argsort(received.select(1, 0))});
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> frontier_views(
    const at::Tensor& received, std::int64_t n, std::int64_t words) {
    TORCH_CHECK(received.is_cuda() && received.scalar_type() == at::kLong && received.dim() == 2
                && n >= 1 && n <= 256 && words == (n + 63) / 64
                && received.size(1) == 5 + n * words, "invalid CNOT received frontier");
    const c10::cuda::CUDAGuard guard(received.device());
    return {received.index({Slice(), Slice(2, 5)}).contiguous(),
            received.index({Slice(), Slice(5, at::indexing::None)}).reshape({-1, n, words}).contiguous(),
            received.select(1, 1).contiguous()};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_cnot_domain, m) {
    m.def("winner_packet(Tensor ranks, Tensor local, Tensor new_lex, Tensor meta, Tensor children, Tensor closed, int rank, int world) -> (Tensor, Tensor, Tensor)");
    m.def("sort_received(Tensor received) -> Tensor");
    m.def("frontier_views(Tensor received, int n, int words) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("winner_packet", TORCH_FN(winner_packet));
    m.impl("sort_received", TORCH_FN(sort_received));
    m.impl("frontier_views", TORCH_FN(frontier_views));
}
