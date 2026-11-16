// Preserve the existing generation order and full-matrix CNOT deduplication.
#include "cnot_torch_internal.hpp"
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <tuple>

namespace {
using at::indexing::Slice;

std::tuple<at::Tensor, at::Tensor> distinct_candidates(
    const at::Tensor& incoming, const at::Tensor& closed, std::int64_t n, std::int64_t words) {
    TORCH_CHECK(incoming.is_cuda() && incoming.scalar_type() == at::kLong && incoming.dim() == 2,
                "CNOT incoming records must be CUDA int64 rows");
    TORCH_CHECK(n >= 2 && n <= 256 && words == (n + 63) / 64 &&
                incoming.size(1) == 4 + n * words, "invalid CNOT incoming record geometry");
    TORCH_CHECK(closed.device() == incoming.device(), "CNOT closed set is on a different device");
    const c10::cuda::CUDAGuard guard(incoming.device());
    auto order = at::argsort(incoming.index({Slice(), 2}), true, 0, false);
    auto sorted = incoming.index({order});
    auto children = sorted.index({Slice(), Slice(4, at::indexing::None)})
                            .reshape({-1, n, words}).contiguous();
    auto first = beam::cnot::torch_bridge::first_unseen(closed.contiguous(), children);
    return {children.index({first}), sorted.index({first, Slice(0, 4)})};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_cnot_domain, m) {
    m.def("distinct_candidates(Tensor incoming, Tensor closed, int n, int words) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("distinct_candidates", TORCH_FN(distinct_candidates));
}
