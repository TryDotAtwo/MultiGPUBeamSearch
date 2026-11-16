// The existing local CNOT goal evaluation expressed as batched ATen operations.
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <limits>
#include <tuple>

namespace {
std::tuple<std::int64_t, std::int64_t> local_goal(
    const at::Tensor& parents, const at::Tensor& lex, std::int64_t begin,
    const at::Tensor& identity) {
    TORCH_CHECK(parents.is_cuda() && parents.scalar_type() == at::kLong && parents.dim() == 3,
                "invalid CNOT goal parents");
    const auto count = parents.size(0), n = parents.size(1), words = parents.size(2);
    TORCH_CHECK(n >= 2 && n <= 256 && words == (n + 63) / 64 && begin >= 0 &&
                lex.device() == parents.device() && lex.scalar_type() == at::kLong &&
                lex.dim() == 1 && lex.size(0) == count && identity.device() == parents.device() &&
                identity.scalar_type() == at::kLong && identity.dim() == 3 &&
                identity.size(0) == 1 && identity.size(1) == n && identity.size(2) == words,
                "invalid CNOT goal geometry");
    if (!count) return {std::numeric_limits<std::int64_t>::max(), -1};
    const c10::cuda::CUDAGuard guard(parents.device());
    auto diff = at::bitwise_xor(parents, identity);
    auto changed = diff.ne(0).any(2);
    auto target = changed.to(at::kLong).argmax(1);
    auto index = at::arange(count, parents.options());
    auto needed = diff.index({index, target});
    auto matches = at::eq(parents, needed.unsqueeze(1)).all(2);
    matches.index_put_({index, target}, false);
    auto hits = at::logical_and(changed.sum(1).eq(1), matches.any(1)).nonzero().flatten();
    if (!hits.numel()) return {std::numeric_limits<std::int64_t>::max(), -1};
    auto hit = hits.index({lex.index({hits}).argmin()});
    auto control = matches.index({hit}).to(at::kLong).argmax();
    auto target_hit = target.index({hit});
    auto move = control * (n - 1) + target_hit - target_hit.gt(control).to(at::kLong);
    auto key = (lex.index({hit}) * (n * (n - 1)) + move).item<std::int64_t>();
    auto slot = begin + hit.item<std::int64_t>();
    return {key, slot};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_cnot_domain, m) {
    m.def("local_goal(Tensor parents, Tensor lex, int begin, Tensor identity) -> (int, int)");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("local_goal", TORCH_FN(local_goal));
}
