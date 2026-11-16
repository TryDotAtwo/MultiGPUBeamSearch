// Batched expansion expressed with the same ATen operations as the reference.
#include "cnot_torch_internal.hpp"
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <ATen/ops/_assert_async.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <optional>
#include <tuple>

namespace {
using at::indexing::Slice;

std::tuple<at::Tensor, at::Tensor> expand_candidates(
    const at::Tensor& parents, const at::Tensor& lex,
    const std::optional<at::Tensor>& scores, std::int64_t begin,
    std::int64_t rank, std::int64_t world, std::int64_t k, bool proposal) {
    TORCH_CHECK(parents.is_cuda() && parents.scalar_type() == at::kLong && parents.dim() == 3
                && parents.size(1) >= 2 && parents.size(1) <= 256,
                "invalid CNOT parent matrices");
    const auto n = parents.size(1), words = parents.size(2), count = parents.size(0);
    const auto alphabet = n * (n - 1);
    TORCH_CHECK(words == (n + 63) / 64 && begin >= 0 && world >= 1 && world <= 65536
                && rank >= 0 && rank < world && k >= 1 && k <= alphabet,
                "invalid CNOT expansion geometry");
    TORCH_CHECK(lex.device() == parents.device() && lex.scalar_type() == at::kLong
                && lex.dim() == 1 && lex.size(0) == count, "invalid CNOT lexical ranks");
    const c10::cuda::CUDAGuard guard(parents.device());
    at::Tensor moves;
    if (count && k < alphabet) {
        TORCH_CHECK(scores.has_value() && scores->device() == parents.device()
                    && scores->is_floating_point() && scores->dim() == 2
                    && scores->size(0) == count && scores->size(1) == alphabet,
                    "invalid CNOT parent scores");
        moves = at::argsort(*scores, true, 1, proposal).index({Slice(), Slice(0, k)});
    } else {
        moves = at::arange(alphabet, parents.options()).slice(0, 0, k).expand({count, k});
    }
    auto indices = at::repeat_interleave(at::arange(count, parents.options()), k);
    moves = moves.reshape({-1});
    auto children = beam::cnot::torch_bridge::children(
        parents.contiguous(), indices.contiguous(), moves.contiguous());
    auto parent_slots = indices + begin;
    auto generation = parent_slots * alphabet + at::arange(k, parents.options()).repeat({count});
    at::_assert_async(at::logical_and(at::ge(moves, 0), at::le(moves, 65535)).all(),
                      "CNOT move must fit uint16");
    auto route = at::bitwise_or(moves, rank << 16);
    auto metadata = at::stack({parent_slots, lex.index({indices}) * alphabet + moves,
                               generation, route}, 1);
    auto wire = at::cat({metadata, children.reshape({-1, n * words})}, 1);
    return {wire, beam::cnot::torch_bridge::owners(children, world)};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_cnot_domain, m) {
    m.def("expand_candidates(Tensor parents, Tensor lex, Tensor? scores, int begin, int rank, int world, int k, bool proposal) -> (Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("expand_candidates", TORCH_FN(expand_candidates));
}
