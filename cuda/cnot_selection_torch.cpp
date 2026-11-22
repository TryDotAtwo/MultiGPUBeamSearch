// CNOT selection tensor stages, using the reference ATen operation sequence.
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <limits>
#include <tuple>

namespace {
using at::indexing::Slice;

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> selection_prepare(
    const at::Tensor& values, const at::Tensor& words, bool prune) {
    TORCH_CHECK(values.is_cuda() && values.dim() == 1 &&
                words.device() == values.device() && words.scalar_type() == at::kLong &&
                words.sizes() == values.sizes(), "invalid CNOT selection input");
    const c10::cuda::CUDAGuard guard(values.device());
    auto scores = values.to(at::kDouble) + 0.0;
    auto bits = scores.contiguous().view(at::kLong);
    auto selected = at::ones_like(words, words.options().dtype(at::kBool));
    auto boundary = at::empty({0}, words.options().dtype(at::kBool));
    auto keys = at::empty({0}, words.options());
    if (prune) {
        selected.zero_();
        boundary = at::ones_like(selected);
        keys = at::where(bits.lt(0), at::bitwise_not(bits),
                         at::bitwise_xor(bits, std::numeric_limits<std::int64_t>::min()));
    }
    return {bits, selected, boundary, keys};
}

at::Tensor selection_records(const at::Tensor& bits, const at::Tensor& words,
                             const at::Tensor& selected, std::int64_t rank) {
    TORCH_CHECK(bits.is_cuda() && bits.scalar_type() == at::kLong && bits.dim() == 1 &&
                words.device() == bits.device() && words.scalar_type() == at::kLong &&
                words.sizes() == bits.sizes() && selected.device() == bits.device() &&
                selected.scalar_type() == at::kBool && selected.sizes() == bits.sizes(),
                "invalid CNOT selection records input");
    const c10::cuda::CUDAGuard guard(bits.device());
    auto local = at::nonzero(selected).flatten();
    return at::stack({bits.index({local}), words.index({local}), at::full_like(local, rank), local}, 1);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> selection_order(const at::Tensor& records) {
    TORCH_CHECK(records.is_cuda() && records.scalar_type() == at::kLong &&
                records.dim() == 2 && records.size(1) == 4, "invalid CNOT selection records");
    const c10::cuda::CUDAGuard guard(records.device());
    auto by_word = at::argsort(records.index({Slice(), 1}), true, 0, false);
    auto scores = records.index({by_word, 0}).contiguous().view(at::kDouble);
    auto in_order = by_word.index({at::argsort(scores, true, 0, false)});
    auto lexical = at::empty_like(by_word);
    lexical.index_put_({by_word}, at::arange(by_word.numel(), by_word.options()));
    return {records.index({in_order, 2}), records.index({in_order, 3}), lexical.index({in_order})};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_cnot_domain, m) {
    m.def("selection_prepare(Tensor values, Tensor words, bool prune) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def("selection_records(Tensor bits, Tensor words, Tensor selected, int rank) -> Tensor");
    m.def("selection_order(Tensor records) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("selection_prepare", TORCH_FN(selection_prepare));
    m.impl("selection_records", TORCH_FN(selection_records));
    m.impl("selection_order", TORCH_FN(selection_order));
}
