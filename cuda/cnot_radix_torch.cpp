// Batched CNOT radix bookkeeping, preserving the original ATen operations.
#include <ATen/ATen.h>
#include <ATen/TensorIndexing.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include <tuple>

namespace {
std::tuple<at::Tensor, at::Tensor> radix_histogram(const at::Tensor& keys,
                                                 const at::Tensor& boundary,
                                                 std::int64_t shift) {
    TORCH_CHECK(keys.is_cuda() && keys.scalar_type() == at::kLong && keys.dim() == 1 &&
                boundary.device() == keys.device() && boundary.scalar_type() == at::kBool &&
                boundary.sizes() == keys.sizes() && shift >= 0 && shift <= 56,
                "invalid CNOT radix histogram input");
    const c10::cuda::CUDAGuard guard(keys.device());
    auto digit = at::bitwise_and(at::bitwise_right_shift(keys, shift), 255);
    auto histogram = at::bincount(digit.index({boundary}), {}, 256);
    return {digit, histogram};
}

std::tuple<std::int64_t, std::int64_t> radix_update(
    const at::Tensor& digit, const at::Tensor& histogram,
    at::Tensor selected, at::Tensor boundary, std::int64_t remaining) {
    TORCH_CHECK(digit.is_cuda() && digit.scalar_type() == at::kLong && digit.dim() == 1 &&
                histogram.device() == digit.device() && histogram.scalar_type() == at::kLong &&
                histogram.dim() == 1 && histogram.size(0) == 256 &&
                selected.device() == digit.device() && selected.scalar_type() == at::kBool &&
                selected.sizes() == digit.sizes() && boundary.device() == digit.device() &&
                boundary.scalar_type() == at::kBool && boundary.sizes() == digit.sizes() && remaining > 0,
                "invalid CNOT radix update input");
    const c10::cuda::CUDAGuard guard(digit.device());
    auto prefix = at::cumsum(histogram, 0);
    auto cutoff = at::searchsorted(prefix, at::Scalar(remaining)).item<std::int64_t>();
    TORCH_CHECK(cutoff >= 0 && cutoff < 256, "CNOT radix histogram is smaller than remaining width");
    auto below = cutoff ? prefix.index({cutoff - 1}).item<std::int64_t>() : 0;
    selected.bitwise_or_(at::bitwise_and(boundary, digit.lt(cutoff)));
    boundary.bitwise_and_(digit.eq(cutoff));
    auto equal = histogram.index({cutoff}).item<std::int64_t>();
    return {below, equal};
}
} // namespace

TORCH_LIBRARY_FRAGMENT(multigpubeamsearch_cnot_domain, m) {
    m.def("radix_histogram(Tensor keys, Tensor boundary, int shift) -> (Tensor, Tensor)");
    m.def("radix_update(Tensor digit, Tensor histogram, Tensor(a!) selected, Tensor(b!) boundary, int remaining) -> (int, int)");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("radix_histogram", TORCH_FN(radix_histogram));
    m.impl("radix_update", TORCH_FN(radix_update));
}
