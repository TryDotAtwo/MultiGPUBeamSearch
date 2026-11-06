#include "cnot_domain.hpp"
#include "../src/cnot_domain.hpp"

#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <algorithm>
#include <limits>

namespace {

int check_states(const at::Tensor& states) {
    TORCH_CHECK(states.is_cuda() && states.scalar_type() == at::kLong && states.is_contiguous(),
                "CNOT packed states must be contiguous int64 CUDA tensors");
    TORCH_CHECK(states.dim() == 3 && states.size(1) >= 1
                && states.size(1) <= beam::cnot::max_qubits, "expected [batch,1..256,words]");
    const int n = static_cast<int>(states.size(1));
    TORCH_CHECK(states.size(2) == beam::cnot::words_per_row(n), "wrong CNOT row width");
    return n;
}

at::Tensor children(const at::Tensor& states, const at::Tensor& indices, const at::Tensor& moves) {
    const int n = check_states(states);
    TORCH_CHECK(indices.device() == states.device() && moves.device() == states.device(),
                "CNOT descriptors and parents must be on the same device");
    TORCH_CHECK(indices.is_contiguous() && moves.is_contiguous() && indices.dim() == 1
                && moves.dim() == 1 && indices.size(0) == moves.size(0), "invalid CNOT descriptor shape");
    TORCH_CHECK(indices.scalar_type() == at::kLong, "CNOT parent indices must be int64");
    const bool u16 = moves.scalar_type() == at::kUInt16;
    TORCH_CHECK(u16 || moves.scalar_type() == at::kLong, "CNOT moves must be uint16 or int64");
    TORCH_CHECK(moves.numel() == 0 || (n >= 2 && states.size(0) > 0),
                "cannot generate children from an empty frontier or an empty alphabet");
    const c10::cuda::CUDAGuard guard(states.device());
    auto output = at::empty({moves.size(0), n, states.size(2)}, states.options());
    C10_CUDA_CHECK(beam::cnot::launch_children(states.const_data_ptr<std::int64_t>(), states.size(0), n,
        indices.const_data_ptr<std::int64_t>(), moves.const_data_ptr(), u16, moves.size(0),
        output.mutable_data_ptr<std::int64_t>(), c10::cuda::getCurrentCUDAStream(states.get_device())));
    return output;
}

at::Tensor owners(const at::Tensor& states, std::int64_t world_size) {
    const int n = check_states(states);
    TORCH_CHECK(world_size >= 1 && world_size <= 65536, "CNOT world size must be in [1, 65536]");
    const c10::cuda::CUDAGuard guard(states.device());
    auto output = at::empty({states.size(0)}, states.options());
    C10_CUDA_CHECK(beam::cnot::launch_owners(states.const_data_ptr<std::int64_t>(), states.size(0), n,
        static_cast<int>(world_size), output.mutable_data_ptr<std::int64_t>(),
        c10::cuda::getCurrentCUDAStream(states.get_device())));
    return output;
}

at::Tensor first_unseen(const at::Tensor& closed, const at::Tensor& children) {
    const int n = check_states(closed);
    TORCH_CHECK(check_states(children) == n && closed.device() == children.device(),
                "closed and child CNOT matrices must have the same shape and device");
    const auto capacity = children.size(0);
    const c10::cuda::CUDAGuard guard(children.device());
    auto selected = at::empty({capacity}, children.options());
    if (!capacity) return selected;
    TORCH_CHECK(closed.size(0) <= std::numeric_limits<int>::max() - capacity,
                "CNOT sort exceeds CUB int32 ordinal capacity");
    auto ordinals = at::empty({closed.size(0) + capacity}, children.options().dtype(at::kInt));
    auto flags = at::empty({capacity}, children.options().dtype(at::kByte));
    auto count = at::empty({1}, children.options().dtype(at::kInt));
    const auto stream = c10::cuda::getCurrentCUDAStream(children.get_device());
    std::size_t bytes = 0;
    const auto run = [&](void* temp) {
        C10_CUDA_CHECK(beam::cnot::first_unseen(temp, bytes, closed.const_data_ptr<std::int64_t>(),
            static_cast<int>(closed.size(0)), children.const_data_ptr<std::int64_t>(),
            static_cast<int>(capacity), n, ordinals.mutable_data_ptr<int>(),
            flags.mutable_data_ptr<std::uint8_t>(), selected.mutable_data_ptr<std::int64_t>(),
            count.mutable_data_ptr<int>(), stream));
    };
    run(nullptr);
    auto temp = at::empty({static_cast<std::int64_t>(std::max<std::size_t>(1, bytes))},
                          children.options().dtype(at::kByte));
    run(temp.mutable_data_ptr());
    // The returned tensor has a data-dependent shape. Read just its count.
    return selected.narrow(0, 0, count.item<int>());
}

} // namespace

TORCH_LIBRARY(multigpubeamsearch_cnot_domain, m) {
    m.def("children(Tensor states, Tensor parent_indices, Tensor moves) -> Tensor");
    m.def("owners(Tensor states, int world_size) -> Tensor");
    m.def("first_unseen(Tensor closed, Tensor children) -> Tensor");
}
TORCH_LIBRARY_IMPL(multigpubeamsearch_cnot_domain, CUDA, m) {
    m.impl("children", TORCH_FN(children));
    m.impl("owners", TORCH_FN(owners));
    m.impl("first_unseen", TORCH_FN(first_unseen));
}
