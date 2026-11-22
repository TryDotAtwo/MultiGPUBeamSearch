#pragma once

#include <ATen/core/Tensor.h>
#include <cstdint>

namespace beam::cnot::torch_bridge {
// These wrappers retain the existing CUDA launch, device and stream checks.
at::Tensor children(const at::Tensor& states, const at::Tensor& indices, const at::Tensor& moves);
at::Tensor owners(const at::Tensor& states, std::int64_t world_size);
at::Tensor first_unseen(const at::Tensor& closed, const at::Tensor& children);
}
