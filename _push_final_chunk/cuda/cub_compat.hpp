#pragma once

#if __has_include(<cub/iterator/constant_input_iterator.cuh>)
#include <cub/iterator/constant_input_iterator.cuh>
namespace beam {
template <typename T>
using CubConstantInputIterator = cub::ConstantInputIterator<T>;
}
#else
#include <thrust/iterator/constant_iterator.h>
namespace beam {
template <typename T>
using CubConstantInputIterator = thrust::constant_iterator<T>;
}
#endif

namespace beam {
struct CubSum {
    template <typename T>
    __host__ __device__ constexpr T operator()(T a, T b) const {
        return a + b;
    }
};
} // namespace beam
