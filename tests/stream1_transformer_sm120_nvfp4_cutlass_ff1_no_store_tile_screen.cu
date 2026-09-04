#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <type_traits>
#include <vector>

namespace ff1_tile_screen {

using namespace cute;
using ElementAB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementScale = cutlass::float_ue4m3_t;
using ElementHidden = cutlass::float_e2m1_t;
using ElementBias = cutlass::half_t;
using ElementOutput = cutlass::half_t;
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using ClusterShape = Shape<_1, _1, _1>;
using ProblemShape = Shape<int, int, int, int>;
using ArchTag = cutlass::arch::Sm120;
using KernelSchedule =
    cutlass::gemm::KernelTmaWarpSpecializedPingpongNvf4Sm120;
using EpilogueSchedule =
    cutlass::epilogue::collective::EpilogueScheduleAuto;

template <int TileN, bool Transposed>
struct KernelTypes {
  static_assert(TileN == 128 || TileN == 256);
  // The transposed formulation computes [hidden, rows].  Column-major D makes
  // its physical byte order identical to the production row-major
  // [rows, hidden] buffer, so this comparison needs no relayout kernel.
  using LayoutD = std::conditional_t<
      Transposed, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
  using Tile = std::conditional_t<
      TileN == 128, Shape<_128, _128, _128>, Shape<_128, _256, _128>>;
  using PerRowFusion =
      cutlass::epilogue::fusion::LinCombPerRowBiasEltActBlockScaleFactor<
          cutlass::epilogue::thread::ReLU,
          16,
          ElementHidden,
          ElementCompute,
          ElementScale,
          LayoutD,
          ElementBias,
          ElementOutput>;
  using PerColFusion =
      cutlass::epilogue::fusion::LinCombPerColBiasEltActBlockScaleFactor<
          cutlass::epilogue::thread::ReLU,
          16,
          ElementHidden,
          ElementCompute,
          ElementScale,
          LayoutD,
          ElementBias,
          ElementOutput>;
  using Fusion = std::conditional_t<Transposed, PerRowFusion, PerColFusion>;
  using BuiltEpilogue =
      typename cutlass::epilogue::collective::CollectiveBuilder<
          ArchTag, cutlass::arch::OpClassTensorOp,
          Tile, ClusterShape,
          cutlass::epilogue::collective::EpilogueTileAuto,
          ElementAccumulator, ElementCompute,
          ElementOutput, LayoutD, 8,
          ElementHidden, LayoutD, 32,
          EpilogueSchedule, Fusion>::CollectiveOp;
  using Stages = cutlass::gemm::collective::StageCountAutoCarveout<
      static_cast<int>(sizeof(typename BuiltEpilogue::SharedStorage))>;
  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      ArchTag, cutlass::arch::OpClassBlockScaledTensorOp,
      ElementAB, LayoutA, 32,
      ElementAB, LayoutB, 32,
      ElementAccumulator, Tile, ClusterShape,
      Stages, KernelSchedule>::CollectiveOp;
  using Epilogue = BuiltEpilogue;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      ProblemShape, Mainloop, Epilogue, void>;
};

#ifndef STREAM1_FF1_SCREEN_TILE_N
#define STREAM1_FF1_SCREEN_TILE_N 128
#endif
#ifndef STREAM1_FF1_SCREEN_TRANSPOSED
#define STREAM1_FF1_SCREEN_TRANSPOSED 0
#endif
constexpr bool kTransposedScreen = STREAM1_FF1_SCREEN_TRANSPOSED != 0;
using Selected = KernelTypes<STREAM1_FF1_SCREEN_TILE_N, kTransposedScreen>;
using Kernel = typename Selected::Kernel;

__global__ __launch_bounds__(
    Kernel::MaxThreadsPerBlock, Kernel::MinBlocksPerMultiprocessor)
void run_ff1_no_store(CUTLASS_GRID_CONSTANT Kernel::Params const params) {
  extern __shared__ char storage[];
  Kernel{}(params, storage);
}

void require_cuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    std::cerr << what << ": " << cudaGetErrorString(status) << '\n';
    std::exit(2);
  }
}

void require_cutlass(cutlass::Status status, const char* what) {
  if (status != cutlass::Status::kSuccess) {
    std::cerr << what << ": " << cutlassGetStatusString(status) << '\n';
    std::exit(3);
  }
}

template <class T>
T* allocate_bytes(std::size_t bytes) {
  T* pointer = nullptr;
  require_cuda(cudaMalloc(&pointer, bytes), "cudaMalloc");
  return pointer;
}

void set_packed_nibble(
    std::vector<std::uint8_t>& bytes, std::size_t logical_offset,
    std::uint8_t raw) {
  auto& byte = bytes.at(logical_offset / 2U);
  const unsigned shift = static_cast<unsigned>(logical_offset & 1U) * 4U;
  byte = static_cast<std::uint8_t>(
      (byte & ~(0x0fU << shift)) | ((raw & 0x0fU) << shift));
}

std::uint8_t input_raw(int row, int column) {
  return static_cast<std::uint8_t>(1 + ((row * 5 + column * 3 + 1) % 7));
}

std::uint8_t weight_raw(int output, int column) {
  return static_cast<std::uint8_t>(1 + ((output * 3 + column * 5 + 2) % 7));
}

std::uint8_t input_scale_raw(int row, int group) {
  return ElementScale(0.5F + 0.25F * ((row + group * 3) % 5)).raw();
}

std::uint8_t weight_scale_raw(int output, int group) {
  return ElementScale(0.5F + 0.25F * ((output * 3 + group) % 5)).raw();
}

}  // namespace ff1_tile_screen

using namespace ff1_tile_screen;

int main(int argc, char** argv) {
  const int rows = argc > 1 ? std::atoi(argv[1]) : 51072;
  const int iterations = argc > 2 ? std::atoi(argv[2]) : 200;
  if (argc > 3 || rows < 128 || rows % 128 != 0 || iterations < 1) {
    std::cerr << "usage: tile_screen [rows_multiple_of_128] [iterations]\n";
    return 5;
  }
  constexpr int hidden = 1024;
  constexpr int k = 256;
  const int problem_m = kTransposedScreen ? hidden : rows;
  const int problem_n = kTransposedScreen ? rows : hidden;
  const auto problem_shape = make_shape(problem_m, problem_n, k, 1);
  const auto stride_a = cutlass::make_cute_packed_stride(
      typename Kernel::StrideA{}, {problem_m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(
      typename Kernel::StrideB{}, {problem_n, k, 1});
  const auto stride_c = cutlass::make_cute_packed_stride(
      typename Kernel::StrideC{}, {problem_m, problem_n, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(
      typename Kernel::StrideD{}, {problem_m, problem_n, 1});
  using ScaleConfig = typename Selected::Mainloop::Sm1xxBlkScaledConfig;
  // CUTLASS selects the scale-factor major from the output layout: row-major
  // uses K-major SFD, while column-major uses MN-major SFD.  Mirror the exact
  // callback specialization here so allocation and validation describe the
  // bytes the kernel actually writes.
  using OutputScaleConfig = std::conditional_t<
      kTransposedScreen,
      cutlass::detail::Sm1xxBlockScaledOutputConfig<16, cute::UMMA::Major::MN>,
      cutlass::detail::Sm1xxBlockScaledOutputConfig<16>>;
  const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
  const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);
  const auto layout_sfd =
      OutputScaleConfig::tile_atom_to_shape_SFD(problem_shape);
  const auto layout_a = make_layout(
      make_shape(problem_m, k, 1), stride_a);
  const auto layout_b = make_layout(
      make_shape(problem_n, k, 1), stride_b);
  const std::size_t a_bytes = static_cast<std::size_t>(problem_m) * k / 2;
  const std::size_t b_bytes = static_cast<std::size_t>(problem_n) * k / 2;
  const std::size_t c_elements = static_cast<std::size_t>(problem_m) * problem_n;
  const std::size_t sfa_bytes = cute::size(cute::filter_zeros(layout_sfa));
  const std::size_t sfb_bytes = cute::size(cute::filter_zeros(layout_sfb));
  const std::size_t sfd_bytes = cute::size(cute::filter_zeros(layout_sfd));

  auto* a = allocate_bytes<ElementHidden>(a_bytes);
  auto* b = allocate_bytes<ElementHidden>(b_bytes);
  // E2M1 stores two logical values per byte.
  const std::size_t d_bytes = (c_elements + 1U) / 2U;
  auto* d = allocate_bytes<std::uint8_t>(d_bytes);
  auto* sfa = allocate_bytes<ElementScale>(sfa_bytes);
  auto* sfb = allocate_bytes<ElementScale>(sfb_bytes);
  auto* sfd = allocate_bytes<ElementScale>(sfd_bytes);
  const std::size_t bias_elements = kTransposedScreen ? problem_m : problem_n;
  auto* bias = allocate_bytes<ElementBias>(bias_elements * sizeof(ElementBias));
  auto* norm = allocate_bytes<ElementCompute>(sizeof(ElementCompute));
  const std::uint8_t one_scale = ElementScale(1.0F).raw();
  std::vector<std::uint8_t> host_a(a_bytes, 0);
  std::vector<std::uint8_t> host_b(b_bytes, 0);
  std::vector<std::uint8_t> host_sfa(sfa_bytes, one_scale);
  std::vector<std::uint8_t> host_sfb(sfb_bytes, one_scale);
  std::vector<ElementBias> host_bias(bias_elements);
  if constexpr (kTransposedScreen) {
    for (int output = 0; output < hidden; ++output) {
      host_bias[output] = ElementBias(0.03125F * ((output % 9) - 4));
      for (int column = 0; column < k; ++column) {
        set_packed_nibble(
            host_a, static_cast<std::size_t>(layout_a(output, column, 0)),
            weight_raw(output, column));
      }
      for (int group = 0; group < k / 16; ++group) {
        host_sfa[static_cast<std::size_t>(layout_sfa(output, group * 16, 0))] =
            weight_scale_raw(output, group);
      }
    }
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < k; ++column) {
        set_packed_nibble(
            host_b, static_cast<std::size_t>(layout_b(row, column, 0)),
            input_raw(row, column));
      }
      for (int group = 0; group < k / 16; ++group) {
        host_sfb[static_cast<std::size_t>(layout_sfb(row, group * 16, 0))] =
            input_scale_raw(row, group);
      }
    }
  } else {
    for (int row = 0; row < rows; ++row) {
      for (int column = 0; column < k; ++column) {
        set_packed_nibble(
            host_a, static_cast<std::size_t>(layout_a(row, column, 0)),
            input_raw(row, column));
      }
      for (int group = 0; group < k / 16; ++group) {
        host_sfa[static_cast<std::size_t>(layout_sfa(row, group * 16, 0))] =
            input_scale_raw(row, group);
      }
    }
    for (int output = 0; output < hidden; ++output) {
      host_bias[output] = ElementBias(0.03125F * ((output % 9) - 4));
      for (int column = 0; column < k; ++column) {
        set_packed_nibble(
            host_b, static_cast<std::size_t>(layout_b(output, column, 0)),
            weight_raw(output, column));
      }
      for (int group = 0; group < k / 16; ++group) {
        host_sfb[static_cast<std::size_t>(layout_sfb(output, group * 16, 0))] =
            weight_scale_raw(output, group);
      }
    }
  }
  require_cuda(cudaMemcpy(
      a, host_a.data(), a_bytes, cudaMemcpyHostToDevice), "copy A");
  require_cuda(cudaMemcpy(
      b, host_b.data(), b_bytes, cudaMemcpyHostToDevice), "copy B");
  require_cuda(cudaMemcpy(
      sfa, host_sfa.data(), sfa_bytes, cudaMemcpyHostToDevice), "copy SFA");
  require_cuda(cudaMemcpy(
      sfb, host_sfb.data(), sfb_bytes, cudaMemcpyHostToDevice), "copy SFB");
  require_cuda(cudaMemcpy(
      bias, host_bias.data(), bias_elements * sizeof(ElementBias),
      cudaMemcpyHostToDevice), "copy bias");
  require_cuda(cudaMemset(d, 0, d_bytes), "memset D");
  require_cuda(cudaMemset(sfd, 0, sfd_bytes), "memset SFD");
  const ElementCompute norm_value = 6.0F;
  require_cuda(cudaMemcpy(
      norm, &norm_value, sizeof(norm_value), cudaMemcpyHostToDevice),
      "copy norm");

  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {a, stride_a, b, stride_b, sfa, layout_sfa, sfb, layout_sfb},
      {{1.0F, 0.0F}, nullptr, stride_c,
       reinterpret_cast<ElementHidden*>(d), stride_d}};
  arguments.epilogue.thread.block_scale_factor_ptr = sfd;
  arguments.epilogue.thread.norm_constant_ptr = norm;
  arguments.epilogue.thread.bias_ptr = bias;
  if (!Kernel::can_implement(arguments)) {
    std::cerr << "can_implement=false\n";
    return 4;
  }
  const std::size_t workspace_bytes = Kernel::get_workspace_size(arguments);
  auto* workspace = allocate_bytes<std::uint8_t>(workspace_bytes == 0 ? 1 : workspace_bytes);
  require_cutlass(
      Kernel::initialize_workspace(arguments, workspace, nullptr),
      "initialize workspace");
  auto params = Kernel::to_underlying_arguments(arguments, workspace);
  const auto grid = Kernel::get_grid_shape(params);
  constexpr int shared_bytes = sizeof(typename Kernel::SharedStorage);
  require_cuda(cudaFuncSetAttribute(
      run_ff1_no_store,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_bytes), "set dynamic shared memory");
  auto launch = [&] {
    run_ff1_no_store<<<grid, Kernel::MaxThreadsPerBlock, shared_bytes>>>(params);
    require_cuda(cudaGetLastError(), "kernel launch");
  };
  for (int i = 0; i < 10; ++i) launch();
  require_cuda(cudaDeviceSynchronize(), "warmup sync");
  cudaEvent_t begin{}, end{};
  require_cuda(cudaEventCreate(&begin), "create begin");
  require_cuda(cudaEventCreate(&end), "create end");
  require_cuda(cudaEventRecord(begin), "record begin");
  for (int i = 0; i < iterations; ++i) launch();
  require_cuda(cudaEventRecord(end), "record end");
  require_cuda(cudaEventSynchronize(end), "sync end");
  float elapsed_ms = 0.0F;
  require_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end), "elapsed");
  std::vector<std::uint8_t> host_sfd(sfd_bytes);
  std::vector<std::uint8_t> host_d(d_bytes);
  require_cuda(cudaMemcpy(
      host_d.data(), d, d_bytes, cudaMemcpyDeviceToHost), "copy D");
  require_cuda(cudaMemcpy(
      host_sfd.data(), sfd, sfd_bytes, cudaMemcpyDeviceToHost), "copy SFD");
  const auto sfd_nonzero = std::count_if(
      host_sfd.begin(), host_sfd.end(), [](auto value) { return value != 0; });
  auto fnv1a = [](const std::uint8_t* data, std::size_t count) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (std::size_t i = 0; i < count; ++i) {
      hash ^= data[i];
      hash *= 1099511628211ULL;
    }
    return hash;
  };
  const auto d_hash = fnv1a(host_d.data(), host_d.size());
  std::vector<std::uint8_t> logical_sfd;
  logical_sfd.reserve(static_cast<std::size_t>(rows) * hidden / 16U);
  for (int row = 0; row < rows; ++row) {
    for (int column = 0; column < hidden; column += 16) {
      const auto offset = kTransposedScreen
          ? static_cast<std::size_t>(layout_sfd(column, row, 0))
          : static_cast<std::size_t>(layout_sfd(row, column, 0));
      logical_sfd.push_back(host_sfd[offset]);
    }
  }
  const auto logical_sfd_hash = fnv1a(logical_sfd.data(), logical_sfd.size());
  const double us = elapsed_ms * 1000.0 / iterations;
  const double flops = 2.0 * rows * hidden * k;
  std::cout << "ff1_no_store_tile_screen"
            << " orientation=" << (kTransposedScreen ? "weight_m" : "rows_m")
            << " tile=128x" << STREAM1_FF1_SCREEN_TILE_N << "x128"
            << " rows=" << rows
            << " mma_threads="
            << cute::thr_size(typename Selected::Mainloop::TiledMma{})
            << " stages=" << Selected::Mainloop::DispatchPolicy::Stages
            << " shared_bytes=" << shared_bytes
            << " us=" << us
            << " useful_tflops=" << flops / (us * 1.0e6)
            << " sfd_nonzero=" << sfd_nonzero
            << " d_hash=" << d_hash
            << " logical_sfd_hash=" << logical_sfd_hash
            << " workspace_bytes=" << workspace_bytes << '\n';
  return 0;
}
