#include "stream1_transformer_sm120_nvfp4_cutlass_ff1_shared_handoff.hpp"

#include <cutlass/util/packed_stride.hpp>

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

using namespace stream1_sm120_nvfp4_cutlass_dsm;

#ifndef STREAM1_HANDOFF_CUSTOM
#define STREAM1_HANDOFF_CUSTOM 0
#endif

#ifndef STREAM1_HANDOFF_WARMUP_ITERATIONS
#define STREAM1_HANDOFF_WARMUP_ITERATIONS 5
#endif

#ifndef STREAM1_HANDOFF_MEASURE_ITERATIONS
#define STREAM1_HANDOFF_MEASURE_ITERATIONS 200
#endif

static_assert(STREAM1_HANDOFF_CUSTOM >= 0 && STREAM1_HANDOFF_CUSTOM <= 2);
using SelectedKernel = std::conditional_t<
    STREAM1_HANDOFF_CUSTOM == 2,
    Ff1SharedOnlyKernel,
    std::conditional_t<
        STREAM1_HANDOFF_CUSTOM == 1, Ff1SharedHandoffKernel, Ff1Kernel>>;

#if STREAM1_HANDOFF_CUSTOM == 2
__global__ void instantiate_ff1_shared_only_operator(
    CUTLASS_GRID_CONSTANT typename SelectedKernel::Params const params);
#define STREAM1_SELECTED_KERNEL_ENTRY instantiate_ff1_shared_only_operator
#else
__global__ void run_selected_kernel(
    CUTLASS_GRID_CONSTANT typename SelectedKernel::Params const params);
#define STREAM1_SELECTED_KERNEL_ENTRY run_selected_kernel
#endif

namespace {

using ScaleConfig = typename Ff1Mainloop::Sm1xxBlkScaledConfig;
using OutputScaleConfig =
    cutlass::detail::Sm1xxBlockScaledOutputConfig<Contract::kScaleVector>;

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

std::uint64_t fnv1a(std::vector<std::uint8_t> const& bytes) {
  std::uint64_t hash = 1469598103934665603ULL;
  for (auto byte : bytes) {
    hash ^= byte;
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

int main(int argc, char** argv) {
#if defined(STREAM1_HANDOFF_VALUE_READBACK)
  require_cuda(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 16U * 1024U * 1024U),
               "set diagnostic printf FIFO");
#endif
#if defined(STREAM1_HANDOFF_FIXED_AFFINE)
  // Exercise the host-side fail-closed contract on the target as well.
  for (int invalid = 0; invalid < 4; ++invalid) {
    Ff1SharedFusionCallbacks::Arguments a;
    ElementCompute dummy = 1;
    if (invalid == 0) a.alpha = 2;
    if (invalid == 1) a.beta = 1;
    if (invalid == 2) a.alpha_ptr = &dummy;
    if (invalid == 3) a.beta_ptr = &dummy;
    bool rejected = false;
    try { auto converted = static_cast<Ff1SharedFusionCallbacks::Impl::Arguments>(a); (void)converted; }
    catch (std::invalid_argument const&) { rejected = true; }
    if (!rejected) { std::cerr << "fixed_affine_guard_failed\n"; return 9; }
  }
#endif
  const int m = argc > 1 ? std::atoi(argv[1]) : 128;
  const int iterations = argc > 2
      ? std::atoi(argv[2]) : STREAM1_HANDOFF_MEASURE_ITERATIONS;
  const int slots = argc > 3 ? std::atoi(argv[3]) : 1;
  constexpr int n = 1024;
  constexpr int k = 256;
  if (argc > 4 || m < 128 || m % 128 != 0 || iterations < 1 ||
      iterations > 10000 || (slots != 1 && slots != 2 && slots != 4)) {
    std::cerr <<
        "usage: benchmark [rows_multiple_of_128] [iterations] [slots=1|2|4]\n";
    return 5;
  }
  const auto problem_shape = make_shape(m, n, k, 1);
  const auto stride_a = cutlass::make_cute_packed_stride(
      typename SelectedKernel::StrideA{}, {m, k, 1});
  const auto stride_b = cutlass::make_cute_packed_stride(
      typename SelectedKernel::StrideB{}, {n, k, 1});
  const auto stride_c = cutlass::make_cute_packed_stride(
      typename SelectedKernel::StrideC{}, {m, n, 1});
  const auto stride_d = cutlass::make_cute_packed_stride(
      typename SelectedKernel::StrideD{}, {m, n, 1});
  const auto layout_sfa = ScaleConfig::tile_atom_to_shape_SFA(problem_shape);
  const auto layout_sfb = ScaleConfig::tile_atom_to_shape_SFB(problem_shape);
  const auto layout_sfd = OutputScaleConfig::tile_atom_to_shape_SFD(problem_shape);
  const std::size_t a_elements = static_cast<std::size_t>(m) * k;
  const std::size_t b_elements = static_cast<std::size_t>(n) * k;
  const std::size_t output_elements = static_cast<std::size_t>(m) * n;
  const std::size_t sfa_elements = cute::size(cute::filter_zeros(layout_sfa));
  const std::size_t sfb_elements = cute::size(cute::filter_zeros(layout_sfb));
  const std::size_t sfd_elements = cute::size(cute::filter_zeros(layout_sfd));

  auto* a = allocate_bytes<ElementHidden>((a_elements + 1) / 2);
  auto* b = allocate_bytes<ElementHidden>((b_elements + 1) / 2);
  auto* c = allocate_bytes<ElementOutput>(output_elements * sizeof(ElementOutput));
  auto* d = allocate_bytes<ElementHidden>((output_elements + 1) / 2);
  auto* sfa = allocate_bytes<ElementScale>(sfa_elements);
  auto* sfb = allocate_bytes<ElementScale>(sfb_elements);
  auto* sfd = allocate_bytes<ElementScale>(sfd_elements);
  // This component uses CUTLASS per-row bias (Stride<1,0>), not per-column.
  // Even the zero-bias timing fixture must allocate the entire M extent.
  auto* bias = allocate_bytes<ElementBias>(m * sizeof(ElementBias));
  auto* norm = allocate_bytes<ElementCompute>(sizeof(ElementCompute));
  require_cuda(cudaMemset(a, 0x11, (a_elements + 1) / 2), "memset A");
  require_cuda(cudaMemset(b, 0x21, (b_elements + 1) / 2), "memset B");
  require_cuda(cudaMemset(c, 0, output_elements * sizeof(ElementOutput)), "memset C");
  require_cuda(cudaMemset(d, 0, (output_elements + 1) / 2), "memset D");
  require_cuda(cudaMemset(bias, 0, m * sizeof(ElementBias)), "memset bias");
  const std::uint8_t one_scale = ElementScale(1.0F).raw();
  require_cuda(cudaMemset(sfa, one_scale, sfa_elements), "memset SFA");
  require_cuda(cudaMemset(sfb, one_scale, sfb_elements), "memset SFB");
  // Opt-in numerical fixture only; preserve the historical timing workload.
  // Uniform scales cannot expose accidental aliasing of distinct scale bytes.
  if (std::getenv("STREAM1_HANDOFF_NONUNIFORM")) {
    std::vector<std::uint8_t> input_scales(sfa_elements);
    std::vector<std::uint8_t> weight_scales(sfb_elements);
    for (std::size_t i = 0; i < input_scales.size(); ++i) {
      input_scales[i] = ElementScale(
          0.125F * static_cast<float>(1U << ((i * 17 + i / 7) % 7))).raw();
    }
    for (std::size_t i = 0; i < weight_scales.size(); ++i) {
      weight_scales[i] = ElementScale(
          0.125F * static_cast<float>(1U << ((i * 13 + i / 11) % 7))).raw();
    }
    require_cuda(cudaMemcpy(sfa, input_scales.data(), input_scales.size(),
                            cudaMemcpyHostToDevice), "copy varied SFA");
    require_cuda(cudaMemcpy(sfb, weight_scales.data(), weight_scales.size(),
                            cudaMemcpyHostToDevice), "copy varied SFB");
  }
  const ElementCompute norm_value = std::getenv("STREAM1_HANDOFF_NONUNIFORM")
      ? 0.00390625F : 6.0F;
#if defined(STREAM1_HANDOFF_VALUE_READBACK)
  if (const char* seed_text = std::getenv("STREAM1_HANDOFF_VALUE_SEED")) {
    std::uint32_t seed = std::strtoul(seed_text, nullptr, 10);
    auto random_byte = [&]() { seed = seed * 1664525U + 1013904223U; return std::uint8_t(seed >> 24); };
    std::vector<std::uint8_t> av((a_elements + 1) / 2), bv((b_elements + 1) / 2);
    std::vector<ElementBias> bias_values(m);
    for (auto& v : av) v = random_byte();
    for (auto& v : bv) v = random_byte();
    for (auto& v : bias_values) v = ElementBias((int(random_byte()) - 128) * 0.125F);
    require_cuda(cudaMemcpy(a, av.data(), av.size(), cudaMemcpyHostToDevice), "copy seeded A");
    require_cuda(cudaMemcpy(b, bv.data(), bv.size(), cudaMemcpyHostToDevice), "copy seeded B");
    require_cuda(cudaMemcpy(bias, bias_values.data(), bias_values.size() * sizeof(ElementBias), cudaMemcpyHostToDevice), "copy seeded bias");
  }
#endif
  require_cuda(cudaMemcpy(norm, &norm_value, sizeof(norm_value), cudaMemcpyHostToDevice),
               "copy norm");

  typename SelectedKernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem_shape,
      {a, stride_a, b, stride_b, sfa, layout_sfa, sfb, layout_sfb},
      {{1.0F, 0.0F}, c, stride_c, d, stride_d}};
  arguments.epilogue.thread.block_scale_factor_ptr = sfd;
  arguments.epilogue.thread.norm_constant_ptr = norm;
  arguments.epilogue.thread.bias_ptr = bias;

  const std::size_t workspace_bytes =
      SelectedKernel::get_workspace_size(arguments);
  auto* workspace = allocate_bytes<std::uint8_t>(workspace_bytes == 0 ? 1 : workspace_bytes);
  if (!SelectedKernel::can_implement(arguments)) {
    std::cerr << "can_implement: false\n";
    return 4;
  }
  require_cutlass(
      SelectedKernel::initialize_workspace(
          arguments, workspace, nullptr),
      "initialize_workspace");
  auto params = SelectedKernel::to_underlying_arguments(arguments, workspace);
  const auto grid = SelectedKernel::get_grid_shape(params);
#if defined(STREAM1_HANDOFF_CHECKSUM_CONTROL)
  if (slots != 1 || std::size_t(grid.x) * grid.y * grid.z * 128 * sizeof(float) > sfd_elements) {
    std::cerr << "checksum capacity or single-stream contract failed\n";
    return 7;
  }
#endif
  constexpr int shared_bytes = sizeof(typename SelectedKernel::SharedStorage);
  require_cuda(
      cudaFuncSetAttribute(
          STREAM1_SELECTED_KERNEL_ENTRY,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          shared_bytes),
      "set dynamic shared memory");
  std::vector<cudaStream_t> streams(slots);
  for (auto& stream : streams) {
    require_cuda(cudaStreamCreate(&stream), "stream create");
  }
  auto launch = [&](cudaStream_t stream) {
    STREAM1_SELECTED_KERNEL_ENTRY<<<
        grid, SelectedKernel::MaxThreadsPerBlock, shared_bytes, stream>>>(params);
    require_cuda(cudaGetLastError(), "kernel launch");
  };
  for (int i = 0; i < STREAM1_HANDOFF_WARMUP_ITERATIONS; ++i) {
    for (auto stream : streams) launch(stream);
  }
  require_cuda(cudaDeviceSynchronize(), "warmup sync");
  cudaEvent_t begin{}, end{};
  require_cuda(cudaEventCreate(&begin), "event create begin");
  require_cuda(cudaEventCreate(&end), "event create end");
  require_cuda(cudaEventRecord(begin), "event record begin");
  for (auto stream : streams) {
    require_cuda(cudaStreamWaitEvent(stream, begin), "stream wait begin");
  }
  for (int i = 0; i < iterations; ++i) {
    for (auto stream : streams) launch(stream);
  }
  std::vector<cudaEvent_t> done(slots);
  for (int slot = 0; slot < slots; ++slot) {
    require_cuda(cudaEventCreate(&done[slot]), "event create done");
    require_cuda(cudaEventRecord(done[slot], streams[slot]), "event record done");
    require_cuda(cudaStreamWaitEvent(nullptr, done[slot]), "default wait done");
  }
  require_cuda(cudaEventRecord(end), "event record end");
  require_cuda(cudaEventSynchronize(end), "event sync");
  float elapsed_ms = 0.0F;
  require_cuda(cudaEventElapsedTime(&elapsed_ms, begin, end), "event elapsed");
  std::vector<std::uint8_t> host_d((output_elements + 1) / 2);
  std::vector<std::uint8_t> host_sfd(sfd_elements);
  require_cuda(cudaMemcpy(host_d.data(), d, host_d.size(), cudaMemcpyDeviceToHost),
               "copy D");
  require_cuda(cudaMemcpy(host_sfd.data(), sfd, host_sfd.size(), cudaMemcpyDeviceToHost),
               "copy SFD");
  const auto d_nonzero =
      std::count_if(host_d.begin(), host_d.end(), [](auto x) { return x != 0; });
  const auto sfd_nonzero =
      std::count_if(host_sfd.begin(), host_sfd.end(), [](auto x) { return x != 0; });
#if defined(STREAM1_HANDOFF_CHECKSUM_CONTROL)
  std::vector<float> checksum_values(std::size_t(grid.x) * grid.y * grid.z * 128);
  require_cuda(cudaMemcpy(checksum_values.data(), sfd,
                          checksum_values.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "read checksum values");
  // Uniform fixture: each dot product is 256 * .5 * (.5+1)/2 = 96.
  // Native 32x64 epilogue visits 16 outputs per consumer thread.
  if (!std::getenv("STREAM1_HANDOFF_NONUNIFORM")) {
    for (float value : checksum_values) {
      // Persistent launch may include CTAs that receive no tile. Their
      // pre-zeroed checksum slots are untouched; validate every written slot.
      if (value != 0.0F && value != 1536.0F) {
        std::cerr << "checksum numerical mismatch expected=1536 observed="
                  << value << '\n';
        return 9;
      }
    }
  }
  if (sfd_nonzero == 0 || d_nonzero != 0) {
    std::cerr << "checksum control missing or unexpected D output\n";
    return 8;
  }
#elif defined(STREAM1_CUTLASS_SKIP_GLOBAL_SFD)
  if (d_nonzero != 0 || sfd_nonzero != 0) {
    std::cerr << "no_global_store_contract=FAIL d_nonzero=" << d_nonzero
              << " sfd_nonzero=" << sfd_nonzero << '\n';
    return 6;
  }
#endif
  constexpr const char* variant_name =
      STREAM1_HANDOFF_CUSTOM == 2 ? "shared_only" :
      STREAM1_HANDOFF_CUSTOM == 1 ? "shared_handoff" : "stock";
  const double kernel_us = elapsed_ms * 1000.0 / iterations;
  const double useful_flops = 2.0 * m * n * k * slots;
  std::cout << "ff1_variant=" << variant_name
            << " rows=" << m
            << " slots=" << slots
            << " us=" << kernel_us
            << " useful_tflops=" << useful_flops / (kernel_us * 1.0e6)
            << " d_hash=" << fnv1a(host_d)
            << " sfd_hash=" << fnv1a(host_sfd)
            << " d_nonzero=" << d_nonzero
            << " sfd_nonzero=" << sfd_nonzero
#if defined(STREAM1_HANDOFF_CHECKSUM_CONTROL)
            << " checksum_control=pass checksum_store_in_timing=1"
#elif defined(STREAM1_CUTLASS_SKIP_GLOBAL_SFD)
            << " no_global_store_contract=pass"
#endif
            << " workspace_bytes=" << workspace_bytes << '\n';
  return 0;
}

#undef STREAM1_SELECTED_KERNEL_ENTRY
