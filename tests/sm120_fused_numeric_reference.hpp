#pragma once

// Test-only scalar oracle. It deliberately does not use the GPU handoff writer,
// MMA fragment coordinates, or shared-memory layouts to derive expected values.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

namespace sm120_fused_reference {

constexpr int kRows = 128;
constexpr int kModel = 256;
#ifndef STREAM1_FUSED_TEST_HIDDEN
#define STREAM1_FUSED_TEST_HIDDEN 256
#endif
constexpr int kHidden = STREAM1_FUSED_TEST_HIDDEN;
static_assert(kHidden >= 256 && kHidden <= 1024 && kHidden % 256 == 0);
constexpr int kScaleVector = 16;

inline float decode_e2m1(std::uint8_t raw) {
  constexpr float magnitudes[] = {0, 0.5F, 1, 1.5F, 2, 3, 4, 6};
  return (raw & 8U) ? -magnitudes[raw & 7U] : magnitudes[raw & 7U];
}

inline std::uint32_t mix(std::uint32_t x) {
  x ^= x >> 16;
  x *= 0x7feb352dU;
  x ^= x >> 15;
  x *= 0x846ca68bU;
  return x ^ (x >> 16);
}

struct Matrix {
  int rows;
  int columns;
  std::vector<std::uint8_t> packed;
  std::vector<float> scales;

  Matrix(int m, int k)
      : rows(m), columns(k), packed(m * k / 2),
        scales(m * k / kScaleVector, 1.0F) {}

  float value(int row, int column) const {
    const int index = row * columns + column;
    const auto raw = (packed[index / 2] >> (4 * (index & 1))) & 15;
    return decode_e2m1(raw) * scales[index / kScaleVector];
  }
};

struct Fixture {
  Matrix a;
  Matrix b1{kHidden, kModel};
  Matrix b2{kModel, kHidden};
  std::vector<float> bias = std::vector<float>(kHidden);
  std::vector<float> ff1;
  std::vector<float> hidden;
  std::vector<float> ff2;
  std::vector<float> ff2_bias = std::vector<float>(kModel);
  std::vector<float> residual;
  std::vector<float> layernorm_gamma = std::vector<float>(kModel);
  std::vector<float> layernorm_beta = std::vector<float>(kModel);
  std::vector<float> residual_fp32;
  std::vector<float> residual_fp16;
  std::vector<float> normalized;
  std::vector<float> next_nvfp4;

  explicit Fixture(int rows = kRows)
      : a(rows, kModel), ff1(rows*kHidden), hidden(rows*kHidden),
        ff2(rows*kModel), residual(rows*kModel), residual_fp32(rows*kModel),
        residual_fp16(rows*kModel), normalized(rows*kModel),
        next_nvfp4(rows*kModel) {}
};

inline void fill_matrix(Matrix& matrix, std::uint32_t seed,
                        std::uint32_t salt, std::uint8_t uniform_byte) {
  for (std::size_t i = 0; i < matrix.packed.size(); ++i) {
    matrix.packed[i] = seed == 0 ? uniform_byte :
        static_cast<std::uint8_t>(mix(seed ^ salt ^ std::uint32_t(i)));
  }
  for (std::size_t i = 0; i < matrix.scales.size(); ++i) {
    matrix.scales[i] = seed == 0 ? 1.0F :
        std::ldexp(1.0F, int(mix(seed + salt + std::uint32_t(i)) % 4) - 2);
  }
}

inline Fixture make_fixture(std::uint32_t seed, int rows = kRows) {
  Fixture f(rows);
  fill_matrix(f.a, seed, 0x1234U, 0x11U);
  fill_matrix(f.b1, seed, 0x5678U, 0x21U);
  fill_matrix(f.b2, seed, 0x9abcU, 0x31U);
  // Keep the deterministic oracle inside the production FP16 residual range.
  // The original all-positive hand fixture produced FF2=98,304 for H=1024,
  // so both the expected and actual half stores became +inf and stopped being
  // a useful residual/LN contract check.  Quartering only FF2's offline scale
  // preserves exact NVFP4 arithmetic while making the hand case finite.
  for (float& scale : f.b2.scales) scale *= 0.25F;
  for (int n = 0; n < kHidden; ++n) {
    f.bias[n] = seed == 0 ? 0.0F : float(int(mix(seed ^ n) % 33) - 16) / 8;
  }
  for (int n = 0; n < kModel; ++n) {
    f.ff2_bias[n] = seed == 0 ? 0.25F :
        float(int(mix(seed ^ 0x2468U ^ std::uint32_t(n)) % 33) - 16) / 32;
    f.layernorm_gamma[n] = seed == 0 ? 1.0F :
        1.0F + float(int(mix(seed ^ 0x1357U ^ std::uint32_t(n)) % 9) - 4) / 32;
    f.layernorm_beta[n] = seed == 0 ? 0.0F :
        float(int(mix(seed ^ 0xabcdU ^ std::uint32_t(n)) % 17) - 8) / 64;
  }
  for (int m = 0; m < rows; ++m) {
    for (int n = 0; n < kModel; ++n) {
      f.residual[m*kModel+n] = seed == 0 ? float(n & 3) / 8 :
          float(int(mix(seed ^ 0xdeadU ^ std::uint32_t(m*kModel+n)) % 33) - 16) / 32;
    }
  }
  return f;
}

inline std::vector<int> select_oracle_rows(int rows, int oracle_tiles) {
  const int total_tiles = rows / kRows;
  std::vector<int> selected;
  selected.reserve(std::size_t(oracle_tiles) * kRows);
  for (int sample = 0; sample < oracle_tiles; ++sample) {
    const int tile = oracle_tiles == 1
        ? 0
        : int((std::int64_t(sample) * (total_tiles - 1) +
               (oracle_tiles - 1) / 2) /
              (oracle_tiles - 1));
    for (int row = tile * kRows; row < (tile + 1) * kRows; ++row) {
      selected.push_back(row);
    }
  }
  return selected;
}

// Raw FF1 is compared before quantization so a wrong slice, shared overlay,
// or register coordinate cannot hide behind a coincidentally equal checksum.
// CUTLASS scalar conversion specifies only the public FP4/UE4M3 encoding; the
// reference matrix multiplication and activation are independent host loops.
inline void compute_rows(Fixture& f, std::vector<int> const& rows) {
  for (int m : rows) {
    for (int n = 0; n < kHidden; ++n) {
      double sum = 0;
      for (int k = 0; k < kModel; ++k) sum += double(f.a.value(m, k)) * f.b1.value(n, k);
      f.ff1[m * kHidden + n] = float(sum);
      f.hidden[m * kHidden + n] = std::max(0.0F, float(sum) + f.bias[n]);
    }
    for (int first = 0; first < kHidden; first += kScaleVector) {
      float maximum = 0;
      for (int i = 0; i < kScaleVector; ++i) maximum = std::max(maximum, f.hidden[m * kHidden + first + i]);
      const float scale = float(cutlass::float_ue4m3_t(maximum > 0 ? maximum / 6 : 1));
      // The encoder contract uses one rounded FP32 reciprocal per group and
      // then multiplies each value. Direct division differs at FP4 midpoints:
      // seed17/H1024 yields 75/60=1.25 but 75*float(1/60)=1.25000012.
      // This arithmetic boundary is independently checked by the GPU probe.
      const float inverse_scale = scale == 0 ? 0 : 1.0F / scale;
      for (int i = 0; i < kScaleVector; ++i) {
        auto& value = f.hidden[m * kHidden + first + i];
        // A zero representable scale reconstructs zero. Do not create a host
        // NaN while checking that boundary; raw encodings are not the gate.
        value = scale == 0 ? 0 : float(cutlass::float_e2m1_t(value * inverse_scale)) * scale;
      }
    }
    for (int n = 0; n < kModel; ++n) {
      double sum = 0;
      for (int k = 0; k < kHidden; ++k) sum += double(f.hidden[m * kHidden + k]) * f.b2.value(n, k);
      f.ff2[m * kModel + n] = float(sum);
    }
    float mean_sum = 0.0F;
    for (int n = 0; n < kModel; ++n) {
      const auto index = m*kModel+n;
      const float x = f.ff2[index] + f.ff2_bias[n] + f.residual[index];
      f.residual_fp32[index] = x;
      f.residual_fp16[index] = static_cast<float>(cutlass::half_t(x));
      mean_sum += x;
    }
    const float mean = mean_sum / float(kModel);
    float variance_sum = 0.0F;
    for (int n = 0; n < kModel; ++n) {
      const float centered = f.residual_fp32[m*kModel+n] - mean;
      variance_sum += centered * centered;
    }
    const float inverse_std = 1.0F / std::sqrt(variance_sum / float(kModel) + 1.0e-5F);
    for (int n = 0; n < kModel; ++n) {
      const auto index = m*kModel+n;
      f.normalized[index] =
          (f.residual_fp32[index] - mean) * inverse_std *
              f.layernorm_gamma[n] + f.layernorm_beta[n];
    }
    for (int first = 0; first < kModel; first += kScaleVector) {
      float maximum = 0.0F;
      for (int i = 0; i < kScaleVector; ++i) {
        maximum = std::max(maximum, std::abs(f.normalized[m*kModel+first+i]));
      }
      const float scale = float(cutlass::float_ue4m3_t(
          maximum > 0.0F ? maximum / 6.0F : 1.0F));
      const float inverse_scale = scale == 0.0F ? 0.0F : 1.0F / scale;
      for (int i = 0; i < kScaleVector; ++i) {
        const auto index = m*kModel+first+i;
        f.next_nvfp4[index] = scale == 0.0F ? 0.0F :
            float(cutlass::float_e2m1_t(f.normalized[index] * inverse_scale)) * scale;
      }
    }
  }
}

inline void compute(Fixture& f) {
  std::vector<int> rows(f.a.rows);
  for (int row = 0; row < f.a.rows; ++row) rows[row] = row;
  compute_rows(f, rows);
}

template <class Layout>
std::vector<std::uint8_t> encode_scales(Matrix const& matrix, Layout layout,
                                       std::size_t storage_bytes) {
  std::vector<std::uint8_t> bytes(storage_bytes, 0);
  for (int row = 0; row < matrix.rows; ++row) {
    for (int col = 0; col < matrix.columns; col += kScaleVector) {
      const auto offset = static_cast<std::size_t>(layout(row, col, 0));
      bytes.at(offset) = cutlass::float_ue4m3_t(
          matrix.scales[(row * matrix.columns + col) / kScaleVector]).raw();
    }
  }
  return bytes;
}

inline bool compare(char const* name, std::vector<float> const& actual,
                    std::vector<float> const& expected) {
  std::size_t errors = 0;
  double max_abs = 0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const double delta = std::abs(double(actual.at(i)) - expected[i]);
    const double tolerance = 1e-4 + 1e-5 * std::abs(double(expected[i]));
    if (!std::isfinite(actual[i]) || delta > tolerance) {
      if (errors < 4) std::cerr << name << " mismatch index=" << i
          << " actual=" << actual[i] << " expected=" << expected[i] << '\n';
      ++errors;
    }
    if (std::isfinite(delta)) max_abs = std::max(max_abs, delta);
  }
  std::cout << "oracle=" << name << " mismatches=" << errors
            << " elements=" << expected.size() << " max_abs=" << max_abs << '\n';
  return errors == 0;
}

inline bool compare_rows(char const* name, std::vector<float> const& actual,
                         std::vector<float> const& expected, int columns,
                         std::vector<int> const& rows) {
  std::size_t errors = 0;
  double max_abs = 0;
  for (int row : rows) {
    for (int column = 0; column < columns; ++column) {
      const auto index = std::size_t(row) * columns + column;
      const double delta = std::abs(double(actual.at(index)) - expected.at(index));
      const double tolerance = 1e-4 + 1e-5 * std::abs(double(expected.at(index)));
      if (!std::isfinite(actual.at(index)) || delta > tolerance) {
        if (errors < 4) std::cerr << name << " mismatch index=" << index
            << " actual=" << actual.at(index) << " expected=" << expected.at(index) << '\n';
        ++errors;
      }
      if (std::isfinite(delta)) max_abs = std::max(max_abs, delta);
    }
  }
  std::cout << "oracle=" << name << " mismatches=" << errors
            << " elements=" << std::size_t(columns) * rows.size()
            << " sampled_rows=" << rows.size() << " max_abs=" << max_abs << '\n';
  return errors == 0;
}

}  // namespace sm120_fused_reference
