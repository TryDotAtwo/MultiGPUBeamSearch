#pragma once

struct CutlassBenchmarkExpectation {
    template <class T>
    CutlassBenchmarkExpectation& operator<<(const T&) { return *this; }
};

#if defined(BEAM_BENCH_K_SWEEP_VERIFY)
#include <stdexcept>
inline CutlassBenchmarkExpectation cutlass_benchmark_check(bool passed, const char* expression) {
    if (!passed) throw std::runtime_error(expression);
    return {};
}
// CUTLASS initialization itself is nested in EXPECT_TRUE. Never discard
// these expressions in a validated benchmark (including NDEBUG builds).
#define EXPECT_TRUE(...) cutlass_benchmark_check(bool(__VA_ARGS__), #__VA_ARGS__)
#define EXPECT_EQ(a, b) cutlass_benchmark_check((a) == (b), #a " == " #b)
#define EXPECT_GT(a, b) cutlass_benchmark_check((a) > (b), #a " > " #b)
#else
#define EXPECT_TRUE(...) CutlassBenchmarkExpectation{}
#define EXPECT_EQ(...) CutlassBenchmarkExpectation{}
#define EXPECT_GT(...) CutlassBenchmarkExpectation{}
#endif
#define TEST(suite_name, test_name) static void suite_name##_##test_name()
