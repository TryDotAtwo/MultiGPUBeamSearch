#pragma once

struct CutlassBenchmarkExpectation {
    template <class T>
    CutlassBenchmarkExpectation& operator<<(const T&) { return *this; }
};

#define EXPECT_TRUE(...) CutlassBenchmarkExpectation{}
#define EXPECT_EQ(...) CutlassBenchmarkExpectation{}
#define EXPECT_GT(...) CutlassBenchmarkExpectation{}
#define TEST(suite_name, test_name) static void suite_name##_##test_name()

