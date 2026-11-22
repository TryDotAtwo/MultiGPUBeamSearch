#include "../cuda/stream1_resolved_gemm.hpp"
#include <stdexcept>
#include <iostream>
using namespace beam;
using F = Stream1TransformerGemmFamily;
using P = Stream1TransformerGemmPolicy;
using S = Stream1TransformerGemmStagePolicy;
using W = Stream1TransformerGemmSwizzlePolicy;
using E = Stream1TransformerResidualEpiloguePolicy;
static void require(bool value, const char* why) { if (!value) throw std::runtime_error(why); }
template<class Fn> static void rejected(Fn fn) {
    bool bad = false;
    try { fn(); } catch (const std::invalid_argument&) { bad = true; }
    require(bad, "unsupported execution admitted");
}
int main() {
    auto q = resolve_stream1_gemm(F::Qkv, 75, true, nullptr, nullptr, nullptr, nullptr);
    require(q.policy == P::Baseline && q.stage == S::Stages2 && q.swizzle == W::Identity1,
            "SM75 QKV does not use three-stage default");
    auto r75 = resolve_stream1_gemm(F::AttentionOut, 75, true, nullptr, nullptr, nullptr, nullptr);
    require(r75.stage == S::Stages2, "SM75 separate residual inherits CUTLASS two-stage default");
    auto f = resolve_stream1_gemm(F::Ff1, 75, true, nullptr, nullptr, nullptr, nullptr);
    require(f.policy == P::Baseline && f.stage == S::Stages2, "SM75 FF1 default must be two stages");
    auto modern = resolve_stream1_gemm(F::Ff1, 86, true, nullptr, nullptr, nullptr, nullptr);
    require(modern.policy == P::Baseline && modern.stage == S::Stages3, "SM86 FF1 default changed");
    auto blackwell = resolve_stream1_gemm(F::Ff1, 120, true, nullptr, nullptr, nullptr, nullptr);
    require(blackwell.policy == P::M128N128 && blackwell.swizzle == W::Identity4,
            "SM120 FF1 default selection differs from launch");
    auto qb = resolve_stream1_gemm(F::Qkv, 120, true, nullptr, nullptr, nullptr, nullptr);
    require(qb.policy == P::M128N128 && qb.swizzle == W::Identity1, "SM120 QKV default");
    auto bf = resolve_stream1_gemm(F::Ff1, 86, false, "m64n128", "2", "1", nullptr);
    require(bf.policy == P::Baseline && bf.stage == S::Stages3 && bf.swizzle == W::Identity1,
            "BF16 launch is fixed baseline, not requested FP16 tile");
    auto fused = resolve_stream1_gemm(F::Ff2, 75, true, "m128n128", nullptr, "2", "fused");
    require(fused.policy == P::M128N128 && fused.epilogue == E::FusedBiasRound &&
            fused.stage == S::Stages3 && fused.swizzle == W::Identity2, "SM75 fused residual selection");
    rejected([] { resolve_stream1_gemm(F::Ff1, 75, true, nullptr, "3", nullptr, nullptr); });
    rejected([] { resolve_stream1_gemm(F::Qkv, 75, true, "m128n128", nullptr, "4", nullptr); });
    rejected([] { resolve_stream1_gemm(F::Ff1, 86, true, "m128n128", "2", "4", nullptr); });
    rejected([] { resolve_stream1_gemm(F::AttentionOut, 86, true, "m128n128", nullptr, "2", "separate"); });
    rejected([] { resolve_stream1_gemm(F::Ff2, 120, true, nullptr, nullptr, nullptr, "fused"); });
    rejected([] { resolve_stream1_gemm(F::Ff2, 90, false, "m128n128", nullptr, nullptr, "fused"); });
    rejected([] { resolve_stream1_gemm(F::Qkv, 70, true, nullptr, nullptr, nullptr, nullptr); });
    rejected([] { resolve_stream1_gemm(F::Qkv, 75, false, nullptr, nullptr, nullptr, nullptr); });
    rejected([] { resolve_stream1_gemm(F::Qkv, 86, true, "m64n64", nullptr, nullptr, nullptr); });
    rejected([] { resolve_stream1_gemm(F::Cls, 86, true, nullptr, nullptr, nullptr, nullptr); });
    std::cout << "resolved_gemm=pass\n";
}
