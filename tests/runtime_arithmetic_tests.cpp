#include "../cuda/runtime_config.hpp"
#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <string>
#include <algorithm>

int main() {
    using namespace beam;
    unsigned failures = 0;
    auto check = [&](bool ok, const char* label) {
        if (!ok) { ++failures; std::cerr << "FAIL " << label << '\n'; }
    };
    auto overflow = [&](auto action, const char* label) {
        try { action(); check(false, label); }
        catch (const std::overflow_error&) {}
    };
    // Exercise the real configuration entry point, not a copy of its parser.
    const char* setting = "BEAM_GPU_HEADROOM_BYTES";
    const char* previous = std::getenv(setting);
    const bool had_previous = previous != nullptr;
    const std::string saved = previous ? previous : "";
    auto set_headroom = [&](const char* value) {
#ifdef _WIN32
        _putenv_s(setting, value ? value : "");
#else
        if (value) setenv(setting, value, 1); else unsetenv(setting);
#endif
    };
    for (const char* bad : {"-1", "-0", "+1", " 1", "18446744073709551616",
                            "999999999999999999999999999999999999"}) {
        set_headroom(bad);
        bool rejected_by_parser = false;
        try { build_runtime_config_from_budget(4096, 1, 0, 16ULL << 30); }
        catch (const std::invalid_argument& e) {
            rejected_by_parser = std::string(e.what()).find(setting) != std::string::npos;
        }
        catch (const std::exception&) {} // Later budget failures do not prove admission.
        check(rejected_by_parser, bad);
    }
    set_headroom(had_previous ? saved.c_str() : nullptr);
    {
        const char* names[] = {"BEAM_STREAM1_EXECUTOR", "BEAM_ENSEMBLE_RESERVE_BYTES", "BEAM_BLEND_DIR", "BEAM_RING_COUNT_LIMIT"};
        std::string saved_values[4]; bool existed[4];
        auto set = [](const char* name, const char* value) {
#ifdef _WIN32
            _putenv_s(name, value ? value : "");
#else
            if (value) setenv(name, value, 1); else unsetenv(name);
#endif
        };
        for (unsigned i=0;i<4;++i) {
            const char* value=std::getenv(names[i]); existed[i]=value!=nullptr;
            saved_values[i]=value?value:""; set(names[i],nullptr);
        }
        const auto legacy=build_runtime_config_from_budget(1048576,2,0,16ULL<<30);
        set(names[0],"libtorch_eager");
        const auto unreserved=build_runtime_config_from_budget(1048576,2,0,16ULL<<30);
        check(unreserved.config.ring_count<=16,"eager ensemble automatic default bounds physical graphs");
        check(unreserved.estimated_non_static_device_bytes==legacy.estimated_non_static_device_bytes,
              "unreserved LibTorch retains legacy conservative accounting");
        set(names[1],"1073741824");
        const auto reserved=build_runtime_config_from_budget(1048576,2,0,16ULL<<30);
        check(reserved.estimated_non_static_device_bytes>0 &&
              reserved.estimated_non_static_device_bytes<legacy.estimated_non_static_device_bytes,
              "reserved LibTorch retains tables without native weights and scratch");
        set(names[0],"native_graph");
        const auto native=build_runtime_config_from_budget(1048576,2,0,16ULL<<30);
        check(native.estimated_non_static_device_bytes==legacy.estimated_non_static_device_bytes,
              "Torch reserve setting does not reduce native executor accounting");
        set(names[0],"libtorch_eager"); set(names[1],nullptr); set(names[2],"/test/blend");
        const auto blend=build_runtime_config_from_budget(1048576,2,0,16ULL<<30);
        check(blend.estimated_non_static_device_bytes==reserved.estimated_non_static_device_bytes,
              "legacy blend reserve uses the same table accounting");
        for (unsigned i=0;i<4;++i) set(names[i],existed[i]?saved_values[i].c_str():nullptr);
    }
    {
        const auto admitted = build_runtime_config_from_budget(1048576, 2, 0, 16ULL << 30);
        const auto& c = admitted.config;
        const auto recv_reserve = (std::uint64_t(c.stream3_batch_candidates) *
            c.stream5_recv_capacity_scale_ppm + 999999ULL) / 1000000ULL;
        const auto writable_reserve = std::max<std::uint64_t>(c.stream3_batch_candidates,recv_reserve);
        check(c.shard_capacity_candidates >= logical_shard_size_for(c) + writable_reserve,
              "automatic shard capacity retains an entire owner batch reserve");
    }
    {
        const char* key="BEAM_RING_COUNT_LIMIT";
        const char* old=std::getenv(key);const bool existed=old!=nullptr;
        const std::string saved_limit=old?old:"";
        auto set_limit=[&](const char* value) {
#ifdef _WIN32
            _putenv_s(key,value?value:"");
#else
            if(value)setenv(key,value,1);else unsetenv(key);
#endif
        };
        set_limit(nullptr);
        const auto original=build_runtime_config_from_budget(4194304,2,0,32ULL<<30);
        set_limit("1");
        const auto bounded=build_runtime_config_from_budget(4194304,2,0,32ULL<<30);
        check(bounded.config.ring_count==1,"physical ring pool is bounded in automatic mode");
        check(bounded.config.user_global_beam_width==original.config.user_global_beam_width &&
              local_frontier_capacity(bounded.config)>=4194304/2,
              "physical ring bound never truncates the logical frontier");
        for(const char* bad:{"0","-1","4294967296"}) {
            set_limit(bad);bool rejected=false;
            try{build_runtime_config_from_budget(4194304,2,0,32ULL<<30);}
            catch(const std::invalid_argument& e){rejected=std::string(e.what()).find(key)!=std::string::npos;}
            check(rejected,"invalid physical ring limit rejected");
        }
        set_limit(existed?saved_limit.c_str():nullptr);
    }
    check(ceil_div_u64(UINT64_MAX, 2) == (1ULL << 63), "ceil maximum / 2");
    check(ceil_div_u64(UINT64_MAX, UINT64_MAX) == 1, "ceil maximum / maximum");
    check(ceil_div_u64(0, UINT64_MAX) == 0, "ceil zero");
    check(ceil_div_u64(17, 4) == 5, "ceil normal");
    RuntimeConfig config{};
    config.world_size = 1; config.shard_count = 1; config.stream4_batch_alignment = 1;
    config.user_global_beam_width = UINT64_MAX;
    overflow([&] { gross_generated_candidates_per_depth(config); }, "candidate product overflow");
    overflow([&] { estimated_sort_work_units(UINT64_MAX, 4); }, "sort work overflow");
    config.user_global_beam_width = 256000000;
    check(estimated_stream4_input_candidates_per_depth(config, 1000000) == 256000000ULL * MOVE_COUNT,
          "normal ppm identity");
    config.user_global_beam_width = UINT64_MAX / MOVE_COUNT;
    check(estimated_stream4_input_candidates_per_depth(config, 1000000) ==
          (UINT64_MAX / MOVE_COUNT) * MOVE_COUNT, "large ppm identity without intermediate overflow");
    overflow([&] { estimated_stream4_input_candidates_per_depth(config, 2000000); },
             "ppm result overflow");
    config.user_global_beam_width = 1;
    check(estimated_stream4_input_candidates_per_depth(config, 1000001) == MOVE_COUNT + 1,
          "ppm rounds fractional candidate upward");
    config.world_size = UINT32_MAX; config.shard_count = UINT32_MAX;
    config.stream4_batch_alignment = UINT32_MAX;
    overflow([&] { local_frontier_capacity(config); }, "alignment product overflow");
    std::cout << "runtime_arithmetic failures=" << failures << '\n';
    return failures ? 1 : 0;
}
