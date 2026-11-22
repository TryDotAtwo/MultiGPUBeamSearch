#pragma once
#include "../src/config.hpp"
#include "../src/types.hpp"
#include "../cuda/stream1_executor_contract.hpp"
#include "../third_party/nlohmann/json.hpp"
#include <filesystem>
#include <fstream>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>
namespace beam::score_mode {
struct Request {
    std::filesystem::path reference_dir, output_dir, weight_dir;
    std::uint32_t outer=0, inner=0, lanes=0;
    Stream1Executor executor=Stream1Executor::NativeGraph;
    std::vector<State128> states;
    std::string reference_text;
};
inline std::uint32_t profile_u32(const char* key,std::uint32_t fallback) {
    const char* text=std::getenv(key);
    if(!text || !*text)return fallback;
    std::uint32_t value=0;
    const std::string s(text);
    const auto p=std::from_chars(s.data(),s.data()+s.size(),value);
    if(p.ec!=std::errc{} || p.ptr!=s.data()+s.size())
        throw std::invalid_argument(std::string("invalid production-score profile: ")+key);
    return value;
}
inline std::string bounded_text(const std::filesystem::path& path) {
    if(!std::filesystem::is_regular_file(path) || std::filesystem::file_size(path)>8U*1024U*1024U)
        throw std::invalid_argument("missing/oversized production-score input");
    std::ifstream file(path,std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(file)),{});
    if(file.bad() || text.size()>8U*1024U*1024U)
        throw std::invalid_argument("cannot read bounded production-score input");
    return text;
}
inline Request parse_request(int argc,const char*const* argv) {
    if(argc!=4 || !argv || !argv[1] || std::string(argv[1])!="--validate-stream1-scores" ||
       !argv[2] || !*argv[2] || !argv[3] || !*argv[3])
        throw std::invalid_argument("usage: production_runner --validate-stream1-scores REFERENCE_DIR OUTPUT_DIR");
    if(STATE_LEN!=96 || MOVE_COUNT!=24)
        throw std::invalid_argument("production-score mode currently requires Cube4 build");
    Request r;
    r.executor=resolve_stream1_executor(std::getenv("BEAM_STREAM1_EXECUTOR"));
    if(r.executor==Stream1Executor::LibTorchEager)
        throw std::invalid_argument("production-score mode requires native executor");
    const char* scoring_mode=std::getenv("BEAM_STREAM1_MODE");
    if(scoring_mode && std::string(scoring_mode)!="model")
        throw std::invalid_argument("production-score mode requires BEAM_STREAM1_MODE=model or unset");
    r.outer=profile_u32("BEAM_B_MICRO",8192);
    r.inner=profile_u32("BEAM_STREAM1_TRANSFORMER_MICRO",r.outer);
    r.lanes=profile_u32("BEAM_STREAM1_CONCURRENCY",1);
    if(!r.outer || r.outer>65536 || !r.inner || r.inner>r.outer || !r.lanes || r.lanes>64 ||
       2ULL*r.outer*r.lanes*MOVE_COUNT>4194304ULL)
        throw std::invalid_argument("production-score profile exceeds bounded coverage/output budget");
    r.reference_dir=std::filesystem::canonical(argv[2]);
    r.output_dir=std::filesystem::absolute(argv[3]).lexically_normal();
    if(std::filesystem::exists(r.output_dir) || std::filesystem::is_symlink(r.output_dir))
        throw std::invalid_argument("production-score output must not exist");
    const char* weight_text=std::getenv("BEAM_WEIGHT_DIR");
    r.weight_dir=std::filesystem::canonical(weight_text && *weight_text ?
        std::filesystem::path(weight_text):r.reference_dir/"weights_fp16");
    if(!std::filesystem::is_regular_file(r.weight_dir/"manifest.json"))
        throw std::invalid_argument("missing production-score weight manifest");
    r.reference_text=bounded_text(r.reference_dir/"reference.json");
    const auto j=nlohmann::json::parse(r.reference_text);
    if(!j.is_object() || !j.contains("states") || !j["states"].is_array() ||
       j["states"].empty() || j["states"].size()>1024 || !j.contains("scores_fp32") ||
       !j["scores_fp32"].is_array() || j["scores_fp32"].size()!=j["states"].size())
        throw std::invalid_argument("invalid production-score reference rows");
    r.states.resize(j["states"].size());
    for(std::size_t row=0;row<r.states.size();++row) {
        const auto& values=j["states"][row];
        const auto& scores=j["scores_fp32"][row];
        if(!values.is_array() || values.size()!=96 || !scores.is_array() || scores.size()!=24)
            throw std::invalid_argument("invalid production-score reference shape");
        auto& state=r.states[row];state={};
        for(unsigned i=0;i<96;++i) {
            if(!values[i].is_number_integer())throw std::invalid_argument("noninteger reference state");
            const auto value=values[i].get<long long>();
            if(value<0 || value>=6)throw std::invalid_argument("reference state exceeds Cube4 alphabet");
            state.v[i]=static_cast<std::uint8_t>(value);
        }
        for(const auto& score:scores)
            if(!score.is_number() || !std::isfinite(score.get<double>()))
                throw std::invalid_argument("nonfinite reference score");
    }
    return r;
}
}
