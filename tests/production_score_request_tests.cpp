#include "../tools/production_score_request.hpp"
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace beam;
using namespace beam::score_mode;
static void env(const char* name,const char* value) {
#ifdef _WIN32
    if(_putenv_s(name,value)) throw std::runtime_error("fixture environment");
#else
    if(setenv(name,value,1)) throw std::runtime_error("fixture environment");
#endif
}
template<class Fn> void rejects(Fn fn) {
    bool bad=false;try{fn();}catch(const std::exception&){bad=true;}
    if(!bad)throw std::runtime_error("invalid production-score request accepted");
}
int main() {
    auto root=std::filesystem::temp_directory_path()/
        ("beam-score-request-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(root/"weights_fp16");
    struct Cleanup{std::filesystem::path root;~Cleanup(){std::error_code ec;std::filesystem::remove_all(root,ec);}} cleanup{root};
    nlohmann::json state=nlohmann::json::array();for(unsigned i=0;i<96;++i)state.push_back(i%6);
    nlohmann::json reference={{"states",nlohmann::json::array({state})},
        {"scores_fp32",nlohmann::json::array({std::vector<double>(24,0.)})}};
    auto write=[&]{std::ofstream(root/"reference.json")<<reference.dump();};write();
    std::ofstream(root/"weights_fp16/manifest.json")<<"{}";
    const auto r=root.string(),o=(root/"out").string();
    const char* argv[]={"runner","--validate-stream1-scores",r.c_str(),o.c_str()};
    env("BEAM_B_MICRO","32");env("BEAM_STREAM1_TRANSFORMER_MICRO","13");
    env("BEAM_STREAM1_CONCURRENCY","2");env("BEAM_STREAM1_EXECUTOR","native_cuda_graph");
    env("BEAM_STREAM1_MODE","model");
    env("BEAM_WEIGHT_DIR",(root/"weights_fp16").string().c_str());
    const auto request=parse_request(4,argv);
    if(request.outer!=32 || request.inner!=13 || request.lanes!=2 || request.states.size()!=1)
        throw std::runtime_error("actual profile/state request was not preserved");
    for(unsigned i=96;i<112;++i)if(request.states[0].v[i])throw std::runtime_error("input padding not zero");
    rejects([&]{parse_request(3,argv);});
    // A neural check must not qualify a different uniform-score beam profile.
    for(const auto* unsupported:{"uniform","bogus",""}) {
        env("BEAM_STREAM1_MODE",unsupported);
        rejects([&]{parse_request(4,argv);});
    }
    env("BEAM_STREAM1_MODE","model");
    env("BEAM_STREAM1_EXECUTOR","libtorch_eager");rejects([&]{parse_request(4,argv);});
    env("BEAM_STREAM1_EXECUTOR","native_eager");
    if(parse_request(4,argv).executor!=Stream1Executor::NativeEager)throw std::runtime_error("wrong executor");
    env("BEAM_STREAM1_TRANSFORMER_MICRO","33");rejects([&]{parse_request(4,argv);});
    env("BEAM_STREAM1_TRANSFORMER_MICRO","13");
    env("BEAM_B_MICRO","0");rejects([&]{parse_request(4,argv);});env("BEAM_B_MICRO","32");
    env("BEAM_STREAM1_CONCURRENCY","65");rejects([&]{parse_request(4,argv);});
    env("BEAM_STREAM1_CONCURRENCY","2");
    reference["states"][0][0]=6;write();rejects([&]{parse_request(4,argv);});
    reference["states"][0][0]=0;reference["states"][0].push_back(0);write();rejects([&]{parse_request(4,argv);});
    reference["states"][0]=state;reference["scores_fp32"][0].push_back(0.);write();rejects([&]{parse_request(4,argv);});
    reference["scores_fp32"][0]=std::vector<double>(24,0.);write();
    std::filesystem::create_directory(root/"out");rejects([&]{parse_request(4,argv);});
    std::filesystem::remove(root/"out");std::filesystem::remove(root/"reference.json");
    rejects([&]{parse_request(4,argv);});
    std::cout<<"production_score_request=pass\n";
}
