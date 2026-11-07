#pragma once
#include "engine_control.hpp"
#include <string>
#include <vector>

namespace beam::native {
class HistoryBridge {
    beam_history_api_v1 api_{};
public:
    HistoryBridge() {
        if(auto* api=current_history()) api_=*api;
    }
    explicit operator bool() const {return api_.user_data != nullptr;}
    void check(int32_t result) const {
        if(result == BEAM_OK) return;
        uint64_t required=0;
        api_.error(api_.user_data, nullptr, 0, &required);
        if(required==0 || required>(1U<<20U)) throw std::runtime_error("Rust history callback failed");
        std::vector<char> error(required,0);
        if(api_.error(api_.user_data,error.data(),error.size(),&required)!=BEAM_OK)
            throw std::runtime_error("Rust history error retrieval failed");
        throw std::runtime_error(std::string("Rust history: ")+error.data());
    }
    void initialize(const std::string& config) const {check(api_.initialize(api_.user_data,config.data(),config.size()));}
    void write(uint32_t depth,const void* candidates,uint64_t count) const {check(api_.write(api_.user_data,depth,candidates,count));}
    beam_history_entry_v1 read(uint32_t depth,uint64_t index) const {
        beam_history_entry_v1 entry{};check(api_.read(api_.user_data,depth,index,&entry));return entry;
    }
    beam_history_stats_v1 stats() const {
        beam_history_stats_v1 stats{};check(api_.stats(api_.user_data,&stats));return stats;
    }
    void close() const {check(api_.close(api_.user_data));}
};
static_assert(sizeof(beam_history_entry_v1)==16);
static_assert(sizeof(beam_history_stats_v1)==112);
static_assert(sizeof(beam_history_api_v1)==64);
}
