#pragma once

// The owning Python process is the only writer. Every rank observes the same
// atomically published request; a rank failure invalidates the entire group.
#include <filesystem>
#include <fstream>
#include <thread>
#include <chrono>
#include "../third_party/nlohmann/json.hpp"
#include <iostream>
#include <cstdint>
#if defined(__linux__)
#include <unistd.h>
#endif

namespace beam::resident {
class Mailbox {
    std::filesystem::path root_;
    std::uint32_t rank_;
    std::ofstream log_;
    std::streambuf* original_ = nullptr;
#if defined(__linux__)
    pid_t owner_ = getppid();
#endif
public:
    Mailbox(std::filesystem::path root, std::uint32_t rank):root_(std::move(root)),rank_(rank) {}
    ~Mailbox() { restore(); }
    bool enabled() const { return !root_.empty(); }
    void restore() {
        if(original_) { std::cout.flush(); std::cout.rdbuf(original_); original_=nullptr; }
        if(log_.is_open()) log_.close();
    }
    bool next(std::size_t sequence, nlohmann::json& request) {
        auto path=root_/ ("request-"+std::to_string(sequence)+".json");
        while(!std::filesystem::exists(path)) {
            if(std::filesystem::exists(root_/"stop")) return false;
#if defined(__linux__)
            if(getppid()!=owner_) return false;
#endif
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        if(std::filesystem::file_size(path)>65536) throw std::runtime_error("resident request too large");
        std::ifstream input(path); input>>request;
        if(request.at("sequence").get<std::size_t>()!=sequence)
            throw std::runtime_error("resident request sequence mismatch");
        const auto directory=std::filesystem::path(request.at("run_dir").get<std::string>());
        if(!directory.is_absolute() || !std::filesystem::is_directory(directory))
            throw std::runtime_error("resident output directory must exist and be absolute");
        log_.open(directory/("resident-rank-"+std::to_string(rank_)+".log"),std::ios::trunc);
        if(!log_) throw std::runtime_error("cannot open resident task log");
        original_=std::cout.rdbuf(log_.rdbuf());
        return true;
    }
    void complete(std::size_t sequence) {
        restore();
        auto path=root_/("done-"+std::to_string(sequence)+"-"+std::to_string(rank_)+".json");
        auto temporary=path; temporary += ".tmp";
        { std::ofstream out(temporary); out<<nlohmann::json{{"sequence",sequence},{"rank",rank_}};
          out.flush(); if(!out) throw std::runtime_error("cannot write resident completion"); }
        std::filesystem::rename(temporary,path);
    }
};
}
