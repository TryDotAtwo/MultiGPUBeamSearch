#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace beam::sha256 {

class Digest {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        total_bytes_ += size;
        while (size != 0U) {
            const std::size_t take = std::min(size, block_.size() - buffered_);
            for (std::size_t i = 0; i < take; ++i) block_[buffered_ + i] = data[i];
            buffered_ += take;
            data += take;
            size -= take;
            if (buffered_ == block_.size()) {
                transform(block_.data());
                buffered_ = 0U;
            }
        }
    }

    std::string finish_hex() {
        const std::uint64_t bits = total_bytes_ * 8U;
        block_[buffered_++] = 0x80U;
        if (buffered_ > 56U) {
            while (buffered_ < 64U) block_[buffered_++] = 0U;
            transform(block_.data());
            buffered_ = 0U;
        }
        while (buffered_ < 56U) block_[buffered_++] = 0U;
        for (int shift = 56; shift >= 0; shift -= 8) block_[buffered_++] = static_cast<std::uint8_t>(bits >> shift);
        transform(block_.data());
        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (std::uint32_t value : state_) out << std::setw(8) << value;
        return out.str();
    }

private:
    static constexpr std::array<std::uint32_t, 64> k_ = {
        0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
        0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
        0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
        0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
        0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
        0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
        0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
        0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

    static std::uint32_t rotr(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32U - n)); }
    void transform(const std::uint8_t* p) {
        std::uint32_t w[64]{};
        for (unsigned i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(p[4*i]) << 24U) |
                (static_cast<std::uint32_t>(p[4*i+1]) << 16U) |
                (static_cast<std::uint32_t>(p[4*i+2]) << 8U) | p[4*i+3];
        }
        for (unsigned i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3U);
            const std::uint32_t s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10U);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (unsigned i=0;i<64;++i) {
            const std::uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25);
            const std::uint32_t ch=(e&f)^((~e)&g);
            const std::uint32_t t1=h+s1+ch+k_[i]+w[i];
            const std::uint32_t s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            const std::uint32_t maj=(a&b)^(a&c)^(b&c);
            const std::uint32_t t2=s0+maj;
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
    std::array<std::uint32_t,8> state_={0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};
    std::array<std::uint8_t,64> block_{};
    std::size_t buffered_=0U;
    std::uint64_t total_bytes_=0U;
};

inline std::string file_hex(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot hash required file: " + path.string());
    Digest digest;
    std::array<std::uint8_t, 1U << 16U> buffer{};
    while (file) {
        file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size()));
        digest.update(buffer.data(), static_cast<std::size_t>(file.gcount()));
    }
    return digest.finish_hex();
}

} // namespace beam::sha256
