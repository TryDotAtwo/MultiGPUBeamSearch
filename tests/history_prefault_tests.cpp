#include "tools/host_memory_budget.hpp"
#include <array>
#include <iostream>
#include <stdexcept>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#endif
int main() {
    try {
        std::array<unsigned char, 8200> bytes;
        bytes.fill(0xaa);
        beam::host_memory::prefault_pages(bytes.data() + 1, 8193);
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            const bool touched = index == 1 || index == 4097 || index == 8193;
            if (bytes[index] != (touched ? 0 : 0xaa))
                throw std::runtime_error("prefault crossed buffer or missed page");
        }
        bytes.fill(0xaa);
        beam::host_memory::prefault_pages(bytes.data() + 1, 17);
        for (std::size_t index = 0; index < bytes.size(); ++index) {
            const bool touched = index == 1 || index == 17;
            if (bytes[index] != (touched ? 0 : 0xaa))
                throw std::runtime_error("partial-page prefault crossed buffer");
        }
        beam::host_memory::prefault_pages(nullptr, 0);
        bool rejected = false;
        try { beam::host_memory::prefault_pages(nullptr, 1); }
        catch (const std::invalid_argument&) { rejected = true; }
        if (!rejected) throw std::runtime_error("nonempty null prefault accepted");
#if defined(__linux__)
        const auto page = sysconf(_SC_PAGESIZE);
        if (page < 4096) throw std::runtime_error("prefault requires host pages >=4KiB");
        const std::size_t length = static_cast<std::size_t>(page) * 16;
        void* mapping = mmap(nullptr, length, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (mapping == MAP_FAILED) throw std::runtime_error("mmap fixture failed");
        std::vector<unsigned char> residency(16);
        if (mincore(mapping, length, residency.data()) != 0)
            throw std::runtime_error("initial residency query failed");
        unsigned initial = 0;
        for (const auto value : residency) initial += value & 1;
        beam::host_memory::prefault_pages(mapping, length);
        if (mincore(mapping, length, residency.data()) != 0)
            throw std::runtime_error("final residency query failed");
        for (const auto value : residency)
            if (!(value & 1)) throw std::runtime_error("prefault left nonresident anonymous page");
        if (munmap(mapping, length) != 0) throw std::runtime_error("fixture unmap failed");
        std::cout << "resident_before=" << initial << " resident_after=16\n";
#endif
        std::cout << "history_prefault_tests=pass\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
