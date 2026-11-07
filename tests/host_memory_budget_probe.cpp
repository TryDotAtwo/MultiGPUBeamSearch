#include "tools/host_memory_budget.hpp"
#include <iostream>
int main(int argc, char** argv) {
    if (argc != 4) return 2;
    try {
        std::cout << beam::host_memory::available_bytes(argv[1], argv[2], argv[3]) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
