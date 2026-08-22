#include "../tools/sha256.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>

int main() {
    const std::filesystem::path path = std::filesystem::path("test_results") / "sha256_abc.bin";
    std::filesystem::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary | std::ios::trunc) << "abc";
    if (beam::sha256::file_hex(path) !=
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") {
        return EXIT_FAILURE;
    }
    std::cout << "sha256_tests=pass\n";
    return EXIT_SUCCESS;
}
