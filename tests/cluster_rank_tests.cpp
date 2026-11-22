#include "../src/distributed_rank.hpp"
#include <iostream>
#include <stdexcept>

namespace {
void require(bool success) {
    if (!success) throw std::runtime_error("rank geometry mismatch");
}

template <class F> void rejects(F operation) {
    try { operation(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid rank geometry accepted");
}
}

int main() {
    using beam::distributed::Rank;
    try {
        const Rank rank(16, 15, 8, 7);
        // Standalone score validation can also select a CUDA ordinal without
        // creating any communicator or requiring a scheduler world.
        require(beam::distributed::cuda_ordinal(7, 1) == 0);
        require(beam::distributed::cuda_ordinal(3, 4) == 3);
        require(beam::distributed::cuda_ordinal(rank.local_rank, 8) == 7);
        require(rank.node_share(8000) == 1000);
        require(rank.default_host_headroom() == (10ULL << 30));
        rejects([&] { beam::distributed::cuda_ordinal(rank.local_rank, 0); });
        rejects([&] { beam::distributed::cuda_ordinal(rank.local_rank, 4); });
        const Rank last(128, 127, 8, 7);
        require(beam::distributed::cuda_ordinal(last.local_rank, 8) == 7);
        rejects([] { Rank(129, 0, 8, 0); });
        rejects([] { Rank(16, 16, 8, 0); });
        rejects([] { Rank(16, 0, 17, 0); });
        rejects([] { Rank(16, 0, 8, 8); });
        rejects([] { Rank(0, 0, 0, 0); });
        std::cout << "cluster_rank_tests=pass\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
