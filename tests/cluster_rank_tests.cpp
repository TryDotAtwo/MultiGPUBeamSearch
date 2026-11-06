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
        require(rank.cuda_ordinal(8) == 7);
        require(rank.cuda_ordinal(1) == 0);  // srun --gpus-per-task=1
        require(rank.node_share(8000) == 1000);
        require(rank.default_host_headroom() == (10ULL << 30));
        rejects([&] { rank.cuda_ordinal(0); });
        rejects([&] { rank.cuda_ordinal(4); });
        const Rank last(128, 127, 8, 7);
        require(last.cuda_ordinal(8) == 7);
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
