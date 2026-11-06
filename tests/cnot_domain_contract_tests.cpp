#include "../src/cnot_domain.hpp"

#include <iostream>
#include <stdexcept>

int main() {
    namespace cnot = beam::cnot;
    std::uint64_t verified = 0;
    for (int n = 2; n <= cnot::max_qubits; ++n) {
        int expected = 0;
        for (int control = 0; control < n; ++control) {
            for (int target = 0; target < n; ++target) {
                if (control == target) continue;
                const auto move = cnot::encode_move(n, control, target);
                const auto gate = cnot::decode_move(n, move);
                if (move != expected++ || gate.control != control || gate.target != target) {
                    throw std::runtime_error("physical CNOT alphabet is not an ordered bijection");
                }
                ++verified;
            }
        }
        if (expected != cnot::move_count(n)) throw std::runtime_error("incorrect alphabet size");
    }

    const auto route = cnot::pack_route(65535, cnot::encode_move(144, 143, 142));
    beam::CandidateMeta metadata{};
    metadata.parent_idx = 0x100000007ULL;
    metadata.route_packed = route;
    const cnot::FinalRequest request{metadata.parent_idx, 0x80000001U,
                                     cnot::route_source(route), cnot::route_move(route)};
    if (request.move != 20591 || request.return_rank != 65535 || request.parent_idx != 0x100000007ULL
        || request.target_local_idx != 0x80000001U) throw std::runtime_error("CNOT request lost bits");
    if (cnot::tail_mask(144) != 0xffff || cnot::tail_mask(64) != ~std::uint64_t{0}) {
        throw std::runtime_error("incorrect packed row mask");
    }
    std::cout << "verified " << verified << " physical gates; CandidateMeta=" << sizeof(metadata)
              << " FinalRequest=" << sizeof(request) << " move=" << request.move << '\n';
}
