#include "stream3.hpp"

#include "hash.hpp"

#include <algorithm>

namespace beam {

std::uint64_t pack_stream3_val(std::uint32_t score_key, std::uint32_t payload_id) {
    return (static_cast<std::uint64_t>(score_key) << 32) | static_cast<std::uint64_t>(payload_id);
}

Stream3SplitResult stream3_threshold_dedup_split(
    const std::vector<Stream3CandidateInput>& input,
    std::uint32_t current_threshold,
    std::uint16_t local_rank,
    std::uint32_t world_size) {
    struct Packed {
        Hash128 hash;
        std::uint64_t val;
        std::uint64_t parent_idx;
        std::uint8_t move;
    };

    std::vector<Packed> compacted;
    compacted.reserve(input.size());
    for (const auto& item : input) {
        if (item.score_key <= current_threshold) {
            compacted.push_back(Packed{item.hash, pack_stream3_val(item.score_key, item.payload_id), item.parent_idx, item.move});
        }
    }

    std::sort(compacted.begin(), compacted.end(), [](const Packed& a, const Packed& b) {
        if (!hash_less(a.hash, b.hash) && !(a.hash == b.hash)) {
            return false;
        }
        if (!(a.hash == b.hash)) {
            return true;
        }
        return a.val < b.val;
    });

    std::vector<CandidateMeta> unique;
    for (const auto& item : compacted) {
        if (unique.empty() || !(unique.back().hash == item.hash)) {
            const std::uint32_t score_key = static_cast<std::uint32_t>(item.val >> 32);
            const std::uint8_t owner = owner_from_hash128(item.hash, world_size);
            unique.push_back(CandidateMeta{item.hash, item.parent_idx, score_key, pack_route(local_rank, owner, item.move)});
        }
    }

    Stream3SplitResult result;
    result.send_count.assign(world_size, 0);
    result.send_offset.assign(static_cast<std::size_t>(world_size) + 1, 0);
    for (const auto& meta : unique) {
        const auto owner = unpack_owner(meta.route_packed);
        if (owner == local_rank) {
            result.local_pending.push_back(meta);
        } else {
            ++result.send_count[owner];
        }
    }
    for (std::uint32_t peer = 0; peer < world_size; ++peer) {
        result.send_offset[peer + 1] = result.send_offset[peer] + result.send_count[peer];
    }
    result.remote_send.resize(result.send_offset[world_size]);
    std::vector<std::uint32_t> cursor = result.send_offset;
    for (const auto& meta : unique) {
        const auto owner = unpack_owner(meta.route_packed);
        if (owner != local_rank) {
            result.remote_send[cursor[owner]++] = meta;
        }
    }
    return result;
}

} // namespace beam

