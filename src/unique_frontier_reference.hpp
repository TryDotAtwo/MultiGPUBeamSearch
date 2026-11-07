#pragma once
#include "stream4.hpp"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace beam {
// Independent host oracle only; does not replace GPU-resident production union.
inline std::vector<CandidateMeta> unique_union_reference(
    const std::vector<CandidateMeta>& a, const std::vector<CandidateMeta>& b) {
    std::vector<CandidateMeta> combined=a;
    combined.insert(combined.end(),b.begin(),b.end());
    return stream4_threshold_sort_dedup(combined,std::numeric_limits<std::uint32_t>::max());
}

// Physical inputs must each be internally unique within one owner/shard domain.
// Pointwise max of bins is NOT safe: use max of cumulative distributions.
inline void conservative_union_histogram(const std::uint64_t* a, const std::uint64_t* b,
    std::uint32_t bins, std::uint64_t* output) {
    if (bins && (!a || !b || !output)) throw std::invalid_argument("null histogram");
    std::uint64_t ca=0,cb=0,previous=0;
    for (std::uint32_t i=0;i<bins;++i) {
        if (a[i]>std::numeric_limits<std::uint64_t>::max()-ca ||
            b[i]>std::numeric_limits<std::uint64_t>::max()-cb)
            throw std::overflow_error("histogram cumulative overflow");
        ca+=a[i]; cb+=b[i];
        const auto lower=std::max(ca,cb);
        output[i]=lower-previous; previous=lower;
    }
}
} // namespace beam
