#include "unique_frontier_reference.hpp"
#include "stream4.hpp"
#include <iostream>
#include <map>
#include <random>
#include <stdexcept>
#include <limits>
using namespace beam;
using Key = std::pair<std::uint64_t, std::uint64_t>;
void require(bool condition) { if (!condition) throw std::runtime_error("unique frontier oracle mismatch"); }
int main() {
    try {
        CandidateMeta x{{1,0},0,1,0}, y{{2,0},1,2,0};
        auto merged = unique_union_reference({x,y}, {x});
        require(merged.size() == 2 && merged[0].hash.lo == 1 && merged[1].hash.lo == 2);
        std::uint64_t a[3]{0,1,1}, b[3]{0,1,0}, lower[3]{};
        conservative_union_histogram(a,b,3,lower);
        require(lower[0] == 0 && lower[1] == 1 && lower[2] == 1);
        // Old multiset count reaches K=2 at score1; safe lower reaches at score2.
        require(a[1] + b[1] == 2 && lower[1] < 2 && lower[1] + lower[2] == 2);
        std::uint64_t shifted_a[3]{1,0,1}, shifted_b[3]{0,1,0};
        conservative_union_histogram(shifted_a,shifted_b,3,lower);
        require(lower[0]==1 && lower[1]==0 && lower[2]==1);
        require(unique_union_reference({},{}).empty());
        std::mt19937 random(481);
        for (unsigned seed=0; seed<256; ++seed) {
            std::map<Key,CandidateMeta> physical[2], oracle;
            for (unsigned i=0; i<100; ++i) {
                CandidateMeta c{{random()%24,random()%4},i,random()%8,random()%16};
                auto& bank = physical[random()%2];
                Key key{c.hash.lo,c.hash.hi};
                auto found=bank.find(key);
                if (found==bank.end() || std::make_tuple(c.score_key,c.parent_idx,c.route_packed) <
                    std::make_tuple(found->second.score_key,found->second.parent_idx,found->second.route_packed)) bank[key]=c;
            }
            std::vector<CandidateMeta> inputs[2];
            std::uint64_t hist[2][8]{}, safe[8]{};
            for (unsigned bank=0; bank<2; ++bank) for (auto [key,c]:physical[bank]) {
                inputs[bank].push_back(c); ++hist[bank][c.score_key];
                auto found=oracle.find(key);
                if (found==oracle.end() || std::make_tuple(c.score_key,c.parent_idx,c.route_packed) <
                    std::make_tuple(found->second.score_key,found->second.parent_idx,found->second.route_packed)) oracle[key]=c;
            }
            conservative_union_histogram(hist[0],hist[1],8,safe);
            std::uint64_t cumulative=0;
            for (unsigned t=0;t<8;++t) {
                cumulative+=safe[t]; unsigned exact=0;
                for (auto [key,c]:oracle) exact+=c.score_key<=t;
                require(cumulative<=exact);
            }
            auto result=unique_union_reference(inputs[0],inputs[1]);
            require(result.size()==oracle.size());
            for (auto c:result) {
                auto expected=oracle.at({c.hash.lo,c.hash.hi});
                require(c.score_key==expected.score_key && c.parent_idx==expected.parent_idx && c.route_packed==expected.route_packed);
            }
            for (unsigned owners : {1U,2U,8U}) {
                std::uint64_t global_lower[8]{};
                for (unsigned owner=0;owner<owners;++owner) {
                    std::uint64_t owner_hist[2][8]{}, owner_lower[8]{};
                    for (unsigned bank=0;bank<2;++bank) for (auto c:inputs[bank])
                        if (c.hash.lo%owners==owner) ++owner_hist[bank][c.score_key];
                    conservative_union_histogram(owner_hist[0],owner_hist[1],8,owner_lower);
                    for (unsigned t=0;t<8;++t) global_lower[t]+=owner_lower[t];
                }
                cumulative=0;
                for (unsigned t=0;t<8;++t) {
                    cumulative+=global_lower[t]; unsigned exact=0;
                    for (auto [key,c]:oracle) exact+=c.score_key<=t;
                    require(cumulative<=exact);
                }
            }
        }
        std::uint64_t huge[2]{std::numeric_limits<std::uint64_t>::max(),1}, zero[2]{};
        bool rejected=false;
        try { conservative_union_histogram(huge,zero,2,zero); } catch (const std::overflow_error&) { rejected=true; }
        require(rejected);
        std::vector<std::uint64_t> global_hist(SCORE_BIN_COUNT,0);
        global_hist[0]=std::numeric_limits<std::uint64_t>::max()-1;
        global_hist[1]=2;
        rejected=false;
        try { histogram_threshold(global_hist,std::numeric_limits<std::uint64_t>::max()); }
        catch (const std::overflow_error&) { rejected=true; }
        require(rejected);
        std::cout << "unique_frontier_tests=pass seeds=256\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
