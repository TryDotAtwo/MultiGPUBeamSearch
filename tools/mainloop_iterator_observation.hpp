#pragma once
#include <cutlass/transform/threadblock/predicated_tile_access_iterator_params.h>
#include <type_traits>

namespace beam::score_mode {
using MainloopBase=cutlass::transform::threadblock::PredicatedTileAccessIteratorParams;
template<class T,class=void>struct HasUnderlyingIterator:std::false_type{};
template<class T>struct HasUnderlyingIterator<T,std::void_t<typename T::UnderlyingIterator>>:std::true_type{};
template<class T,class=void>struct HasTileAccessIterator:std::false_type{};
template<class T>struct HasTileAccessIterator<T,std::void_t<typename T::TileAccessIterator>>:std::true_type{};

// Pinned classic CUTLASS wrappers each have one first member, Params params_.
// Standard-layout first-member pointer-interconvertibility is transitive down
// this explicitly checked chain. No byte offset, private macro or padding read.
template<class Iterator>constexpr void check_mainloop_layout() {
    using P=typename Iterator::Params;
    static_assert(std::is_standard_layout_v<P> && std::is_trivially_copyable_v<P>);
    static_assert(sizeof(P)==sizeof(MainloopBase) && alignof(P)==alignof(MainloopBase));
    if constexpr(HasUnderlyingIterator<Iterator>::value)
        check_mainloop_layout<typename Iterator::UnderlyingIterator>();
    else if constexpr(HasTileAccessIterator<Iterator>::value)
        check_mainloop_layout<typename Iterator::TileAccessIterator>();
    else {
        static_assert(std::is_same_v<typename P::Base,MainloopBase>);
        static_assert(std::is_base_of_v<MainloopBase,P>);
    }
}
template<class Iterator>const MainloopBase& observe_mainloop_base(const typename Iterator::Params& params) {
    check_mainloop_layout<Iterator>();
    return *reinterpret_cast<const MainloopBase*>(&params);
}
}
