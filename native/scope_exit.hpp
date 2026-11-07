#pragma once
#include <utility>
namespace beam::native {
template<class F> class ScopeExit {
    F cleanup_;
public:
    explicit ScopeExit(F cleanup) : cleanup_(std::move(cleanup)) {}
    ScopeExit(const ScopeExit&) = delete;
    ScopeExit& operator=(const ScopeExit&) = delete;
    ~ScopeExit() noexcept { try { cleanup_(); } catch(...) {} }
};
template<class F> ScopeExit(F) -> ScopeExit<F>;
}
