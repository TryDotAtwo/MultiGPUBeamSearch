#pragma once
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>

namespace beam {
// Own policy strings: getenv pointers must never outlive capture.
class Stream1PolicySnapshot {
public:
    static Stream1PolicySnapshot capture(std::initializer_list<const char*> keys) {
        Stream1PolicySnapshot result;
        for (const auto* key : keys) {
            if (!key || !*key) throw std::invalid_argument("empty Stream1 policy key");
            const auto* value = std::getenv(key);
            result.values_.emplace(key, value ? std::optional<std::string>(value) : std::nullopt);
        }
        return result;
    }
    const char* get(const char* key) const {
        if (!key) throw std::invalid_argument("null Stream1 policy key");
        const auto entry = values_.find(key);
        if (entry == values_.end()) throw std::invalid_argument("unbound Stream1 policy key");
        return entry->second ? entry->second->c_str() : nullptr;
    }
    const auto& values() const noexcept { return values_; }
private:
    std::map<std::string, std::optional<std::string>, std::less<>> values_;
};
inline thread_local const Stream1PolicySnapshot* active_stream1_policy_snapshot = nullptr;
class Stream1PolicyScope {
public:
    explicit Stream1PolicyScope(const Stream1PolicySnapshot& snapshot) noexcept
        : previous_(active_stream1_policy_snapshot) {
        active_stream1_policy_snapshot = &snapshot;
    }
    Stream1PolicyScope(Stream1PolicySnapshot&&) = delete;
    Stream1PolicyScope(const Stream1PolicyScope&) = delete;
    Stream1PolicyScope& operator=(const Stream1PolicyScope&) = delete;
    ~Stream1PolicyScope() { active_stream1_policy_snapshot = previous_; }
private:
    const Stream1PolicySnapshot* previous_;
};
inline const char* stream1_policy_value(const char* key) {
    if (!key || !*key) throw std::invalid_argument("empty Stream1 policy key");
    return active_stream1_policy_snapshot ? active_stream1_policy_snapshot->get(key) : std::getenv(key);
}
} // namespace beam
