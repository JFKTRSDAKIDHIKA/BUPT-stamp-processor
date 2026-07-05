/// @file sync_manager.cpp

#include "common/sync_manager.h"

namespace mobol {

void SyncManager::release(SyncTag tag) {
    auto& cnt = counters_[tag];
    cnt++;
    events_.push_back({Event::RELEASE, tag, 0, cnt});
}

bool SyncManager::acquire(SyncTag tag, uint32_t arity) {
    uint32_t cnt = counters_[tag];
    events_.push_back({Event::ACQUIRE, tag, arity, cnt});

    if (cnt < arity) {
        // In sequential execution, this is a scheduling error.
        // In a real system, this would block until cnt reaches arity.
        return false;
    }

    events_.push_back({Event::ACQUIRE_PASS, tag, arity, cnt});
    return true;
}

void SyncManager::reset_counter(SyncTag tag) {
    counters_[tag] = 0;
    events_.push_back({Event::RESET, tag, 0, 0});
}

uint32_t SyncManager::get_count(SyncTag tag) const {
    auto it = counters_.find(tag);
    return (it != counters_.end()) ? it->second : 0;
}

void SyncManager::reset() {
    counters_.clear();
    events_.clear();
}

} // namespace mobol
