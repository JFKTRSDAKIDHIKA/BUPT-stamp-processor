/// @file sync_manager.h
/// @brief Tag-based release/acquire with arity-k join counters.
///
/// Per MOBOL spec §3: synchronization is tag-named, not address-monitored.
/// Join counters are per-(consumer_tile, tag). Incremented on:
///   1. Local producer completion (if consumer == producer)
///   2. Remote release token arrival
/// acquire(tag, arity) blocks (functionally: asserts) until cnt == arity.
#pragma once

#include "common/types.h"
#include <unordered_map>
#include <utility>
#include <string>
#include <cassert>
#include <vector>

namespace mobol {

/// Unique tag identifier: (consumer_tile_id, tag_index).
/// tag_index is assigned by the compiler/schedule (sequential per consumer).
struct SyncTag {
    TileId consumer;
    uint32_t index;

    bool operator==(const SyncTag& o) const {
        return consumer == o.consumer && index == o.index;
    }
};

struct SyncTagHash {
    size_t operator()(const SyncTag& t) const {
        return std::hash<uint64_t>{}(
            (static_cast<uint64_t>(t.consumer) << 32) | t.index);
    }
};

class SyncManager {
public:
    /// Producer signals completion of data associated with `tag`.
    /// Increments the join counter on the consumer tile.
    void release(SyncTag tag);

    /// Consumer waits for `arity` producers.
    /// In functional (sequential) mode: asserts counter == arity.
    /// Returns true if the acquire succeeds.
    bool acquire(SyncTag tag, uint32_t arity);

    /// Reset a counter to 0 (after consumer finishes reading).
    void reset_counter(SyncTag tag);

    /// Query current counter value (for debugging/tracing).
    uint32_t get_count(SyncTag tag) const;

    /// Reset all counters.
    void reset();

    /// Get the log of release/acquire events (for trace output).
    struct Event {
        enum Type { RELEASE, ACQUIRE, ACQUIRE_PASS, RESET };
        Type type;
        SyncTag tag;
        uint32_t arity;
        uint32_t count_after;
    };
    const std::vector<Event>& get_events() const { return events_; }

private:
    std::unordered_map<SyncTag, uint32_t, SyncTagHash> counters_;
    std::vector<Event> events_;
};

} // namespace mobol
