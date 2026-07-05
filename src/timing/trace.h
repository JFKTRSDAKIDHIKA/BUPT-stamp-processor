/// @file trace.h
/// @brief Structured trace events and summary report for the Timing Model.
#pragma once

#include "common/types.h"
#include <string>
#include <vector>
#include <map>

namespace mobol {

// ─── Trace Event Types ───────────────────────────────────────

enum class TraceEventType {
    // DMA events
    DMA_START,
    DMA_COMPLETE,
    // MXU events
    MXU_LAUNCH,
    MXU_COMPLETE,
    // Sync events
    RELEASE,
    ACQUIRE_BLOCK,
    ACQUIRE_PASS,
    // DRAM events
    DRAM_REQ_SENT,
    DRAM_REQ_DONE,
    // Tile lifecycle
    TILE_START,
    TILE_DONE,
};

struct TraceEvent {
    Cycle       timestamp;   ///< Logic cycle when event occurred
    TraceEventType type;
    TileId      tile_id;

    // Payload (varies by type)
    Addr        src_addr  = 0;
    Addr        dst_addr  = 0;
    size_t      size_bytes = 0;
    uint32_t    tag_index  = 0;
    uint32_t    arity      = 0;
    uint32_t    join_count = 0; ///< Counter value after event
    int         dram_latency = 0; ///< DRAM cycles (for DRAM events)
    std::string note;            ///< Free-form annotation
};

// ─── Summary Report ──────────────────────────────────────────

struct TimingSummary {
    Cycle total_cycles = 0;

    // Per-tile stats
    struct TileStats {
        Cycle busy_cycles  = 0;
        Cycle idle_cycles  = 0;
        Cycle stall_cycles = 0; ///< Blocked on acquire or resource
    };
    std::map<TileId, TileStats> tile_stats;

    // Resource utilization
    double mxu_utilization   = 0.0; ///< % of cycles MXU was in use (across all tiles)
    double dma_utilization   = 0.0; ///< % of cycles DMA engines were busy
    double noc_utilization   = 0.0; ///< % of max NoC bandwidth used

    // DRAM stats
    int    dram_read_requests  = 0;
    int    dram_write_requests = 0;
    size_t dram_read_bytes     = 0;
    size_t dram_write_bytes    = 0;

    // Deadlock check
    bool   all_acquires_passed = true;
    bool   all_dmas_completed  = true;
    int    pending_acquires    = 0;
    int    pending_dmas        = 0;

    // Causality check
    bool   causality_ok        = true;
    int    causality_violations = 0;
};

// ─── Trace output ────────────────────────────────────────────

/// Write trace events to a structured text file (grep-able).
void write_trace(const std::vector<TraceEvent>& events, const std::string& path);

/// Print summary report to stdout.
void print_summary(const TimingSummary& summary);

} // namespace mobol
