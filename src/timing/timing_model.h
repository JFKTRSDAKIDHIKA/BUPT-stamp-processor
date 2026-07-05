/// @file timing_model.h
/// @brief Cycle-level Timing Model for MOBOL architecture.
///
/// Replays the same tile schedule as the Functional Engine, assigning
/// cycle timestamps based on resource availability and dependencies.
/// Integrates with Ramulator2 for DRAM latency.
///
/// Two-pass design:
///   Pass 1: Functional Engine (numerics, no timing)
///   Pass 2: Timing Model (cycles, no numerics modification)
/// Toggle: timing=off skips Pass 2, results must be bit-exact either way.
#pragma once

#include "common/types.h"
#include "common/address.h"
#include "timing/trace.h"

#include <string>
#include <vector>
#include <map>
#include <functional>
#include <memory>

// Forward declare Ramulator2 types to avoid including in header
namespace Ramulator { class IMemorySystem; class IFrontEnd; }

namespace mobol {

// ─── Timing Configuration ────────────────────────────────────

struct TimingConfig {
    // MXU pipeline (approved Q1)
    int mxu_latency    = MXU_LATENCY;     // 4 cycles
    int mxu_throughput = MXU_THROUGHPUT;  // 1 tile/cycle

    // SRAM ports (approved Q2 with conditions)
    int local_rd_ports = 2;
    int local_wr_ports = 1;
    bool shared_port_contention = false;  // Toggle switch (Q2 condition)
    int shared_contention_count = 0;      // Summary counter (Q2 condition)

    // NoC (approved Q3)
    int noc_hop_latency = NOC_HOP_LATENCY; // 1 cycle
    int noc_link_bw     = NOC_LINK_BW;     // 64 bytes/cycle/link

    // DMA (approved Q4)
    int dma_setup_cycles = DMA_SETUP_CYCLES; // 2 cycles

    // Clock domain (approved Q5)
    int logic_clock_ratio = 1;
    int dram_clock_ratio  = 3;

    // Max simulation cycles (deadlock guard)
    Cycle max_cycles = 1000000;
};

// ─── Schedule Operation (mirrors functional engine operations) ──

enum class SchedOp {
    // DMA from DRAM to LOCAL (strided matrix block copy)
    DMA_DRAM_TO_LOCAL,
    // DMA from LOCAL to DRAM
    DMA_LOCAL_TO_DRAM,
    // DMA from SHARED to LOCAL
    DMA_SHARED_TO_LOCAL,
    // DMA from LOCAL to SHARED
    DMA_LOCAL_TO_SHARED,
    // DMA from one tile's LOCAL to another tile's LOCAL
    DMA_LOCAL_TO_LOCAL,
    // MXU operations
    MXU_F16xF16,    // f16×f16 → f32 (GEMM1)
    MXU_F32xF16,    // f32×f16 → f32 (GEMM2)
    // Reduce
    REDUCE_F32,
    // Convert f32 → f16 (treated as local compute, small latency)
    CONVERT_F32_F16,
    // Sync
    RELEASE_SYNC,
    ACQUIRE_SYNC,
};

struct ScheduleEntry {
    SchedOp op;
    TileId  tile_id;

    // DMA fields
    Addr    src_addr  = 0;
    Addr    dst_addr  = 0;
    size_t  size_bytes = 0;
    // For strided matrix block DMA:
    bool    is_strided = false;
    Addr    base_addr  = 0;
    int     block_row  = 0;
    int     block_col  = 0;
    int     matrix_dim = 32;
    size_t  elem_size  = 2;

    // Sync fields
    uint32_t tag_index = 0;
    uint32_t arity     = 0;

    // Annotation
    std::string note;
};

// ─── Timing Simulator ────────────────────────────────────────

class TimingSimulator {
public:
    explicit TimingSimulator(const TimingConfig& config = TimingConfig{});
    ~TimingSimulator();

    /// Add a schedule entry for a tile.
    void add_operation(const ScheduleEntry& entry);

    /// Run the timing simulation.
    /// Returns the summary report.
    TimingSummary run(const std::string& trace_path = "");

    /// Get the trace events.
    const std::vector<TraceEvent>& get_trace() const { return trace_; }

private:
    TimingConfig config_;

    // Schedule per tile
    struct TileSchedule {
        std::vector<ScheduleEntry> ops;
        size_t next_op = 0;       ///< Index of next operation to process
        Cycle available_at = 0;   ///< Cycle when tile can start next op
        bool done = false;
    };
    std::map<TileId, TileSchedule> tile_schedules_;

    // Trace and stats
    std::vector<TraceEvent> trace_;
    TimingSummary summary_;

    // Sync state (join counters for timing)
    std::map<std::pair<TileId, uint32_t>, uint32_t> join_counters_;

    // DRAM integration
    Ramulator::IMemorySystem* mem_sys_ = nullptr;
    bool init_ramulator();
    void cleanup_ramulator();

    // Simulate a single operation, return duration in logic cycles
    Cycle simulate_op(const ScheduleEntry& op, Cycle start_cycle);

    // Compute DMA transfer cycles
    Cycle dma_cycles(Addr src, Addr dst, size_t bytes);

    // Compute NoC hops between two tiles
    int noc_hops(TileId src, TileId dst) const;

    // Record a trace event
    void emit(TraceEventType type, TileId tile, Cycle ts, const ScheduleEntry& op = {});

    // Causality verification
    void verify_causality();
};

} // namespace mobol
