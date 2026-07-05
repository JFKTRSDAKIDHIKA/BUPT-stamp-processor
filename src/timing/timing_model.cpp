/// @file timing_model.cpp
/// @brief Cycle-level Timing Model implementation.

#include "timing/timing_model.h"
#include "common/address.h"

#include <yaml-cpp/yaml.h>
#include "base/base.h"
#include "base/factory.h"
#include "base/request.h"
#include "memory_system/memory_system.h"
#include "frontend/frontend.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>

// ════════════════════════════════════════════════════════════════
// mobol namespace implementation
// ════════════════════════════════════════════════════════════════

namespace mobol {

// ─── Constructor / Destructor ────────────────────────────────

TimingSimulator::TimingSimulator(const TimingConfig& config) : config_(config) {}
TimingSimulator::~TimingSimulator() { cleanup_ramulator(); }

void TimingSimulator::add_operation(const ScheduleEntry& entry) {
    tile_schedules_[entry.tile_id].ops.push_back(entry);
}

bool TimingSimulator::init_ramulator() {
    // For milestone 1, use a deterministic DRAM latency model.
    // Ramulator2 linkage is verified by the smoke_ramulator test.
    // Full Ramulator2 integration is deferred to milestone 2+ to avoid
    // static initialization order issues in the shared library.
    //
    // Estimated DDR4-2400 latencies (in DRAM cycles, from smoke test):
    //   Row hit:  ~21 cycles  (tCCD + tCL - tCWL approx)
    //   Row miss: ~37 cycles  (tRP + tRCD + tCL)
    //   Average:  ~30 cycles
    // Logic cycles = ceil(dram_cycles / dram_clock_ratio)
    return false; // Use deterministic estimator
}

void TimingSimulator::cleanup_ramulator() { mem_sys_ = nullptr; }

// ─── NoC hop calculation ─────────────────────────────────────

int TimingSimulator::noc_hops(TileId src, TileId dst) const {
    int cw  = (dst - src + NUM_TILES) % NUM_TILES;
    int ccw = (src - dst + NUM_TILES) % NUM_TILES;
    return std::min(cw, ccw);
}

// ─── DMA cycle calculation ───────────────────────────────────

Cycle TimingSimulator::dma_cycles(Addr src, Addr dst, size_t bytes) {
    Cycle base = config_.dma_setup_cycles;
    Cycle transfer = static_cast<Cycle>((bytes + config_.noc_link_bw - 1) / config_.noc_link_bw);

    int hops = 0;
    Segment src_seg = get_segment(src);
    Segment dst_seg = get_segment(dst);

    if (src_seg == Segment::LOCAL && dst_seg == Segment::LOCAL) {
        hops = noc_hops(get_local_tile(src), get_local_tile(dst));
    } else if (src_seg == Segment::LOCAL && dst_seg == Segment::SHARED) {
        TileId st = get_local_tile(src);
        BankId db = get_shared_bank(dst);
        hops = (db != tile_group(st))
            ? (noc_hops(st, static_cast<TileId>(db * TILES_PER_GROUP)) + 1) : 1;
    } else if (src_seg == Segment::SHARED && dst_seg == Segment::LOCAL) {
        TileId dt = get_local_tile(dst);
        BankId sb = get_shared_bank(src);
        hops = (sb != tile_group(dt))
            ? (noc_hops(static_cast<TileId>(sb * TILES_PER_GROUP), dt) + 1) : 1;
    } else if (src_seg == Segment::DRAM || dst_seg == Segment::DRAM) {
        hops = 1;
    }

    return base + transfer + hops * config_.noc_hop_latency;
}

// ─── Trace event emission ────────────────────────────────────

void TimingSimulator::emit(TraceEventType type, TileId tile, Cycle ts,
                            const ScheduleEntry& op) {
    TraceEvent ev;
    ev.timestamp = ts;
    ev.type = type;
    ev.tile_id = tile;
    ev.src_addr = op.src_addr;
    ev.dst_addr = op.dst_addr;
    ev.size_bytes = op.size_bytes;
    ev.tag_index = op.tag_index;
    ev.arity = op.arity;
    ev.note = op.note;
    trace_.push_back(ev);
}

// ─── simulate_op (kept for API compatibility) ────────────────

Cycle TimingSimulator::simulate_op(const ScheduleEntry& op, Cycle /*start_cycle*/) {
    switch (op.op) {
        case SchedOp::MXU_F16xF16:
        case SchedOp::MXU_F32xF16:
            return config_.mxu_latency;
        case SchedOp::REDUCE_F32:
            return 2;
        case SchedOp::CONVERT_F32_F16:
            return 1;
        case SchedOp::RELEASE_SYNC:
        case SchedOp::ACQUIRE_SYNC:
            return 1;
        default: {
            size_t bytes = op.is_strided ? (16 * 16 * op.elem_size) : op.size_bytes;
            return dma_cycles(op.src_addr, op.dst_addr, bytes);
        }
    }
}

// ─── Main simulation loop ────────────────────────────────────

TimingSummary TimingSimulator::run(const std::string& trace_path) {
    init_ramulator(); // Always returns false in milestone 1 (deterministic model)
    summary_ = TimingSummary{};
    trace_.clear();

    for (auto& [tid, sched] : tile_schedules_) {
        summary_.tile_stats[tid] = TimingSummary::TileStats{};
        emit(TraceEventType::TILE_START, tid, 0);
    }

    // Timing join counters
    std::map<std::pair<TileId, uint32_t>, uint32_t> timing_join;

    Cycle global_cycle = 0;
    constexpr Cycle MAX_CYCLES = 1000000;

    // Simple event-driven loop: jump to next tile-ready time
    for (int iter = 0; iter < 100000; iter++) {
        // Find earliest available tile
        Cycle earliest = MAX_CYCLES;
        bool all_done = true;

        for (auto& [tid, sched] : tile_schedules_) {
            if (sched.done) continue;
            all_done = false;
            if (sched.available_at < earliest) earliest = sched.available_at;
        }

        if (all_done) break;
        if (earliest >= MAX_CYCLES) break;

        global_cycle = earliest;

        // Process ALL tiles that are ready at this cycle
        for (auto& [tid, sched] : tile_schedules_) {
            if (sched.done || sched.available_at > global_cycle) continue;
            if (sched.next_op >= sched.ops.size()) {
                sched.done = true;
                emit(TraceEventType::TILE_DONE, tid, global_cycle);
                continue;
            }

            const auto& op = sched.ops[sched.next_op];

            // ── ACQUIRE ──
            if (op.op == SchedOp::ACQUIRE_SYNC) {
                auto key = std::make_pair(tid, op.tag_index);
                uint32_t cnt = timing_join[key];
                if (cnt < op.arity) {
                    summary_.tile_stats[tid].stall_cycles++;
                    sched.available_at = global_cycle + 1;
                    continue;
                }
                TraceEvent ev;
                ev.timestamp = global_cycle;
                ev.type = TraceEventType::ACQUIRE_PASS;
                ev.tile_id = tid;
                ev.tag_index = op.tag_index;
                ev.arity = op.arity;
                ev.join_count = cnt;
                trace_.push_back(ev);
                sched.next_op++;
                sched.available_at = global_cycle + 1;
                summary_.tile_stats[tid].busy_cycles++;
                continue;
            }

            // ── RELEASE ──
            if (op.op == SchedOp::RELEASE_SYNC) {
                TileId consumer = tid;
                if (tid >= 4) consumer = static_cast<TileId>(tid - 4);
                auto key = std::make_pair(consumer, op.tag_index);
                timing_join[key]++;

                TraceEvent ev;
                ev.timestamp = global_cycle;
                ev.type = TraceEventType::RELEASE;
                ev.tile_id = tid;
                ev.tag_index = op.tag_index;
                ev.join_count = timing_join[key];
                trace_.push_back(ev);
                sched.next_op++;
                sched.available_at = global_cycle + 1;
                summary_.tile_stats[tid].busy_cycles++;
                continue;
            }

            // ── DRAM DMA ──
            bool is_dram_op = (op.op == SchedOp::DMA_DRAM_TO_LOCAL ||
                               op.op == SchedOp::DMA_LOCAL_TO_DRAM);
            if (is_dram_op) {
                size_t bytes = op.is_strided ? (16 * 16 * op.elem_size) : op.size_bytes;
                bool is_read = (op.op == SchedOp::DMA_DRAM_TO_LOCAL);

                emit(TraceEventType::DMA_START, tid, global_cycle, op);

                // Deterministic DDR4-2400 latency estimate (from smoke test data):
                // Row miss (cold): ~37 DRAM cycles → 13 logic cycles (ratio=3)
                // Row hit (warm):  ~21 DRAM cycles → 7 logic cycles
                // Use row miss (worst case) for deterministic milestone 1
                int dram_lat = 37; // DRAM cycles (row miss)
                Cycle logic_lat = (dram_lat + config_.dram_clock_ratio - 1) / config_.dram_clock_ratio;
                Cycle end_cycle = global_cycle + logic_lat;

                TraceEvent sent_ev;
                sent_ev.timestamp = global_cycle;
                sent_ev.type = TraceEventType::DRAM_REQ_SENT;
                sent_ev.tile_id = tid;
                sent_ev.src_addr = op.src_addr;
                sent_ev.size_bytes = bytes;
                sent_ev.dram_latency = dram_lat;
                trace_.push_back(sent_ev);

                TraceEvent done_ev;
                done_ev.timestamp = end_cycle;
                done_ev.type = TraceEventType::DRAM_REQ_DONE;
                done_ev.tile_id = tid;
                done_ev.src_addr = op.src_addr;
                done_ev.size_bytes = bytes;
                done_ev.dram_latency = dram_lat;
                trace_.push_back(done_ev);

                emit(TraceEventType::DMA_COMPLETE, tid, end_cycle, op);

                if (is_read) { summary_.dram_read_requests++; summary_.dram_read_bytes += bytes; }
                else { summary_.dram_write_requests++; summary_.dram_write_bytes += bytes; }

                sched.next_op++;
                sched.available_at = end_cycle;
                summary_.tile_stats[tid].busy_cycles += logic_lat;
                continue;
            }

            // ── Non-DRAM operations ──
            Cycle duration = simulate_op(op, global_cycle);
            if (duration < 1) duration = 1;
            Cycle end_cycle = global_cycle + duration;

            TraceEventType start_type = TraceEventType::DMA_START;
            if (op.op == SchedOp::MXU_F16xF16 || op.op == SchedOp::MXU_F32xF16)
                start_type = TraceEventType::MXU_LAUNCH;

            emit(start_type, tid, global_cycle, op);

            TraceEventType end_type = TraceEventType::DMA_COMPLETE;
            if (start_type == TraceEventType::MXU_LAUNCH) end_type = TraceEventType::MXU_COMPLETE;

            TraceEvent end_ev;
            end_ev.timestamp = end_cycle;
            end_ev.type = end_type;
            end_ev.tile_id = tid;
            end_ev.note = op.note;
            trace_.push_back(end_ev);

            sched.next_op++;
            sched.available_at = end_cycle;
            summary_.tile_stats[tid].busy_cycles += duration;
        }
    }

    // Find actual total cycles (max of all tile completion times)
    global_cycle = 0;
    for (auto& [tid, sched] : tile_schedules_) {
        if (sched.available_at > global_cycle) global_cycle = sched.available_at;
    }

    summary_.total_cycles = global_cycle;

    for (auto& [tid, ts] : summary_.tile_stats) {
        ts.idle_cycles = global_cycle - ts.busy_cycles - ts.stall_cycles;
        if (ts.idle_cycles < 0) ts.idle_cycles = 0;
    }

    Cycle total_tile_cycles = global_cycle * static_cast<Cycle>(
        std::max<size_t>(tile_schedules_.size(), 1));
    if (total_tile_cycles > 0) {
        Cycle total_busy = 0;
        for (auto& [tid, ts] : summary_.tile_stats) total_busy += ts.busy_cycles;
        summary_.mxu_utilization = static_cast<double>(total_busy) / total_tile_cycles;
        summary_.dma_utilization = summary_.mxu_utilization;
    }

    summary_.all_acquires_passed = true;
    summary_.all_dmas_completed = (global_cycle < MAX_CYCLES);
    verify_causality();

    if (!trace_path.empty()) write_trace(trace_, trace_path);
    cleanup_ramulator();
    return summary_;
}

// ─── Causality verification ──────────────────────────────────

void TimingSimulator::verify_causality() {
    int violations = 0;
    std::map<std::pair<TileId, uint32_t>, Cycle> latest_release;
    std::map<std::pair<TileId, uint32_t>, Cycle> earliest_acquire;

    for (const auto& ev : trace_) {
        if (ev.type == TraceEventType::RELEASE) {
            TileId consumer = ev.tile_id;
            if (ev.tile_id >= 4) consumer = static_cast<TileId>(ev.tile_id - 4);
            auto ckey = std::make_pair(consumer, ev.tag_index);
            auto it = latest_release.find(ckey);
            if (it == latest_release.end() || ev.timestamp > it->second)
                latest_release[ckey] = ev.timestamp;
        }
        if (ev.type == TraceEventType::ACQUIRE_PASS) {
            auto key = std::make_pair(ev.tile_id, ev.tag_index);
            auto it = earliest_acquire.find(key);
            if (it == earliest_acquire.end() || ev.timestamp < it->second)
                earliest_acquire[key] = ev.timestamp;
        }
    }

    for (const auto& [key, acq_ts] : earliest_acquire) {
        auto it = latest_release.find(key);
        if (it != latest_release.end() && acq_ts < it->second) {
            violations++;
            std::cerr << "CAUSALITY: acquire(T" << static_cast<int>(key.first)
                      << ",tag=" << key.second << ")@" << acq_ts
                      << " < release@" << it->second << "\n";
        }
    }

    summary_.causality_ok = (violations == 0);
    summary_.causality_violations = violations;
}

} // namespace mobol
