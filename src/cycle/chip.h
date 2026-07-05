/// @file chip.h
/// @brief Top-level cycle-accurate MOBOL chip model.
///
/// Composes: 16 TileCores (base die) + RingNoC + 4 BankCtrls (buffer die)
/// + vertical hybrid bonds + DramCtrl/Ramulator2 (DRAM die).
///
/// One call to tick() advances exactly one logic cycle everywhere; the
/// DRAM domain advances `dram_ticks_per_logic` DRAM cycles inside it.
/// Execution-driven: instructions move real bytes and compute real values,
/// so functional results and timing come from a single simulation.
#pragma once

#include "cycle/tile_core.h"
#include "cycle/dram_ctrl.h"
#include "cycle/fabric.h"

#include <memory>
#include <string>
#include <vector>

namespace mobol::cycle {

struct ChipSummary {
    Cycle total_cycles = 0;
    TileStats tiles[NUM_TILES];

    double mxu_utilization = 0.0;      ///< busy tile-MXU cycles / (tiles*cycles)
    double avg_noc_link_utilization = 0.0;
    double max_noc_link_utilization = 0.0;
    uint64_t noc_flit_hops = 0;
    uint64_t noc_eject_blocked = 0;
    uint64_t vbond_tile_bank_flits = 0;
    uint64_t vbond_bank_dram_flits = 0;

    uint64_t shared_accesses[NUM_BANKS] = {};
    uint64_t shared_contention[NUM_BANKS] = {};
    uint64_t nmc_ops[NUM_BANKS] = {};
    Cycle nmc_busy[NUM_BANKS] = {};

    uint64_t dram_reads = 0, dram_writes = 0;
    uint64_t dram_read_bytes = 0, dram_write_bytes = 0;
    double dram_avg_read_latency = 0.0;   ///< DRAM cycles
    bool dram_is_ramulator = false;

    bool causality_ok = true;
    int causality_violations = 0;
    uint64_t total_instrs = 0;
};

class Chip {
public:
    explicit Chip(const CycleConfig& cfg);

    MemoryModel& mem() { return mem_; }
    EventLog& events() { return events_; }

    /// Run `prog` to completion (all tiles halted, fabric drained,
    /// DRAM quiet). Throws on trap or deadlock.
    ChipSummary run(const Program& prog);

    /// Print a human-readable report of the last run.
    void print_summary(const ChipSummary& s) const;

private:
    CycleConfig cfg_;
    MemoryModel mem_;
    Fabric fab_;
    RingNoC noc_;
    std::vector<BankCtrl> banks_;
    DramCtrl dram_;
    std::vector<TileCore> tiles_;
    EventLog events_;

    bool all_done() const;
    bool fabric_empty() const;
    void verify_causality(ChipSummary& s) const;
    [[noreturn]] void deadlock_dump(Cycle now) const;
};

} // namespace mobol::cycle
