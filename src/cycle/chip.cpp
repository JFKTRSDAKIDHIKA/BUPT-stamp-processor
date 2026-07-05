/// @file chip.cpp
/// @brief Top-level chip: global clock loop, completion, verification.

#include "cycle/chip.h"

#include <algorithm>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace mobol::cycle {

Chip::Chip(const CycleConfig& cfg)
    : cfg_(cfg), fab_(cfg_), noc_(cfg_), dram_(cfg_, mem_) {
    banks_.reserve(NUM_BANKS);
    for (int b = 0; b < NUM_BANKS; b++)
        banks_.emplace_back(static_cast<BankId>(b), cfg_, mem_);
    tiles_.reserve(NUM_TILES);
    for (int t = 0; t < NUM_TILES; t++)
        tiles_.emplace_back(static_cast<TileId>(t), cfg_, mem_, fab_, events_);
}

bool Chip::fabric_empty() const {
    for (int t = 0; t < NUM_TILES; t++) {
        for (int d = 0; d < 2; d++)
            for (int v = 0; v < NUM_VCS; v++)
                if (!fab_.ring_in[t][d][v].empty() || !fab_.inject[t][d][v].empty())
                    return false;
        for (int v = 0; v < NUM_VCS; v++)
            if (!fab_.tile_ingress[t][v].empty()) return false;
        if (!fab_.vup_tile_bank[t].empty() || !fab_.vdown_bank_tile[t].empty())
            return false;
    }
    for (int b = 0; b < NUM_BANKS; b++)
        if (!fab_.vup_bank_dram[b].empty() || !fab_.vdown_dram_bank[b].empty())
            return false;
    return true;
}

bool Chip::all_done() const {
    for (const auto& t : tiles_)
        if (!t.idle()) return false;
    for (const auto& b : banks_)
        if (!b.idle()) return false;
    return fabric_empty() && dram_.idle();
}

void Chip::deadlock_dump(Cycle now) const {
    std::ostringstream os;
    os << "DEADLOCK at cycle " << now << ": no progress for "
       << cfg_.deadlock_window << " cycles\n";
    for (const auto& t : tiles_)
        if (!t.idle()) os << "  " << t.state_string() << "\n";
    std::cerr << os.str();
    throw std::runtime_error("simulation deadlock");
}

ChipSummary Chip::run(const Program& prog) {
    for (int t = 0; t < NUM_TILES; t++)
        tiles_[t].set_program(&prog.code[t]);

    Cycle now = 0;
    Cycle last_progress = 0;

    for (now = 0; now < cfg_.max_cycles; now++) {
        // Tick order: DRAM die -> buffer die -> NoC -> tiles. All
        // cross-component queues carry >=1-cycle latency stamps, so the
        // order affects only intra-cycle throughput, not causality.
        dram_.tick(fab_, now);
        for (auto& b : banks_) b.tick(fab_, now);
        noc_.tick(fab_, now);

        bool progress = false;
        for (auto& t : tiles_) progress |= t.tick(now);

        if (progress) last_progress = now;
        if (all_done()) { now++; break; }
        if (now - last_progress > cfg_.deadlock_window) deadlock_dump(now);
    }
    if (now >= cfg_.max_cycles)
        throw std::runtime_error("simulation exceeded max_cycles");

    // ── Summarize ──
    ChipSummary s;
    s.total_cycles = now;
    uint64_t mxu_busy = 0;
    for (int t = 0; t < NUM_TILES; t++) {
        s.tiles[t] = tiles_[t].stats();
        s.total_instrs += s.tiles[t].instrs;
        mxu_busy += static_cast<uint64_t>(s.tiles[t].mxu_busy_cycles);
    }
    s.mxu_utilization = now > 0
        ? static_cast<double>(mxu_busy) / (static_cast<double>(now) * NUM_TILES) : 0.0;

    uint64_t total_link_flits = 0, max_link = 0;
    for (int t = 0; t < NUM_TILES; t++)
        for (int d = 0; d < 2; d++) {
            total_link_flits += fab_.ring_link_flits[t][d];
            max_link = std::max(max_link, fab_.ring_link_flits[t][d]);
        }
    s.avg_noc_link_utilization = now > 0
        ? static_cast<double>(total_link_flits) / (static_cast<double>(now) * NUM_TILES * 2) : 0.0;
    s.max_noc_link_utilization = now > 0
        ? static_cast<double>(max_link) / static_cast<double>(now) : 0.0;
    s.noc_flit_hops = noc_.total_flit_hops();
    s.noc_eject_blocked = noc_.eject_blocked_cycles();
    s.vbond_tile_bank_flits = fab_.vbond_tile_bank_flits;
    s.vbond_bank_dram_flits = fab_.vbond_bank_dram_flits;

    for (int b = 0; b < NUM_BANKS; b++) {
        s.shared_accesses[b] = banks_[b].accesses();
        s.shared_contention[b] = banks_[b].contention_events();
        s.nmc_ops[b] = banks_[b].nmc_ops();
        s.nmc_busy[b] = banks_[b].nmc_busy_cycles();
    }

    s.dram_reads = dram_.reads();
    s.dram_writes = dram_.writes();
    s.dram_read_bytes = dram_.read_bytes();
    s.dram_write_bytes = dram_.write_bytes();
    s.dram_avg_read_latency = dram_.avg_read_latency_dram_cycles();
    s.dram_is_ramulator = dram_.using_ramulator();

    verify_causality(s);
    return s;
}

void Chip::verify_causality(ChipSummary& s) const {
    int violations = 0;

    // 1. Every acquire pass must be at/after enough matching token
    //    arrivals (consumer-side happens-before).
    //    Group token arrivals per (consumer, tag), sorted by cycle;
    //    acquires consume them generation by generation.
    std::map<std::pair<TileId, uint32_t>, std::vector<Cycle>> arrivals;
    for (const auto& a : events_.token_arrivals)
        arrivals[{a.consumer, a.tag}].push_back(a.cycle);
    for (auto& [k, v] : arrivals) std::sort(v.begin(), v.end());

    std::map<std::pair<TileId, uint32_t>, size_t> consumed;
    std::vector<EventLog::Sync> acq = events_.acquire_passes;
    std::sort(acq.begin(), acq.end(),
              [](const auto& x, const auto& y) { return x.cycle < y.cycle; });
    for (const auto& a : acq) {
        auto key = std::make_pair(a.tile, a.tag);
        auto it = arrivals.find(key);
        size_t need = consumed[key] + a.arity;
        if (it == arrivals.end() || it->second.size() < need) {
            violations++;
            std::cerr << "CAUSALITY: acquire(tile" << int(a.tile) << ", tag "
                      << a.tag << ") passed without enough releases\n";
            continue;
        }
        Cycle last_needed = it->second[need - 1];
        if (a.cycle < last_needed) {
            violations++;
            std::cerr << "CAUSALITY: acquire@" << a.cycle
                      << " precedes its release token@" << last_needed << "\n";
        }
        consumed[key] = need;
    }

    // 2. DMA spans must be causal.
    for (const auto& d : events_.dmas)
        if (d.complete < d.start) {
            violations++;
            std::cerr << "CAUSALITY: DMA completes before it starts\n";
        }

    s.causality_ok = (violations == 0);
    s.causality_violations = violations;
}

void Chip::print_summary(const ChipSummary& s) const {
    std::cout << "──────────────────────────────────────────────────\n";
    std::cout << "Total logic cycles : " << s.total_cycles
              << "  (DRAM cycles: " << s.total_cycles * dram_.ratio()
              << ", logic:DRAM = 1:" << dram_.ratio() << ")\n";
    std::cout << "Instructions       : " << s.total_instrs << "\n";
    std::cout << "MXU ops            : " << events_.mxu_ops
              << "   chip MXU occupancy: " << s.mxu_utilization * 100 << "%\n";
    std::cout << "NoC  : flit-hops " << s.noc_flit_hops
              << ", avg link util " << s.avg_noc_link_utilization * 100
              << "%, max link util " << s.max_noc_link_utilization * 100
              << "%, eject-blocked " << s.noc_eject_blocked << "\n";
    std::cout << "3D   : tile<->bank flits " << s.vbond_tile_bank_flits
              << ", bank<->DRAM flits " << s.vbond_bank_dram_flits << "\n";
    std::cout << "SHARED banks accesses [";
    for (int b = 0; b < NUM_BANKS; b++)
        std::cout << s.shared_accesses[b] << (b + 1 < NUM_BANKS ? " " : "");
    std::cout << "]  contention [";
    for (int b = 0; b < NUM_BANKS; b++)
        std::cout << s.shared_contention[b] << (b + 1 < NUM_BANKS ? " " : "");
    std::cout << "]\n";
    if (cfg_.nmc_enable) {
        std::cout << "NMC  : ops [";
        for (int b = 0; b < NUM_BANKS; b++)
            std::cout << s.nmc_ops[b] << (b + 1 < NUM_BANKS ? " " : "");
        std::cout << "]  busy cycles [";
        for (int b = 0; b < NUM_BANKS; b++)
            std::cout << s.nmc_busy[b] << (b + 1 < NUM_BANKS ? " " : "");
        std::cout << "]\n";
    }
    std::cout << "DRAM (" << (s.dram_is_ramulator ? "Ramulator2" : "fallback")
              << "): reads " << s.dram_reads << " (" << s.dram_read_bytes
              << " B), writes " << s.dram_writes << " (" << s.dram_write_bytes
              << " B), avg read latency " << s.dram_avg_read_latency
              << " DRAM cycles\n";
    std::cout << "Causality          : "
              << (s.causality_ok ? "OK" : "VIOLATED") << "\n";
    std::cout << "Per-tile (instrs busy stall[acq dma mxu vpu] mxu_ops):\n";
    for (int t = 0; t < NUM_TILES; t++) {
        const auto& ts = s.tiles[t];
        if (ts.instrs == 0) continue;
        std::cout << "  T" << t << ": " << ts.instrs << " " << ts.busy_cycles
                  << " [" << ts.stall_acquire << " " << ts.stall_dma << " "
                  << ts.stall_mxu << " " << ts.stall_vpu << "] "
                  << ts.mxu_ops << "\n";
    }
    std::cout << "──────────────────────────────────────────────────\n";
}

} // namespace mobol::cycle
