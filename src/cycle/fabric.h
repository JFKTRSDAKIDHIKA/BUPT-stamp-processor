/// @file fabric.h
/// @brief On-chip interconnect: NoC ring, vertical hybrid bonds, bank ctrl.
///
/// Physical topology (MOBOL spec):
///   Base die   : 16 tiles on a bidirectional ring. 1 flit (64 B) per link
///                per direction per cycle, 1 cycle per hop, buffered routers
///                with backpressure.
///   Buffer die : 4 shared-scratchpad banks. Each tile has a private
///                vertical hybrid-bond link to its group's bank (near
///                access = 1 vertical hop). Far-bank traffic rides the ring
///                to a gateway tile of the target group, then goes up.
///   DRAM die   : one memory controller, fed by 4 per-group vertical
///                columns (bank die is a passthrough for DRAM traffic).
///
/// All queues carry latency stamps; nothing crosses a link in 0 cycles.
#pragma once

#include "cycle/flit.h"
#include "cycle/cycle_config.h"
#include "common/address.h"
#include "common/memory_model.h"

#include <array>
#include <cstdint>
#include <type_traits>

namespace mobol::cycle {

/// All fabric queues + link statistics. Owned by Chip, operated on by
/// RingNoC / BankCtrl / DramCtrl / TileCore.
struct Fabric {
    // NoC input buffer at node n holding flits travelling in direction d
    // (ring: 0 = clockwise, 1 = counter-clockwise; mesh/torus: E/W/S/N),
    // separated per virtual channel (VC0 requests / VC1 responses).
    FlitQueue ring_in[NUM_TILES][NOC_DIRS][NUM_VCS];
    // Per-tile injection queues, one per first-hop direction per VC.
    FlitQueue inject[NUM_TILES][NOC_DIRS][NUM_VCS];
    // Flits that have arrived at their destination tile (ring eject or
    // vertical down) and await ingress processing (SRAM commit etc.),
    // per VC so blocked writes cannot head-of-line block responses.
    FlitQueue tile_ingress[NUM_TILES][NUM_VCS];
    // Vertical hybrid bonds.
    FlitQueue vup_tile_bank[NUM_TILES];    ///< tile -> its group's bank
    FlitQueue vdown_bank_tile[NUM_TILES];  ///< bank -> tile
    FlitQueue vup_bank_dram[NUM_BANKS];    ///< bank column -> DRAM ctrl
    FlitQueue vdown_dram_bank[NUM_BANKS];  ///< DRAM ctrl -> bank column

    // ── Statistics ──
    uint64_t ring_link_flits[NUM_TILES][NOC_DIRS] = {};  ///< flits leaving node n, dir d
    uint64_t ring_inject_flits = 0;
    uint64_t vbond_tile_bank_flits = 0;
    uint64_t vbond_bank_dram_flits = 0;

    explicit Fabric(const CycleConfig& cfg);

    /// Route a flit leaving tile `t` into the correct first-hop queue.
    /// Returns false (and leaves `f` unconsumed) if that queue is full.
    /// `now` stamps ready_at = now + 1 (injection/vertical latency).
    bool route_from_tile(TileId t, Flit&& f, Cycle now);

    /// Ring destination for a flit injected at tile `t` (final tile for
    /// tile-directed flits; gateway tile for far-bank memory traffic).
    static TileId ring_dest_for(TileId t, const Flit& f);
};

/// Cycle-accurate bidirectional ring router logic.
class RingNoC {
public:
    explicit RingNoC(const CycleConfig& cfg) : cfg_(cfg) {}

    /// Advance every router by one cycle: eject, forward, then inject.
    void tick(Fabric& fab, Cycle now);

    uint64_t total_flit_hops() const { return flit_hops_; }
    uint64_t eject_blocked_cycles() const { return eject_blocked_; }

private:
    const CycleConfig& cfg_;
    uint64_t flit_hops_ = 0;
    uint64_t eject_blocked_ = 0;

    /// Move the head flit of ring_in[n][d][vc]: eject here or forward.
    /// Returns true if the outgoing link (n -> next) was used.
    bool process_node(Fabric& fab, TileId n, int d, int vc, Cycle now);
    /// Try to inject from inject[n][d][vc]. Returns true if link used.
    bool try_inject(Fabric& fab, TileId n, int d, int vc, Cycle now);

public:
    /// Sink a flit that terminates at node n (tile ingress or gateway
    /// vertical up-link). Topology-independent; shared with GridNoC.
    static bool eject(Fabric& fab, TileId n, Flit&& f, Cycle now);
};

/// Cycle-accurate 2D mesh/torus router logic (XY dimension-order routing,
/// one flit per directed link per cycle, per-cycle link arbitration:
/// responses (VC1) beat requests, through traffic beats injection).
class GridNoC {
public:
    explicit GridNoC(const CycleConfig& cfg) : cfg_(cfg) {}

    void tick(Fabric& fab, Cycle now);

    uint64_t total_flit_hops() const { return flit_hops_; }
    uint64_t eject_blocked_cycles() const { return eject_blocked_; }

private:
    const CycleConfig& cfg_;
    uint64_t flit_hops_ = 0;
    uint64_t eject_blocked_ = 0;

    /// Forward the head of ring_in[n][d][vc] one hop (or eject). Returns
    /// true if it used an output link (marks it in `used`).
    bool forward_head(Fabric& fab, TileId n, int d, int vc, Cycle now,
                      bool used[][NOC_DIRS]);
    bool inject_head(Fabric& fab, TileId n, int d, int vc, Cycle now,
                     bool used[][NOC_DIRS]);
};

/// Topology selected at compile time (MOBOL_TOPOLOGY).
using NocImpl = std::conditional_t<TOPOLOGY == TOPO_RING, RingNoC, GridNoC>;

/// Shared-scratchpad bank controller on the buffer die (one per bank).
/// Serves SHARED reads/writes arriving on the four tile up-links of its
/// group, and passes DRAM-bound traffic through to the DRAM column.
class BankCtrl {
public:
    BankCtrl(BankId bank, const CycleConfig& cfg, MemoryModel& mem)
        : bank_(bank), cfg_(cfg), mem_(mem) {}

    void tick(Fabric& fab, Cycle now);

    uint64_t accesses() const { return accesses_; }
    uint64_t contention_events() const { return contention_; }
    uint64_t nmc_ops() const { return nmc_ops_; }
    Cycle nmc_busy_cycles() const { return nmc_busy_cycles_; }
    uint64_t prefetch_bytes() const { return pf_bytes_; }
    bool idle() const {
        return nmc_q_.empty() && !nmc_active_
            && pf_q_.empty() && !pf_active_;
    }

private:
    BankId bank_;
    const CycleConfig& cfg_;
    MemoryModel& mem_;
    int rr_ = 0;              ///< round-robin pointer over the 4 up-links
    uint64_t accesses_ = 0;   ///< SRAM accesses served (always counted, Q2)
    uint64_t contention_ = 0; ///< requests deferred by the port limit

    // ── Near-memory compute engine (paradigm C) ──
    std::deque<Flit> nmc_q_;
    bool nmc_active_ = false;
    Cycle nmc_done_at_ = 0;
    std::vector<uint8_t> nmc_result_;  ///< staged output, committed at done
    Addr nmc_dst_ = 0;
    Flit nmc_token_;                   ///< completion release token
    uint64_t nmc_ops_ = 0;
    Cycle nmc_busy_cycles_ = 0;

    void nmc_start(const Flit& cmd, Cycle now);
    void nmc_tick(Fabric& fab, Cycle now);

    // ── Bank prefetch (DMA) engine: DRAM -> this bank's SRAM via the
    //    group's own vertical column, no base-die involvement ──
    std::deque<Flit> pf_q_;
    bool pf_active_ = false;
    Flit pf_cmd_;
    uint32_t pf_row_ = 0, pf_off_ = 0;
    bool pf_gen_done_ = false;
    uint32_t pf_outstanding_ = 0;
    uint64_t pf_bytes_ = 0;

    void prefetch_tick(Fabric& fab, Cycle now);
    /// Called when a bank-directed READ_RESP arrives on the down column.
    void prefetch_data(const Flit& f, Fabric& fab, Cycle now);

    /// Route a tile-directed flit downward: direct link if the target is
    /// in this group, else via the gateway tile of this group.
    bool send_down(Fabric& fab, Flit&& f, Cycle now);
};

} // namespace mobol::cycle
