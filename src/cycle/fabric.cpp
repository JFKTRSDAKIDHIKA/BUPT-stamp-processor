/// @file fabric.cpp
/// @brief NoC ring, vertical links and bank controller implementation.

#include "cycle/fabric.h"
#include "cycle/blockops.h"

#include <cassert>
#include <cstring>
#include <stdexcept>

namespace mobol::cycle {

// ═══ Fabric ══════════════════════════════════════════════════

Fabric::Fabric(const CycleConfig& cfg) {
    for (int t = 0; t < NUM_TILES; t++) {
        for (int d = 0; d < NOC_DIRS; d++)
            for (int v = 0; v < NUM_VCS; v++) {
                ring_in[t][d][v].set_capacity(cfg.noc_buffer_depth);
                inject[t][d][v].set_capacity(cfg.noc_inject_queue_depth);
            }
        for (int v = 0; v < NUM_VCS; v++)
            tile_ingress[t][v].set_capacity(cfg.noc_buffer_depth * 4);
        vup_tile_bank[t].set_capacity(cfg.vbond_queue_depth);
        vdown_bank_tile[t].set_capacity(cfg.vbond_queue_depth);
    }
    for (int b = 0; b < NUM_BANKS; b++) {
        vup_bank_dram[b].set_capacity(cfg.dram_col_queue_depth);
        vdown_dram_bank[b].set_capacity(cfg.dram_col_queue_depth);
    }
}

TileId Fabric::ring_dest_for(TileId t, const Flit& f) {
    if (f.tile_directed) return f.target_tile;
    // Memory-op flit: ring destination depends on the target segment.
    Segment seg = get_segment(f.addr);
    if (seg == Segment::LOCAL) return get_local_tile(f.addr);
    if (seg == Segment::SHARED) {
        BankId b = get_shared_bank(f.addr);
        return gateway_tile(t, b);
    }
    // DRAM traffic never rides the ring (each tile has its own column).
    return t;
}

bool Fabric::route_from_tile(TileId t, Flit&& f, Cycle now) {
    // Decide: vertical up, or ring (and in which direction).
    bool vertical;
    if (f.tile_directed) {
        vertical = false;               // tile-to-tile always via ring
        assert(f.target_tile != t && "self-directed flit must be folded");
    } else {
        Segment seg = get_segment(f.addr);
        if (seg == Segment::DRAM) {
            vertical = true;            // own column, straight up
        } else if (seg == Segment::SHARED) {
            vertical = (get_shared_bank(f.addr) == tile_group(t));
        } else {
            vertical = false;           // remote LOCAL: ring
            assert(get_local_tile(f.addr) != t && "self LOCAL must be folded");
        }
    }

    if (vertical) {
        if (vup_tile_bank[t].full()) return false;
        vup_tile_bank[t].push(std::move(f), now + 1);
        vbond_tile_bank_flits++;
        return true;
    }

    TileId dest = ring_dest_for(t, f);
    if (dest == t) {
        // Far-shared gateway happens to be this tile: go straight up.
        if (vup_tile_bank[t].full()) return false;
        vup_tile_bank[t].push(std::move(f), now + 1);
        vbond_tile_bank_flits++;
        return true;
    }
    int d = noc_first_dir(t, dest);
    int vc = flit_vc(f.kind);
    if (inject[t][d][vc].full()) return false;
    f.ring_dest = dest;
    f.dir = d;
    inject[t][d][vc].push(std::move(f), now + 1);
    ring_inject_flits++;
    return true;
}

// ═══ RingNoC ═════════════════════════════════════════════════

bool RingNoC::eject(Fabric& fab, TileId n, Flit&& f, Cycle now) {
    int vc = flit_vc(f.kind);
    if (f.tile_directed) {
        if (f.target_tile == n) {
            if (fab.tile_ingress[n][vc].full()) return false;
            fab.tile_ingress[n][vc].push(std::move(f), now + 1);
            return true;
        }
        // Should not happen: ring_dest is always the final tile.
        throw std::runtime_error("NoC: tile-directed flit ejected at wrong node");
    }
    Segment seg = get_segment(f.addr);
    if (seg == Segment::LOCAL) {
        assert(get_local_tile(f.addr) == n);
        if (fab.tile_ingress[n][vc].full()) return false;
        fab.tile_ingress[n][vc].push(std::move(f), now + 1);
        return true;
    }
    // Gateway exit: SHARED (this node's group bank) — go up the bond.
    assert(seg == Segment::SHARED && get_shared_bank(f.addr) == tile_group(n));
    if (fab.vup_tile_bank[n].full()) return false;
    fab.vup_tile_bank[n].push(std::move(f), now + 1);
    fab.vbond_tile_bank_flits++;
    return true;
}

bool RingNoC::process_node(Fabric& fab, TileId n, int d, int vc, Cycle now) {
    // Ejection phase for one VC: if the head terminates here, sink it.
    // Ejection uses the (per-VC) eject port, not the forward link.
    Flit* head = fab.ring_in[n][d][vc].front_ready(now);
    if (!head || head->moved_at >= now) return false;
    if (head->ring_dest != n) return false;

    Flit f = *head;
    f.moved_at = now;
    if (eject(fab, n, std::move(f), now)) {
        fab.ring_in[n][d][vc].pop();
        return true;
    }
    eject_blocked_++;   // head-of-line blocking within this VC, retry
    return false;
}

bool RingNoC::try_inject(Fabric& fab, TileId n, int d, int vc, Cycle now) {
    // Link phase for one VC: forward the through-flit if it continues past
    // this node, else inject a fresh flit. Returns true iff the physical
    // link (n -> next) was used.
    TileId next = (d == 0) ? static_cast<TileId>((n + 1) % NUM_TILES)
                           : static_cast<TileId>((n + NUM_TILES - 1) % NUM_TILES);

    Flit* head = fab.ring_in[n][d][vc].front_ready(now);
    if (head && head->moved_at < now && head->ring_dest != n) {
        if (fab.ring_in[next][d][vc].full()) return false; // backpressure
        Flit f = *head;
        fab.ring_in[n][d][vc].pop();
        f.moved_at = now;
        fab.ring_in[next][d][vc].push(std::move(f), now + cfg_.noc_hop_latency);
        fab.ring_link_flits[n][d]++;
        flit_hops_++;
        return true;
    }
    // No through traffic on this VC: injection may use the link.
    Flit* inj = fab.inject[n][d][vc].front_ready(now);
    if (!inj || inj->moved_at >= now) return false;
    if (fab.ring_in[next][d][vc].full()) return false;
    Flit f = *inj;
    fab.inject[n][d][vc].pop();
    f.moved_at = now;
    fab.ring_in[next][d][vc].push(std::move(f), now + cfg_.noc_hop_latency);
    fab.ring_link_flits[n][d]++;
    flit_hops_++;
    return true;
}

void RingNoC::tick(Fabric& fab, Cycle now) {
    // One physical link per (node, direction) carries at most one flit per
    // cycle, shared by both VCs; each VC has its own buffers and eject
    // port. Arbitration: responses (VC1) win the link over requests (VC0);
    // within a VC, through traffic beats injection (standard ring rule).
    // VC1 always drains at its target, so request->response protocol
    // deadlock is impossible.
    for (int d = 0; d < 2; d++) {
        for (int n = 0; n < NUM_TILES; n++) {
            TileId t = static_cast<TileId>(n);
            // Ejection (per-VC ports, no link usage).
            process_node(fab, t, d, VC_RESP, now);
            process_node(fab, t, d, VC_REQ, now);
            // Physical link: VC1 first, then VC0.
            if (!try_inject(fab, t, d, VC_RESP, now))
                try_inject(fab, t, d, VC_REQ, now);
        }
    }
}

// ═══ GridNoC (mesh / torus, XY dimension-order routing) ═══════
// Only compiled for grid topologies: in a ring build NOC_DIRS is 2 and
// the 4-direction indexing would be statically out of bounds.
#if MOBOL_TOPOLOGY != 0
//
// Per cycle each directed link (n -> neighbor(n, d)) carries at most one
// flit, tracked in `used[n][d]`. Arbitration mirrors the ring's rules:
// ejection uses per-VC eject ports (no link), responses (VC1) win links
// over requests (VC0), and through traffic beats fresh injection. XY
// routing is deadlock-free on the mesh; on the torus the wrap links can
// in principle form buffer cycles — the simulator's deadlock window
// aborts such a run loudly (the variant is then INVALID) rather than
// producing wrong cycle counts.

bool GridNoC::forward_head(Fabric& fab, TileId n, int d, int vc, Cycle now,
                           bool used[][NOC_DIRS]) {
    Flit* head = fab.ring_in[n][d][vc].front_ready(now);
    if (!head || head->moved_at >= now || head->ring_dest == n) return false;
    int out = grid_next_dir(n, head->ring_dest);
    if (used[n][out]) return false;
    TileId next = grid_neighbor(n, out);
    if (fab.ring_in[next][out][vc].full()) return false;   // backpressure
    Flit f = *head;
    fab.ring_in[n][d][vc].pop();
    f.moved_at = now;
    f.dir = out;
    fab.ring_in[next][out][vc].push(std::move(f), now + cfg_.noc_hop_latency);
    fab.ring_link_flits[n][out]++;
    flit_hops_++;
    used[n][out] = true;
    return true;
}

bool GridNoC::inject_head(Fabric& fab, TileId n, int d, int vc, Cycle now,
                          bool used[][NOC_DIRS]) {
    Flit* inj = fab.inject[n][d][vc].front_ready(now);
    if (!inj || inj->moved_at >= now) return false;
    if (used[n][d]) return false;
    TileId next = grid_neighbor(n, d);
    if (fab.ring_in[next][d][vc].full()) return false;
    Flit f = *inj;
    fab.inject[n][d][vc].pop();
    f.moved_at = now;
    fab.ring_in[next][d][vc].push(std::move(f), now + cfg_.noc_hop_latency);
    fab.ring_link_flits[n][d]++;
    flit_hops_++;
    used[n][d] = true;
    return true;
}

void GridNoC::tick(Fabric& fab, Cycle now) {
    bool used[NUM_TILES][NOC_DIRS] = {};

    // Ejection phase (per-VC eject ports, no link usage).
    for (int n = 0; n < NUM_TILES; n++) {
        TileId t = static_cast<TileId>(n);
        for (int d = 0; d < NOC_DIRS; d++) {
            for (int vc : {VC_RESP, VC_REQ}) {
                Flit* head = fab.ring_in[t][d][vc].front_ready(now);
                if (!head || head->moved_at >= now || head->ring_dest != t)
                    continue;
                Flit f = *head;
                f.moved_at = now;
                if (RingNoC::eject(fab, t, std::move(f), now))
                    fab.ring_in[t][d][vc].pop();
                else
                    eject_blocked_++;
            }
        }
    }

    // Link phase: responses first, through traffic before injection.
    for (int vc : {VC_RESP, VC_REQ}) {
        for (int n = 0; n < NUM_TILES; n++)
            for (int d = 0; d < NOC_DIRS; d++)
                forward_head(fab, static_cast<TileId>(n), d, vc, now, used);
        for (int n = 0; n < NUM_TILES; n++)
            for (int d = 0; d < NOC_DIRS; d++)
                inject_head(fab, static_cast<TileId>(n), d, vc, now, used);
    }
}
#endif  // MOBOL_TOPOLOGY != 0

// ═══ BankCtrl ════════════════════════════════════════════════

bool BankCtrl::send_down(Fabric& fab, Flit&& f, Cycle now) {
    TileId target = f.target_tile;
    TileId exit_tile;
    if (tile_group(target) == bank_) {
        exit_tile = target;                       // direct vertical drop
    } else {
        // Drop at the gateway tile of this group nearest to the target;
        // that tile re-injects the flit into the ring.
        exit_tile = gateway_tile(target, static_cast<GroupId>(bank_));
    }
    if (fab.vdown_bank_tile[exit_tile].full()) return false;
    fab.vdown_bank_tile[exit_tile].push(std::move(f), now + cfg_.vbond_latency);
    return true;
}

void BankCtrl::tick(Fabric& fab, Cycle now) {
    // 1. DRAM column downstream: bank-directed prefetch data is absorbed
    //    here (written into this bank's SRAM); tile-directed flits go to
    //    their tile links. The column moves `vbond_dram_flits_per_cycle`
    //    flits per cycle (hybrid-bond density knob).
    for (int k = 0; k < cfg_.vbond_dram_flits_per_cycle; k++) {
        Flit* f = fab.vdown_dram_bank[bank_].front_ready(now);
        if (!f) break;
        if (f->to_bank) {
            prefetch_data(*f, fab, now);
            fab.vdown_dram_bank[bank_].pop();
            continue;
        }
        // DRAM responses descend the requester's own column, so the target
        // is always in this group.
        assert(tile_group(f->target_tile) == bank_);
        Flit copy = *f;
        if (!send_down(fab, std::move(copy), now)) break;
        fab.vdown_dram_bank[bank_].pop();
    }

    // 2. Serve the four tile up-links, round-robin, up to the tile-link
    //    width per link per cycle. SRAM port limit applies when
    //    contention modeling is on.
    int sram_budget = cfg_.shared_port_contention ? cfg_.shared_rw_ports
                                                  : 1 << 30;
    for (int i = 0; i < TILES_PER_GROUP; i++) {
        TileId t = static_cast<TileId>(bank_ * TILES_PER_GROUP
                                       + (rr_ + i) % TILES_PER_GROUP);
        for (int w = 0; w < cfg_.vbond_flits_per_cycle; w++) {
        Flit* f = fab.vup_tile_bank[t].front_ready(now);
        if (!f) break;

        Segment seg = f->tile_directed ? Segment::NULL_SEG : get_segment(f->addr);

        if (seg == Segment::DRAM) {
            // Buffer die is a passthrough for DRAM traffic.
            if (fab.vup_bank_dram[bank_].full()) break;
            Flit copy = *f;
            fab.vup_tile_bank[t].pop();
            fab.vup_bank_dram[bank_].push(std::move(copy), now + cfg_.vbond_latency);
            fab.vbond_bank_dram_flits++;
            continue;
        }

        if (seg != Segment::SHARED) {
            throw std::runtime_error("BankCtrl: unexpected flit on up-link");
        }
        assert(get_shared_bank(f->addr) == bank_);

        if (f->kind == FlitKind::NMC_CMD) {
            if (f->nmc_op == NmcOp::PREFETCH) {
                if (pf_q_.size() >= static_cast<size_t>(cfg_.nmc_queue_depth)) break;
                pf_q_.push_back(*f);
                fab.vup_tile_bank[t].pop();
                continue;
            }
            if (!cfg_.nmc_enable)
                throw std::runtime_error("NMC_CMD received but NMC is disabled");
            if (nmc_q_.size() >= static_cast<size_t>(cfg_.nmc_queue_depth)) break;
            nmc_q_.push_back(*f);
            fab.vup_tile_bank[t].pop();
            continue;
        }

        if (sram_budget <= 0) { contention_++; break; }

        if (f->kind == FlitKind::DATA_WRITE) {
            Flit ack;
            ack.kind = FlitKind::WRITE_ACK;
            ack.tile_directed = true;
            ack.target_tile = f->requester;
            ack.requester = f->requester;
            ack.dma_seq = f->dma_seq;
            ack.bytes = f->bytes;
            if (!send_down(fab, std::move(ack), now)) break; // retry later
            mem_.write(f->addr, f->payload.data(), f->bytes);
            accesses_++;
            sram_budget--;
            fab.vup_tile_bank[t].pop();
        } else if (f->kind == FlitKind::READ_REQ) {
            Flit resp;
            resp.kind = FlitKind::READ_RESP;
            resp.tile_directed = true;
            resp.target_tile = f->requester;
            resp.requester = f->requester;
            resp.dma_seq = f->dma_seq;
            resp.resp_addr = f->resp_addr;
            resp.bytes = f->bytes;
            resp.payload.resize(f->bytes);
            mem_.read(f->addr, resp.payload.data(), f->bytes);
            if (!send_down(fab, std::move(resp), now)) break;
            accesses_++;
            sram_budget--;
            fab.vup_tile_bank[t].pop();
        } else {
            throw std::runtime_error("BankCtrl: unexpected flit kind");
        }
        }
    }
    rr_ = (rr_ + 1) % TILES_PER_GROUP;

    nmc_tick(fab, now);
    prefetch_tick(fab, now);
}

// ─── Bank prefetch engine (weight streaming) ─────────────────

void BankCtrl::prefetch_tick(Fabric& fab, Cycle now) {
    if (!pf_active_) {
        if (pf_q_.empty()) return;
        pf_cmd_ = pf_q_.front();
        pf_q_.pop_front();
        pf_row_ = pf_off_ = 0;
        pf_gen_done_ = false;
        pf_active_ = true;
    }

    // Generate read requests up the column, at hybrid-bond density rate.
    for (int k = 0; k < cfg_.vbond_dram_flits_per_cycle && !pf_gen_done_; k++) {
        if (pf_outstanding_ >= static_cast<uint32_t>(cfg_.dma_max_inflight_chunks))
            break;
        if (fab.vup_bank_dram[bank_].full()) break;

        Addr sa = static_cast<Addr>(static_cast<int64_t>(pf_cmd_.nmc_src)
                  + static_cast<int64_t>(pf_row_) * pf_cmd_.nmc_sstride + pf_off_);
        Addr da = static_cast<Addr>(static_cast<int64_t>(pf_cmd_.addr)
                  + static_cast<int64_t>(pf_row_) * pf_cmd_.nmc_dstride + pf_off_);
        uint32_t len = std::min<uint32_t>(64, pf_cmd_.nmc_row_bytes - pf_off_);
        len = std::min<uint32_t>(len,
              static_cast<uint32_t>(64 - (get_offset(sa) & 63)));

        Flit req;
        req.kind = FlitKind::READ_REQ;
        req.to_bank = true;
        req.requester = pf_cmd_.requester;   // informational only
        req.addr = sa;
        req.resp_addr = da;
        req.bytes = len;
        fab.vup_bank_dram[bank_].push(std::move(req), now + cfg_.vbond_latency);
        fab.vbond_bank_dram_flits++;
        pf_outstanding_++;

        pf_off_ += len;
        if (pf_off_ >= pf_cmd_.nmc_row_bytes) { pf_off_ = 0; pf_row_++; }
        if (pf_row_ >= pf_cmd_.nmc_count) pf_gen_done_ = true;
    }

    // Completion: all data landed in this bank's SRAM -> release token.
    if (pf_gen_done_ && pf_outstanding_ == 0) {
        Flit tok;
        tok.kind = FlitKind::RELEASE_TOK;
        tok.tile_directed = true;
        tok.target_tile = pf_cmd_.requester;
        tok.requester = pf_cmd_.requester;
        tok.tag = pf_cmd_.tag;
        if (!send_down(fab, std::move(tok), now)) return;  // retry
        pf_active_ = false;
    }
}

void BankCtrl::prefetch_data(const Flit& f, Fabric& fab, Cycle now) {
    (void)fab; (void)now;
    assert(get_segment(f.resp_addr) == Segment::SHARED
           && get_shared_bank(f.resp_addr) == bank_);
    mem_.write(f.resp_addr, f.payload.data(), f.bytes);
    accesses_++;
    pf_bytes_ += f.bytes;
    if (pf_outstanding_ == 0)
        throw std::runtime_error("BankCtrl: prefetch outstanding underflow");
    pf_outstanding_--;
}

// ─── Near-memory compute engine ──────────────────────────────

void BankCtrl::nmc_start(const Flit& cmd, Cycle now) {
    namespace bo = mobol::cycle::blockops;
    const uint32_t n = cmd.nmc_count;
    Cycle lat = 0;

    switch (cmd.nmc_op) {
        case NmcOp::REDUCE_F32: {
            // Fixed-order reduction (slot 0 -> n-1) of contiguous 1 KB
            // f32 blocks — bit-identical to the tile-VPU add chain.
            std::vector<float> acc(256), slot(256);
            mem_.read(cmd.addr, acc.data(), 1024);
            for (uint32_t j = 1; j < n; j++) {
                mem_.read(cmd.addr + j * 1024, slot.data(), 1024);
                bo::add_f32(acc.data(), slot.data(), acc.data());
            }
            nmc_result_.resize(1024);
            std::memcpy(nmc_result_.data(), acc.data(), 1024);
            lat = static_cast<Cycle>(n) * cfg_.nmc_reduce_cycles_per_block;
            break;
        }
        case NmcOp::LN_F16: {
            // Fused layernorm (rows span 16*n columns) + f32->f16 convert.
            const size_t elems = size_t{16} * 16 * n;
            std::vector<float> in(elems), norm(elems);
            mem_.read(cmd.addr, in.data(), elems * 4);
            bo::layernorm_f32(in.data(), static_cast<int>(n), cmd.nmc_scalar,
                              norm.data());
            nmc_result_.resize(elems * 2);
            f16* out = reinterpret_cast<f16*>(nmc_result_.data());
            for (size_t i = 0; i < elems; i++) out[i] = f16(norm[i]);
            lat = static_cast<Cycle>(n)
                  * (cfg_.nmc_ln_cycles_per_block + cfg_.nmc_cvt_cycles_per_block);
            break;
        }
        default:
            throw std::runtime_error("NMC: unknown op");
    }

    nmc_dst_ = cmd.resp_addr;
    nmc_done_at_ = now + lat;
    nmc_active_ = true;
    nmc_ops_++;

    nmc_token_ = Flit{};
    nmc_token_.kind = FlitKind::RELEASE_TOK;
    nmc_token_.tile_directed = true;
    nmc_token_.target_tile = cmd.requester;
    nmc_token_.requester = cmd.requester;
    nmc_token_.tag = cmd.tag;
}

void BankCtrl::nmc_tick(Fabric& fab, Cycle now) {
    if (nmc_active_) {
        nmc_busy_cycles_++;
        if (now >= nmc_done_at_) {
            // Commit the staged result, then signal completion. The token
            // may be back-pressured; retry until it leaves.
            if (!nmc_result_.empty()) {
                mem_.write(nmc_dst_, nmc_result_.data(), nmc_result_.size());
                accesses_++;
                nmc_result_.clear();
            }
            Flit tok = nmc_token_;
            if (!send_down(fab, std::move(tok), now)) return;
            nmc_active_ = false;
        }
        return;
    }
    if (!nmc_q_.empty()) {
        nmc_start(nmc_q_.front(), now);
        nmc_q_.pop_front();
    }
}

} // namespace mobol::cycle
