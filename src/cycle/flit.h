/// @file flit.h
/// @brief Flit (flow-control unit) definitions for NoC and vertical links.
///
/// The fabric moves 64-byte flits. Every data transfer, read request,
/// write acknowledgement and release token is a flit occupying link
/// bandwidth — nothing travels for free.
///
/// Conservative simplification: control flits (ACK, READ_REQ, RELEASE_TOK)
/// are small on real silicon (~8-16 B) but are charged a full flit slot
/// here. This slightly overestimates contention; it never underestimates.
#pragma once

#include "common/types.h"
#include <cstdint>
#include <deque>
#include <vector>

namespace mobol::cycle {

enum class FlitKind : uint8_t {
    DATA_WRITE,   ///< carries payload toward a memory target (addr)
    WRITE_ACK,    ///< write committed; returns to requester tile
    READ_REQ,     ///< asks a memory endpoint for `bytes` at addr
    READ_RESP,    ///< carries payload back; committed at resp_addr
    RELEASE_TOK,  ///< sync token; bumps cnt[tag] at target tile
    NMC_CMD,      ///< buffer-die near-memory compute command (paradigm C):
                  ///< executed by the bank engine; completion is a
                  ///< RELEASE_TOK back to (requester, tag)
};

enum class NmcOp : uint8_t {
    NONE = 0,
    REDUCE_F32,   ///< dst(1KB f32) = fixed-order sum of `count` 1KB slots at addr
    LN_F16,       ///< dst(f16, 16 x 16*count) = layernorm(addr f32) + convert
    PREFETCH,     ///< bank DMA: stream a 2D region DRAM -> this bank's SRAM
                  ///< through the group's own vertical column, bypassing
                  ///< the base die entirely (weight streaming)
};

/// Virtual channel assignment. Requests (VC0) may block on resources that
/// responses free up; responses (VC1) are always sinkable at their target.
/// Keeping them in separate buffers (sharing the physical link, responses
/// prioritized) makes request->response protocol deadlock impossible —
/// the standard 2-VC solution used by real interconnects.
constexpr int VC_REQ = 0;
constexpr int VC_RESP = 1;
constexpr int NUM_VCS = 2;

inline int flit_vc(FlitKind k) {
    return (k == FlitKind::DATA_WRITE || k == FlitKind::READ_REQ
            || k == FlitKind::NMC_CMD) ? VC_REQ : VC_RESP;
}

struct Flit {
    FlitKind kind = FlitKind::DATA_WRITE;

    // Originator (whose DMA engine / sequencer awaits completion).
    TileId requester = 0;
    uint32_t dma_seq = 0;      ///< descriptor id within requester tile

    // Memory-op fields (DATA_WRITE / READ_REQ target address).
    Addr addr = 0;
    Addr resp_addr = 0;        ///< READ_REQ/RESP: local address for the data
    uint32_t bytes = 0;
    std::vector<uint8_t> payload;

    // Tile-directed fields (WRITE_ACK / READ_RESP / RELEASE_TOK).
    TileId target_tile = 0;
    bool tile_directed = false;
    uint32_t tag = 0;          ///< RELEASE_TOK; NMC_CMD completion tag

    // NMC_CMD fields (addr = region on the TARGET BANK — also the routing
    // key; resp_addr = destination region).
    NmcOp nmc_op = NmcOp::NONE;
    uint32_t nmc_count = 0;    ///< blocks / slots / rows to process
    float nmc_scalar = 0.0f;   ///< LN epsilon etc.
    // PREFETCH descriptor (2D): DRAM source + strides.
    Addr nmc_src = 0;
    uint32_t nmc_row_bytes = 0;
    int64_t nmc_sstride = 0;
    int64_t nmc_dstride = 0;

    // Bank-directed DRAM response (prefetch data returning to a bank
    // rather than a tile).
    bool to_bank = false;

    // Ring routing state.
    TileId ring_dest = 0;      ///< tile where this flit leaves the ring
    int dir = 0;               ///< 0 = clockwise (+1), 1 = counter-clockwise
    Cycle moved_at = -1;       ///< last cycle the flit advanced (2-phase guard)
    Cycle ready_at = 0;        ///< earliest cycle a queue pop may see it
};

/// Bounded FIFO of flits with a latency stamp on entry.
class FlitQueue {
public:
    explicit FlitQueue(size_t capacity = 4) : cap_(capacity) {}

    bool full() const { return q_.size() >= cap_; }
    bool empty() const { return q_.empty(); }
    size_t size() const { return q_.size(); }

    /// Push with an earliest-visible cycle (now + link latency).
    bool push(Flit f, Cycle ready_at) {
        if (full()) return false;
        if (f.bytes > 64 || f.dir < 0 || f.dir > 1)
            throw std::logic_error("FlitQueue::push: malformed flit");
        f.ready_at = ready_at;
        q_.push_back(std::move(f));
        return true;
    }

    /// Peek head if it is visible at `now`.
    Flit* front_ready(Cycle now) {
        if (q_.empty() || q_.front().ready_at > now) return nullptr;
        return &q_.front();
    }

    void pop() { q_.pop_front(); }

    void set_capacity(size_t c) { cap_ = c; }

private:
    std::deque<Flit> q_;
    size_t cap_;
};

// ─── Routing helpers ─────────────────────────────────────────

/// Ring distance moving clockwise (increasing tile index).
inline int ring_dist_cw(TileId from, TileId to) {
    return (to - from + NUM_TILES) % NUM_TILES;
}

/// Shortest-path hop count on the bidirectional ring.
inline int ring_hops(TileId from, TileId to) {
    int cw = ring_dist_cw(from, to);
    return cw <= NUM_TILES - cw ? cw : NUM_TILES - cw;
}

/// Deterministic direction choice: shortest path, tie -> clockwise.
inline int ring_direction(TileId from, TileId to) {
    int cw = ring_dist_cw(from, to);
    return (cw <= NUM_TILES - cw) ? 0 : 1;
}

/// Gateway tile: the tile of group `g` nearest to `from` on the ring
/// (tie -> lower tile id). Far SHARED-bank traffic rides the ring to the
/// gateway, then takes that tile's vertical hybrid bond up to the bank.
inline TileId gateway_tile(TileId from, GroupId g) {
    TileId best = static_cast<TileId>(g * TILES_PER_GROUP);
    int best_d = ring_hops(from, best);
    for (int i = 1; i < TILES_PER_GROUP; i++) {
        TileId cand = static_cast<TileId>(g * TILES_PER_GROUP + i);
        int d = ring_hops(from, cand);
        if (d < best_d) { best = cand; best_d = d; }
    }
    return best;
}

} // namespace mobol::cycle
