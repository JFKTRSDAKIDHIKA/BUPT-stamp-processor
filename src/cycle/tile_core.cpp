/// @file tile_core.cpp
/// @brief Cycle-accurate compute-tile implementation.

#include "cycle/tile_core.h"
#include "cycle/blockops.h"

#include <cassert>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace mobol::cycle {

TileCore::TileCore(TileId id, const CycleConfig& cfg, MemoryModel& mem,
                   Fabric& fab, EventLog& events)
    : id_(id), cfg_(cfg), mem_(mem), fab_(fab), ev_(events) {}

uint8_t* TileCore::local_bytes(uint32_t off, size_t size) {
    (void)size;
    return mem_.local_ptr(make_local_addr(id_, off));
}

bool TileCore::idle() const {
    return halted_ && mxu_pipe_.empty() && vpu_staged_.empty()
        && !dma_active_ && dma_q_.empty() && live_descs_.empty()
        && read_service_q_.empty() && sync_wait_seq_ == 0;
}

// ═══ Main per-cycle update ═══════════════════════════════════

bool TileCore::tick(Cycle now) {
    progressed_ = false;
    rd_budget_ = cfg_.local_rd_ports;
    wr_budget_ = cfg_.local_wr_ports;

    commit_mxu(now);
    commit_vpu(now);
    process_vertical_down(now);
    process_ingress(now);
    process_read_service(now);
    dma_engine_tick(now);
    sequencer_tick(now);

    if (!mxu_pipe_.empty()) stats_.mxu_busy_cycles++;
    if (vpu_busy_until_ > now) stats_.vpu_busy_cycles++;
    // Overlap accounting (double-buffer effectiveness): a cycle where the
    // DMA engine is moving data AND the MXU pipe is non-empty is compute
    // overlapping memory. dma_busy = steady-state DMA-engine occupancy.
    if (dma_active_) {
        stats_.dma_busy_cycles++;
        if (!mxu_pipe_.empty()) stats_.mxu_dma_overlap_cycles++;
    }
    return progressed_;
}

// ═══ MXU ═════════════════════════════════════════════════════

void TileCore::mxu_issue(const Instr& in, Cycle now) {
    bool a_is_f32 = (in.op == Op::MXU_F32F16);
    size_t a_bytes = a_is_f32 ? 1024 : 512;
    const uint8_t* A = local_bytes(in.a_off, a_bytes);
    const uint8_t* B = local_bytes(in.b_off, 512);

    std::vector<float> C(256, 0.0f);
    if (in.acc) {
        // In-flight accumulator forwarding: chained accumulation into the
        // same block reads the newest pipeline value, not stale SRAM.
        const MxuInflight* fwd = nullptr;
        for (const auto& m : mxu_pipe_)
            if (m.d_off == in.d_off) fwd = &m;
        if (fwd) std::memcpy(C.data(), fwd->data.data(), 1024);
        else     std::memcpy(C.data(), local_bytes(in.d_off, 1024), 1024);
    }
    blockops::mxu(A, a_is_f32, B, C.data(), in.acc);

    mxu_pipe_.push_back({now + cfg_.mxu_latency, in.d_off, std::move(C)});
    mxu_next_issue_ = now + cfg_.mxu_issue_interval;
    stats_.mxu_ops++;
    ev_.mxu_ops++;
}

void TileCore::commit_mxu(Cycle now) {
    while (!mxu_pipe_.empty() && mxu_pipe_.front().commit_at <= now) {
        if (cfg_.compute_uses_sram_ports) {
            if (wr_budget_ <= 0) { stats_.sram_contention++; break; }
            wr_budget_--;
        }
        auto& m = mxu_pipe_.front();
        mem_.write(make_local_addr(id_, m.d_off), m.data.data(), 1024);
        mxu_pipe_.pop_front();
        progressed_ = true;
    }
}

// ═══ VPU ═════════════════════════════════════════════════════

void TileCore::vpu_issue(const Instr& in, Cycle now) {
    namespace bo = blockops;
    std::vector<uint8_t> out;
    int lat = 0;

    auto a_f32 = [&] { return reinterpret_cast<const float*>(local_bytes(in.a_off, 1024)); };
    auto b_f32 = [&] { return reinterpret_cast<const float*>(local_bytes(in.b_off, 1024)); };
    auto a_f16 = [&] { return reinterpret_cast<const f16*>(local_bytes(in.a_off, 512)); };
    auto b_f16 = [&] { return reinterpret_cast<const f16*>(local_bytes(in.b_off, 512)); };

    switch (in.op) {
        case Op::VPU_ADD_F32: {
            out.resize(1024); lat = cfg_.vpu_add_cycles;
            bo::add_f32(a_f32(), b_f32(), reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_ADD_F16: {
            out.resize(512); lat = cfg_.vpu_add_cycles;
            bo::add_f16(a_f16(), b_f16(), reinterpret_cast<f16*>(out.data()));
            break;
        }
        case Op::VPU_CVT_F32_F16: {
            out.resize(512); lat = cfg_.vpu_convert_cycles;
            bo::cvt_f32_f16(a_f32(), reinterpret_cast<f16*>(out.data()));
            break;
        }
        case Op::VPU_CVT_F16_F32: {
            out.resize(1024); lat = cfg_.vpu_convert_cycles;
            bo::cvt_f16_f32(a_f16(), reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_TRANS_F16: {
            out.resize(512); lat = cfg_.vpu_transpose_cycles;
            bo::transpose_f16(a_f16(), reinterpret_cast<f16*>(out.data()));
            break;
        }
        case Op::VPU_SCALE_F32: {
            out.resize(1024); lat = cfg_.vpu_scale_cycles;
            bo::scale_f32(a_f32(), in.scalar, reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_SOFTMAX_F32: {
            out.resize(1024); lat = cfg_.vpu_softmax_cycles;
            bo::softmax_f32(a_f32(), in.scalar, reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_GELU_F32: {
            out.resize(1024); lat = cfg_.vpu_gelu_cycles;
            bo::gelu_f32(a_f32(), reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_LAYERNORM_F32: {
            size_t bytes = 1024 * in.count;
            out.resize(bytes); lat = cfg_.vpu_layernorm_cycles * static_cast<int>(in.count);
            const float* src = reinterpret_cast<const float*>(local_bytes(in.a_off, bytes));
            bo::layernorm_f32(src, static_cast<int>(in.count), in.scalar,
                              reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_MUL_F32: {
            out.resize(1024); lat = cfg_.vpu_mul_cycles;
            bo::mul_f32(a_f32(), b_f32(), reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_SILU_F32: {
            out.resize(1024); lat = cfg_.vpu_silu_cycles;
            bo::silu_f32(a_f32(), reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_RMSNORM_BLK: {
            size_t bytes = 1024 * in.count;
            out.resize(bytes); lat = cfg_.vpu_rmsnorm_cycles * static_cast<int>(in.count);
            const float* src = reinterpret_cast<const float*>(local_bytes(in.a_off, bytes));
            bo::rmsnorm_blk(src, static_cast<int>(in.count), in.scalar,
                            reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_SOFTMAX_BLK: {
            size_t bytes = 1024 * in.count;
            out.resize(bytes);
            lat = cfg_.vpu_softmax_cycles * static_cast<int>(in.count);
            const float* src = reinterpret_cast<const float*>(local_bytes(in.a_off, bytes));
            bo::softmax_blk(src, static_cast<int>(in.count), in.scalar,
                            static_cast<int>(in.row_bytes), in.aux,
                            static_cast<int>(in.rows),
                            reinterpret_cast<float*>(out.data()));
            break;
        }
        case Op::VPU_ROPE_F16: {
            out.resize(512); lat = cfg_.vpu_rope_cycles;
            const f16* x = reinterpret_cast<const f16*>(local_bytes(in.a_off, 512));
            const float* cosb = reinterpret_cast<const float*>(
                local_bytes(static_cast<uint32_t>(in.aux), 1024));
            const float* sinb = reinterpret_cast<const float*>(
                local_bytes(static_cast<uint32_t>(in.aux) + 1024, 1024));
            bo::rope_block_f16(x, cosb, sinb, static_cast<int>(in.count),
                               reinterpret_cast<f16*>(out.data()));
            break;
        }
        default:
            throw std::runtime_error("vpu_issue: not a VPU opcode");
    }

    vpu_busy_until_ = now + lat;
    vpu_staged_.push_back({now + lat, make_local_addr(id_, in.d_off), std::move(out)});
}

void TileCore::commit_vpu(Cycle now) {
    while (!vpu_staged_.empty() && vpu_staged_.front().commit_at <= now) {
        auto& s = vpu_staged_.front();
        mem_.write(s.dst, s.data.data(), s.data.size());
        vpu_staged_.pop_front();
        progressed_ = true;
    }
}

// ═══ Fabric ingress ══════════════════════════════════════════

void TileCore::process_vertical_down(Cycle now) {
    for (int k = 0; k < cfg_.vbond_flits_per_cycle; k++) {
        Flit* f = fab_.vdown_bank_tile[id_].front_ready(now);
        if (!f) break;
        if (!f->tile_directed)
            throw std::runtime_error("tile: non-tile-directed flit on down-link");
        if (f->target_tile == id_) {
            FlitQueue& q = fab_.tile_ingress[id_][flit_vc(f->kind)];
            if (q.full()) break;
            Flit copy = *f;
            q.push(std::move(copy), now + 1);
        } else {
            // Gateway role: bank dropped a far-tile response here; relay
            // it over the ring.
            Flit copy = *f;
            if (!fab_.route_from_tile(id_, std::move(copy), now)) break;
        }
        fab_.vdown_bank_tile[id_].pop();
        progressed_ = true;
    }
}

void TileCore::process_ingress(Cycle now) {
    // Responses first: they free resources and are always consumable, so
    // a stalled DATA_WRITE can never starve the acks/tokens behind it.
    process_ingress_queue(fab_.tile_ingress[id_][VC_RESP], now);
    process_ingress_queue(fab_.tile_ingress[id_][VC_REQ], now);
}

void TileCore::process_ingress_queue(FlitQueue& q, Cycle now) {
    for (int k = 0; k < 16; k++) {
        Flit* f = q.front_ready(now);
        if (!f) break;

        switch (f->kind) {
            case FlitKind::DATA_WRITE: {
                // Remote push into my LOCAL SRAM: needs a write port, and
                // an ack must be routable back to the producer.
                if (wr_budget_ <= 0) { stats_.sram_contention++; return; }
                assert(get_segment(f->addr) == Segment::LOCAL
                       && get_local_tile(f->addr) == id_);
                assert(f->requester != id_ && "self push must be folded");
                Flit ack;
                ack.kind = FlitKind::WRITE_ACK;
                ack.tile_directed = true;
                ack.target_tile = f->requester;
                ack.requester = f->requester;
                ack.dma_seq = f->dma_seq;
                ack.bytes = f->bytes;
                if (!fab_.route_from_tile(id_, std::move(ack), now)) return;
                mem_.write(f->addr, f->payload.data(), f->bytes);
                wr_budget_--;
                stats_.sram_ingress_writes++;
                break;
            }
            case FlitKind::READ_RESP: {
                if (wr_budget_ <= 0) { stats_.sram_contention++; return; }
                assert(get_segment(f->resp_addr) == Segment::LOCAL
                       && get_local_tile(f->resp_addr) == id_);
                mem_.write(f->resp_addr, f->payload.data(), f->bytes);
                wr_budget_--;
                desc_dec_outstanding(f->dma_seq, now);
                break;
            }
            case FlitKind::WRITE_ACK: {
                desc_dec_outstanding(f->dma_seq, now);
                break;
            }
            case FlitKind::RELEASE_TOK: {
                if (join_cnt_.size() >= static_cast<size_t>(cfg_.max_live_tags)
                    && !join_cnt_.count(f->tag))
                    throw std::runtime_error("join counter table overflow");
                join_cnt_[f->tag]++;
                ev_.token_arrivals.push_back({f->requester, id_, f->tag, now, 0});
                break;
            }
            case FlitKind::READ_REQ: {
                // Remote pull from my LOCAL SRAM: queue for the read
                // service unit.
                if (read_service_q_.size() >= 8) return;
                read_service_q_.push_back(*f);
                break;
            }
            case FlitKind::NMC_CMD:
                throw std::runtime_error("tile: NMC_CMD cannot target a tile");
        }
        q.pop();
        progressed_ = true;
    }
}

void TileCore::process_read_service(Cycle now) {
    if (read_service_q_.empty()) return;
    if (rd_budget_ <= 0) { stats_.sram_contention++; return; }
    Flit& req = read_service_q_.front();
    Flit resp;
    resp.kind = FlitKind::READ_RESP;
    resp.tile_directed = true;
    resp.target_tile = req.requester;
    resp.requester = req.requester;
    resp.dma_seq = req.dma_seq;
    resp.resp_addr = req.resp_addr;
    resp.bytes = req.bytes;
    resp.payload.resize(req.bytes);
    mem_.read(req.addr, resp.payload.data(), req.bytes);
    if (!fab_.route_from_tile(id_, std::move(resp), now)) return;
    rd_budget_--;
    read_service_q_.pop_front();
    progressed_ = true;
}

// ═══ DMA engine ══════════════════════════════════════════════

void TileCore::enqueue_dma(const Instr& in, bool sync) {
    DmaDesc d;
    d.seq = next_seq_++;
    d.src = in.src;
    d.dst = in.dst;
    d.rows = in.rows;
    d.row_bytes = in.row_bytes;
    d.src_stride = in.src_stride;
    d.dst_stride = in.dst_stride;
    d.sync = sync;
    if (d.rows == 0 || d.row_bytes == 0)
        throw std::runtime_error("DMA descriptor with empty shape");
    live_descs_[d.seq] = DescState{};
    live_descs_[d.seq].start = 0; // patched on first chunk
    dma_q_.push_back(d);
    stats_.dma_descriptors++;
    if (sync) sync_wait_seq_ = d.seq;
}

bool TileCore::emit_dma_chunk(Cycle now) {
    DmaDesc& d = dma_cur_;
    Addr sa = static_cast<Addr>(static_cast<int64_t>(d.src)
              + static_cast<int64_t>(d.cur_row) * d.src_stride + d.cur_off);
    Addr da = static_cast<Addr>(static_cast<int64_t>(d.dst)
              + static_cast<int64_t>(d.cur_row) * d.dst_stride + d.cur_off);
    uint32_t len = std::min<uint32_t>(static_cast<uint32_t>(cfg_.noc_flit_bytes),
                                      d.row_bytes - d.cur_off);
    // Keep DRAM-bound chunks aligned to flit-sized lines so each flit
    // maps to whole device transactions (the DRAM controller splits a
    // flit into dram_txn_bytes-granularity accesses).
    const Addr fl = static_cast<Addr>(cfg_.noc_flit_bytes);
    if (get_segment(sa) == Segment::DRAM)
        len = std::min<uint32_t>(len, static_cast<uint32_t>(fl - (get_offset(sa) & (fl - 1))));
    if (get_segment(da) == Segment::DRAM)
        len = std::min<uint32_t>(len, static_cast<uint32_t>(fl - (get_offset(da) & (fl - 1))));

    Segment ss = get_segment(sa), ds = get_segment(da);
    bool src_self = (ss == Segment::LOCAL && get_local_tile(sa) == id_);
    bool dst_self = (ds == Segment::LOCAL && get_local_tile(da) == id_);

    // Bounded outstanding: chip-bound sub-transactions are capped so the
    // fabric can never be flooded beyond drainable occupancy.
    if (!(src_self && dst_self)
        && inflight_chunks_ >= static_cast<uint32_t>(cfg_.dma_max_inflight_chunks))
        return false;

    auto& st = live_descs_[d.seq];

    if (src_self && dst_self) {
        // Self-alias fast path (spec §4): pure SRAM copy, no NoC.
        if (rd_budget_ <= 0 || wr_budget_ <= 0) { stats_.sram_contention++; return false; }
        uint8_t buf[64];
        mem_.read(sa, buf, len);
        mem_.write(da, buf, len);
        rd_budget_--; wr_budget_--;
    } else if (src_self) {
        // Push: read local, emit DATA_WRITE flit toward `da`.
        if (rd_budget_ <= 0) { stats_.sram_contention++; return false; }
        Flit f;
        f.kind = FlitKind::DATA_WRITE;
        f.requester = id_;
        f.dma_seq = d.seq;
        f.addr = da;
        f.bytes = len;
        f.payload.resize(len);
        mem_.read(sa, f.payload.data(), len);
        if (!fab_.route_from_tile(id_, std::move(f), now)) return false;
        rd_budget_--;
        st.outstanding++;
        inflight_chunks_++;
    } else if (dst_self) {
        // Pull: emit READ_REQ toward `sa`; data lands via READ_RESP.
        Flit f;
        f.kind = FlitKind::READ_REQ;
        f.requester = id_;
        f.dma_seq = d.seq;
        f.addr = sa;
        f.resp_addr = da;
        f.bytes = len;
        if (!fab_.route_from_tile(id_, std::move(f), now)) return false;
        st.outstanding++;
        inflight_chunks_++;
    } else {
        std::ostringstream os;
        os << "DMA descriptor touches no local endpoint: tile" << int(id_)
           << " seq=" << d.seq << " src=0x" << std::hex << d.src
           << " dst=0x" << d.dst << " sa=0x" << sa << " da=0x" << da
           << std::dec << " row=" << d.cur_row << "/" << d.rows
           << " off=" << d.cur_off << "/" << d.row_bytes
           << " sstride=" << d.src_stride << " dstride=" << d.dst_stride;
        throw std::runtime_error(os.str());
    }

    if (st.bytes == 0) st.start = now;
    st.bytes += len;
    stats_.dma_bytes += len;

    d.cur_off += len;
    if (d.cur_off >= d.row_bytes) { d.cur_off = 0; d.cur_row++; }
    if (d.cur_row >= d.rows) {
        st.gen_done = true;
        dma_active_ = false;
        desc_check_done(d.seq, now);
    }
    return true;
}

void TileCore::dma_engine_tick(Cycle now) {
    if (!dma_active_ && !dma_q_.empty()) {
        dma_cur_ = dma_q_.front();
        dma_q_.pop_front();
        dma_cur_.setup_remaining = cfg_.dma_setup_cycles;
        dma_active_ = true;
        progressed_ = true;
    }
    if (!dma_active_) return;
    if (dma_cur_.setup_remaining > 0) {
        dma_cur_.setup_remaining--;
        progressed_ = true;
        return;
    }
    // Emission rate: consumption-port DSE knob (default 1 chunk/cycle).
    for (int k = 0; k < cfg_.dma_chunks_per_cycle && dma_active_; k++) {
        if (!emit_dma_chunk(now)) break;
        progressed_ = true;
    }
}

void TileCore::desc_dec_outstanding(uint32_t seq, Cycle now) {
    auto it = live_descs_.find(seq);
    if (it == live_descs_.end())
        throw std::runtime_error("completion for unknown DMA descriptor");
    if (it->second.outstanding == 0 || inflight_chunks_ == 0)
        throw std::runtime_error("DMA outstanding underflow");
    it->second.outstanding--;
    inflight_chunks_--;
    desc_check_done(seq, now);
}

void TileCore::desc_check_done(uint32_t seq, Cycle now) {
    auto it = live_descs_.find(seq);
    if (it == live_descs_.end()) return;
    if (!it->second.gen_done || it->second.outstanding != 0) return;
    it->second.done = true;
    // Retire in strict issue order (live_descs_ is seq-ordered): a later
    // descriptor that finishes early waits for all earlier ones to retire
    // first, so the outstanding count is always the N most-recently-issued.
    for (auto f = live_descs_.begin(); f != live_descs_.end() && f->second.done; ) {
        ev_.dmas.push_back({id_, f->first, f->second.start, now, f->second.bytes});
        f = live_descs_.erase(f);
    }
}

bool TileCore::dma_all_done() const {
    return !dma_active_ && dma_q_.empty() && live_descs_.empty();
}

// ═══ Sequencer ═══════════════════════════════════════════════

void TileCore::sequencer_tick(Cycle now) {
    if (halted_) { stats_.idle_after_halt++; return; }
    if (!prog_ || pc_ >= prog_->size()) { halted_ = true; return; }
    if (vpu_busy_until_ > now) { stats_.stall_vpu++; return; }
    if (sync_wait_seq_ != 0) {
        if (live_descs_.count(sync_wait_seq_)) { stats_.stall_dma++; return; }
        sync_wait_seq_ = 0;
    }
    const Instr& in = (*prog_)[pc_];
    if (issue_instr(in, now)) {
        pc_++;
        stats_.instrs++;
        stats_.busy_cycles++;
        progressed_ = true;
    }
}

bool TileCore::issue_instr(const Instr& in, Cycle now) {
    switch (in.op) {
        case Op::NOP:
            return true;

        case Op::DMA:
            if (dma_q_.size() >= static_cast<size_t>(cfg_.dma_queue_depth)) {
                stats_.stall_dma++;
                return false;
            }
            enqueue_dma(in, false);
            return true;

        case Op::LOAD_SHARED: {
            if (get_segment(in.src) != Segment::SHARED || !is_shared_near(in.src, id_))
                throw std::runtime_error("LOAD_SHARED: not the near bank (trap)");
            if (dma_q_.size() >= static_cast<size_t>(cfg_.dma_queue_depth)) {
                stats_.stall_dma++;
                return false;
            }
            enqueue_dma(in, true);
            return true;
        }
        case Op::STORE_SHARED: {
            if (get_segment(in.dst) != Segment::SHARED || !is_shared_near(in.dst, id_))
                throw std::runtime_error("STORE_SHARED: not the near bank (trap)");
            if (dma_q_.size() >= static_cast<size_t>(cfg_.dma_queue_depth)) {
                stats_.stall_dma++;
                return false;
            }
            enqueue_dma(in, true);
            return true;
        }

        case Op::DMA_FENCE:
            // Partial drain: block until <= keep descriptors remain (keep=0
            // is the classic full fence). live_descs_ holds every not-yet-
            // complete descriptor, so its size is the outstanding count.
            if (live_descs_.size() > in.keep) { stats_.stall_dma++; return false; }
            return true;

        case Op::MXU_F16F16:
        case Op::MXU_F32F16: {
            if (now < mxu_next_issue_) { stats_.stall_mxu++; return false; }
            if (cfg_.compute_uses_sram_ports) {
                if (rd_budget_ < 2) { stats_.sram_contention++; stats_.stall_mxu++; return false; }
                rd_budget_ -= 2;
            }
            mxu_issue(in, now);
            return true;
        }

        case Op::WAIT_MXU:
            if (!mxu_pipe_.empty()) { stats_.stall_mxu++; return false; }
            return true;

        case Op::VPU_ADD_F32:
        case Op::VPU_ADD_F16:
        case Op::VPU_CVT_F32_F16:
        case Op::VPU_CVT_F16_F32:
        case Op::VPU_TRANS_F16:
        case Op::VPU_SCALE_F32:
        case Op::VPU_SOFTMAX_F32:
        case Op::VPU_GELU_F32:
        case Op::VPU_LAYERNORM_F32:
        case Op::VPU_MUL_F32:
        case Op::VPU_SILU_F32:
        case Op::VPU_RMSNORM_BLK:
        case Op::VPU_SOFTMAX_BLK:
        case Op::VPU_ROPE_F16:
            vpu_issue(in, now);
            return true;

        case Op::NMC: {
            // Commands may target any bank (far commands ride the ring
            // to the gateway like any SHARED request); the data stays
            // beside that bank's SRAM — that is the whole point.
            Flit cmd;
            cmd.kind = FlitKind::NMC_CMD;
            cmd.requester = id_;
            cmd.nmc_op = static_cast<NmcOp>(in.arity);
            cmd.nmc_count = in.count;
            cmd.nmc_scalar = in.scalar;
            cmd.tag = in.tag;
            if (cmd.nmc_op == NmcOp::PREFETCH) {
                // Route by the DESTINATION bank region; source is DRAM.
                if (get_segment(in.dst) != Segment::SHARED
                    || get_segment(in.src) != Segment::DRAM)
                    throw std::runtime_error("NMC PREFETCH: needs DRAM src, SHARED dst");
                cmd.addr = in.dst;
                cmd.nmc_src = in.src;
                cmd.nmc_row_bytes = in.row_bytes;
                cmd.nmc_sstride = in.src_stride;
                cmd.nmc_dstride = in.dst_stride;
                cmd.nmc_count = in.rows;
            } else {
                if (get_segment(in.src) != Segment::SHARED)
                    throw std::runtime_error("NMC: source must be a SHARED region");
                cmd.addr = in.src;
                cmd.resp_addr = in.dst;
            }
            if (!fab_.route_from_tile(id_, std::move(cmd), now)) {
                stats_.stall_inject++;
                return false;
            }
            return true;
        }

        case Op::RELEASE: {
            if (in.consumer == id_) {
                // Self-release folds to a local counter bump (spec §6.1).
                join_cnt_[in.tag]++;
                ev_.releases.push_back({id_, id_, in.tag, now, 0});
                ev_.token_arrivals.push_back({id_, id_, in.tag, now, 0});
                return true;
            }
            Flit tok;
            tok.kind = FlitKind::RELEASE_TOK;
            tok.tile_directed = true;
            tok.target_tile = in.consumer;
            tok.requester = id_;
            tok.tag = in.tag;
            tok.bytes = 0;
            if (!fab_.route_from_tile(id_, std::move(tok), now)) {
                stats_.stall_inject++;
                return false;
            }
            ev_.releases.push_back({id_, in.consumer, in.tag, now, 0});
            return true;
        }

        case Op::ACQUIRE: {
            uint32_t& cnt = join_cnt_[in.tag];
            if (cnt < in.arity) { stats_.stall_acquire++; return false; }
            cnt -= in.arity;   // consume this generation
            ev_.acquire_passes.push_back({id_, id_, in.tag, now, in.arity});
            return true;
        }

        case Op::HALT:
            halted_ = true;
            return true;
    }
    throw std::runtime_error("unknown opcode");
}

// ═══ Debug ═══════════════════════════════════════════════════

std::string TileCore::state_string() const {
    std::ostringstream os;
    os << "tile" << static_cast<int>(id_)
       << " pc=" << pc_ << "/" << (prog_ ? prog_->size() : 0)
       << (halted_ ? " HALTED" : "");
    if (prog_ && pc_ < prog_->size()) {
        const Instr& in = (*prog_)[pc_];
        os << " next_op=" << static_cast<int>(in.op)
           << " tag=" << in.tag << " arity=" << in.arity;
        auto it = join_cnt_.find(in.tag);
        os << " cnt=" << (it != join_cnt_.end() ? it->second : 0);
        if (!in.note.empty()) os << " note=" << in.note;
    }
    os << " dma_q=" << dma_q_.size() << (dma_active_ ? "+active" : "")
       << " live_descs=" << live_descs_.size()
       << " mxu_pipe=" << mxu_pipe_.size()
       << " sync_wait=" << sync_wait_seq_;
    return os.str();
}

} // namespace mobol::cycle
