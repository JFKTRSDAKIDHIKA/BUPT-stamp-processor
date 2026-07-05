/// @file dram_ctrl.cpp
/// @brief Ramulator2-backed DRAM controller implementation.

#include "cycle/dram_ctrl.h"

#include <yaml-cpp/yaml.h>
#include "base/base.h"
#include "base/config.h"
#include "base/factory.h"
#include "base/request.h"
#include "memory_system/memory_system.h"
#include "frontend/frontend.h"

#include <iostream>
#include <stdexcept>

namespace mobol::cycle {

DramCtrl::DramCtrl(const CycleConfig& cfg, MemoryModel& mem)
    : cfg_(cfg), mem_(mem) {
    // Ramulator2 is the DRAM timing model. There is deliberately no
    // fallback: an approximate DRAM would silently invalidate every
    // cycle number this simulator produces.
    if (cfg_.ramulator_config.empty())
        throw std::runtime_error(
            "DramCtrl: a Ramulator2 YAML config is required "
            "(cfg.ramulator_config), e.g. config/ramulator_mobol.yaml");
    init_ramulator();
}

DramCtrl::~DramCtrl() {
    // Ramulator2 factory objects are heap-allocated singletons tied to
    // library-internal registries; freeing them at simulator teardown is
    // not supported by the library, so we intentionally leak them.
}

void DramCtrl::init_ramulator() {
    // The official external-driver pattern (mirrors Ramulator2's main.cpp):
    // parse the config, factory-create the GEM5 wrapper frontend and the
    // memory system, then cross-connect. Requests are submitted through
    // IFrontEnd::receive_external_requests().
    ram_config_ = Ramulator::Config::parse_config_file(cfg_.ramulator_config, {});
    frontend_ = Ramulator::Factory::create_frontend(ram_config_);
    mem_sys_ = Ramulator::Factory::create_memory_system(ram_config_);
    frontend_->connect_memory_system(mem_sys_);
    mem_sys_->connect_frontend(frontend_);
    // Single source of truth for the clock domain: the YAML's
    // MemorySystem.clock_ratio is the number of DRAM-controller cycles
    // per logic cycle (e.g. 2 for a 500 MHz base die under a 1 GHz
    // HBM3-class controller).
    ratio_ = mem_sys_->get_clock_ratio();
    if (ratio_ < 1)
        throw std::runtime_error("DramCtrl: invalid MemorySystem.clock_ratio");
}

bool DramCtrl::issue_one(Cycle now) {
    (void)now;
    if (pending_.empty()) return false;
    Txn& txn = pending_.front();

    // Split the flit into device transactions at dram_txn_bytes
    // granularity (the DRAM core's native burst, 32 B for the HBM3-class
    // model) and issue them one per controller-port cycle, in order.
    const Addr gran = static_cast<Addr>(cfg_.dram_txn_bytes);
    if (!txn.remaining) {
        Addr off = get_offset(txn.flit.addr);
        Addr first = off & ~(gran - 1);
        Addr last = (off + txn.flit.bytes - 1) & ~(gran - 1);
        txn.num_lines = static_cast<uint32_t>((last - first) / gran) + 1;
        txn.remaining = std::make_shared<int>(static_cast<int>(txn.num_lines));
    }

    Addr line = (get_offset(txn.flit.addr) & ~(gran - 1)) + txn.next_line * gran;
    int type_id = static_cast<int>(txn.is_write ? Ramulator::Request::Type::Write
                                                : Ramulator::Request::Type::Read);

    // The callback owns a copy of the transaction descriptor.
    // source_id MUST be < frontend->get_num_cores() (== 1 for the GEM5
    // wrapper): Ramulator2 indexes per-core stats vectors with it, and a
    // larger id corrupts the heap. Per-tile attribution is tracked on our
    // side, so all external traffic is presented as core 0.
    Txn txn_copy = txn;
    bool accepted = frontend_->receive_external_requests(
        type_id, static_cast<Ramulator::Addr_t>(line), /*source_id=*/0,
        [this, txn_copy](Ramulator::Request& r) {
            if (!txn_copy.is_write)
                read_lat_sum_ += static_cast<uint64_t>(r.depart - r.arrive);
            outstanding_--;
            if (--*txn_copy.remaining == 0)
                complete(txn_copy);
        });

    if (!accepted) {
        send_rejects_++;        // controller queue full: retry next cycle
        return false;
    }
    outstanding_++;
    txn.next_line++;
    if (txn.next_line >= txn.num_lines) pending_.pop_front();
    return true;
}

void DramCtrl::complete(const Txn& txn) {
    const Flit& f = txn.flit;
    Flit resp;
    resp.tile_directed = true;
    resp.target_tile = f.requester;
    resp.requester = f.requester;
    resp.dma_seq = f.dma_seq;
    resp.bytes = f.bytes;
    resp.to_bank = f.to_bank;
    if (txn.is_write) {
        resp.kind = FlitKind::WRITE_ACK;
    } else {
        resp.kind = FlitKind::READ_RESP;
        resp.resp_addr = f.resp_addr;
        resp.payload.resize(f.bytes);
        mem_.read(f.addr, resp.payload.data(), f.bytes);
    }
    // Bank-directed prefetch data returns down that bank's own column;
    // tile-directed responses descend the requester's group column.
    BankId col = f.to_bank ? get_shared_bank(f.resp_addr)
                           : tile_group(resp.target_tile);
    out_[col].push_back(std::move(resp));
}

void DramCtrl::tick(Fabric& fab, Cycle now) {
    // 1. Ingest from the four vertical columns, up to the hybrid-bond
    //    density per column per cycle (round-robin start for fairness).
    for (int i = 0; i < NUM_BANKS; i++) {
        int b = (rr_ + i) % NUM_BANKS;
        for (int k = 0; k < cfg_.vbond_dram_flits_per_cycle; k++) {
            if (pending_.size() >= static_cast<size_t>(cfg_.dram_ctrl_queue_depth))
                break;
            Flit* f = fab.vup_bank_dram[b].front_ready(now);
            if (!f) break;

            Txn txn;
            txn.flit = *f;
            txn.is_write = (f->kind == FlitKind::DATA_WRITE);
            txn.enq_logic_cycle = now;
            if (txn.is_write) {
                // Functional commit at controller accept (single point of
                // ordering for this address; the ack still waits for timing).
                mem_.write(txn.flit.addr, txn.flit.payload.data(), txn.flit.bytes);
                writes_++;
                write_bytes_ += txn.flit.bytes;
            } else {
                reads_++;
                read_bytes_ += txn.flit.bytes;
            }
            fab.vup_bank_dram[b].pop();
            pending_.push_back(std::move(txn));
        }
    }
    rr_ = (rr_ + 1) % NUM_BANKS;

    // 2. Advance the DRAM clock domain: `ratio` DRAM cycles per logic
    //    cycle. Per DRAM cycle, up to `dram_issue_width` transactions can
    //    enter the (multi-channel) controller — parallel channels each
    //    accept independently; a rejection stalls the head (FIFO port).
    for (int k = 0; k < ratio_; k++) {
        for (int j = 0; j < cfg_.dram_issue_width; j++)
            if (!issue_one(now)) break;
        mem_sys_->tick();
        dram_clk_++;
    }

    // 3. Drain completed responses down each requester column, again at
    //    the per-column hybrid-bond density.
    for (int col = 0; col < NUM_BANKS; col++) {
        for (int k = 0; k < cfg_.vbond_dram_flits_per_cycle; k++) {
            if (out_[col].empty() || fab.vdown_dram_bank[col].full()) break;
            fab.vdown_dram_bank[col].push(std::move(out_[col].front()),
                                          now + cfg_.vbond_latency);
            out_[col].pop_front();
        }
    }
}

bool DramCtrl::idle() const {
    for (const auto& q : out_)
        if (!q.empty()) return false;
    return pending_.empty() && outstanding_ == 0;
}

} // namespace mobol::cycle
