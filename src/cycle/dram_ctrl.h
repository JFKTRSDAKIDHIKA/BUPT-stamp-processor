/// @file dram_ctrl.h
/// @brief DRAM-die memory controller, backed by Ramulator2.
///
/// The controller sits on the top die and is fed by four per-group
/// vertical columns. Every access is a 64-byte transaction (one DRAM
/// burst). Clock domains follow the approved Q5 ratio: for every logic
/// cycle the DRAM controller advances `dram_ticks_per_logic` (default 3)
/// DRAM cycles — e.g. a 400 MHz logic die with DDR4-2400 (1200 MHz).
///
/// Functional semantics: write data commits to the backing store when the
/// controller accepts the transaction; the Ramulator2 completion callback
/// releases the write-ack. Read data is sampled from the backing store at
/// completion time and travels back down the requester's column.
#pragma once

#include "cycle/fabric.h"
#include "common/memory_model.h"

#include <deque>
#include <memory>
#include <string>

#include <yaml-cpp/yaml.h>

namespace Ramulator { class IMemorySystem; class IFrontEnd; }

namespace mobol::cycle {

class DramCtrl {
public:
    DramCtrl(const CycleConfig& cfg, MemoryModel& mem);
    ~DramCtrl();

    void tick(Fabric& fab, Cycle now);

    bool idle() const;

    // ── Stats ──
    uint64_t reads() const { return reads_; }
    uint64_t writes() const { return writes_; }
    uint64_t read_bytes() const { return read_bytes_; }
    uint64_t write_bytes() const { return write_bytes_; }
    double avg_read_latency_dram_cycles() const {
        return reads_ ? static_cast<double>(read_lat_sum_) / reads_ : 0.0;
    }
    bool using_ramulator() const { return mem_sys_ != nullptr; }
    /// logic:DRAM clock ratio, read from the Ramulator2 YAML.
    int ratio() const { return ratio_; }

private:
    struct Txn {
        Flit flit;              ///< original request flit
        bool is_write;
        Cycle enq_logic_cycle;
        // One 64 B fabric flit may span several device transactions
        // (dram_txn_bytes granularity, e.g. 32 B for HBM3-class cores).
        // The response leaves only when every line has completed.
        uint32_t next_line = 0;
        uint32_t num_lines = 1;
        std::shared_ptr<int> remaining;
    };

    const CycleConfig& cfg_;
    MemoryModel& mem_;

    // Ramulator2 (null => fixed-latency fallback model).
    // The YAML config tree must outlive the memory system: Ramulator2
    // components keep YAML::Node handles into it for their whole life.
    YAML::Node ram_config_;
    Ramulator::IMemorySystem* mem_sys_ = nullptr;
    Ramulator::IFrontEnd* frontend_ = nullptr;
    int64_t dram_clk_ = 0;
    int ratio_ = 1;                ///< DRAM ticks per logic cycle (from YAML)

    std::deque<Txn> pending_;      ///< accepted for issue, awaiting send()
    /// Completed responses awaiting their downlink, one queue per group
    /// column so a backed-up column cannot head-of-line block the others.
    std::deque<Flit> out_[NUM_BANKS];
    uint64_t outstanding_ = 0;     ///< sent to Ramulator2, callback pending
    int rr_ = 0;                   ///< round-robin over the 4 columns

    uint64_t reads_ = 0, writes_ = 0;
    uint64_t read_bytes_ = 0, write_bytes_ = 0;
    uint64_t read_lat_sum_ = 0;
    uint64_t send_rejects_ = 0;

    void init_ramulator();
    /// Try to hand the head of pending_ to the DRAM backend. Returns true
    /// if a transaction was issued.
    bool issue_one(Cycle now);
    void complete(const Txn& txn);
};

} // namespace mobol::cycle
