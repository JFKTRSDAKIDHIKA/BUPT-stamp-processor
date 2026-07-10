/// @file tile_core.h
/// @brief Cycle-accurate compute-tile model.
///
/// Microarchitecture per tile (base die):
///   Sequencer   : in-order, single-issue. One instruction fetched, decoded
///                 and dispatched per cycle; blocking instructions stall
///                 with an attributed reason.
///   MXU         : 16x16x16 systolic array. Pipelined: one issue per
///                 `mxu_issue_interval` cycles, results commit to LOCAL
///                 SRAM after `mxu_latency` cycles. Back-to-back
///                 accumulation into the same block is forwarded through
///                 the in-flight accumulator (no SRAM round trip), as in a
///                 real output-stationary array.
///   VPU         : 16-lane f32 vector unit. Synchronous: the sequencer
///                 stalls while a VPU op is in flight (conservative).
///   DMA engine  : one channel (Q4). Descriptor queue -> 2-cycle setup ->
///                 one 64 B sub-transaction generated per cycle. Pushes
///                 emit DATA_WRITE flits (completion by WRITE_ACK); pulls
///                 emit READ_REQ flits (completion when READ_RESP data has
///                 been committed to LOCAL SRAM).
///   LOCAL SRAM  : 2 read + 1 write ports, 64 B each, arbitrated per cycle
///                 between DMA reads, fabric ingress writes and the remote
///                 read service. MXU/VPU stream operands via dedicated
///                 latch paths (config: compute_uses_sram_ports).
///   Join table  : cnt[tag] counters; RELEASE tokens (local or via NoC)
///                 increment, ACQUIRE blocks until cnt >= arity then
///                 consumes arity (generation auto-reset).
#pragma once

#include "cycle/isa.h"
#include "cycle/fabric.h"
#include "cycle/cycle_config.h"
#include "common/memory_model.h"

#include <deque>
#include <map>
#include <unordered_map>
#include <vector>

namespace mobol::cycle {

/// Chip-wide event log used for causality verification and tracing.
struct EventLog {
    struct Sync { TileId tile; TileId consumer; uint32_t tag; Cycle cycle; uint32_t arity; };
    std::vector<Sync> releases;        ///< release issued (producer side)
    std::vector<Sync> token_arrivals;  ///< counter increment (consumer side)
    std::vector<Sync> acquire_passes;
    struct Dma { TileId tile; uint32_t seq; Cycle start; Cycle complete; uint64_t bytes; };
    std::vector<Dma> dmas;
    uint64_t mxu_ops = 0;
};

struct TileStats {
    uint64_t instrs = 0;
    Cycle busy_cycles = 0;         ///< cycles an instruction issued
    Cycle stall_acquire = 0;
    Cycle stall_dma = 0;           ///< DMA_FENCE / queue-full / sync ld-st
    Cycle stall_mxu = 0;
    Cycle stall_vpu = 0;
    Cycle stall_inject = 0;        ///< fabric backpressure on release
    Cycle idle_after_halt = 0;
    uint64_t mxu_ops = 0;
    Cycle mxu_busy_cycles = 0;     ///< cycles with >=1 op in the MXU pipe
    Cycle vpu_busy_cycles = 0;
    Cycle dma_busy_cycles = 0;     ///< cycles the DMA engine was moving data
    Cycle mxu_dma_overlap_cycles = 0; ///< cycles MXU busy AND DMA busy
    uint64_t dma_descriptors = 0;
    uint64_t dma_bytes = 0;
    uint64_t sram_ingress_writes = 0;
    uint64_t sram_contention = 0;  ///< port-starved requests this tile saw
};

class TileCore {
public:
    TileCore(TileId id, const CycleConfig& cfg, MemoryModel& mem,
             Fabric& fab, EventLog& events);

    void set_program(const std::vector<Instr>* prog) { prog_ = prog; }

    /// Advance one logic cycle. Returns true if any progress was made.
    bool tick(Cycle now);

    bool halted() const { return halted_; }
    /// Fully drained: halted, no in-flight MXU/VPU/DMA work.
    bool idle() const;

    const TileStats& stats() const { return stats_; }

    /// Debug: describe the current blocking condition (deadlock dumps).
    std::string state_string() const;

private:
    // ── Identity / shared structures ──
    TileId id_;
    const CycleConfig& cfg_;
    MemoryModel& mem_;
    Fabric& fab_;
    EventLog& ev_;

    // ── Sequencer ──
    const std::vector<Instr>* prog_ = nullptr;
    size_t pc_ = 0;
    bool halted_ = false;

    // ── MXU pipeline ──
    struct MxuInflight {
        Cycle commit_at;
        uint32_t d_off;
        std::vector<float> data;   ///< 256 f32 results
    };
    std::deque<MxuInflight> mxu_pipe_;
    Cycle mxu_next_issue_ = 0;

    // ── VPU ──
    struct VpuStaged {
        Cycle commit_at;
        Addr dst;                  ///< full PGAS local address
        std::vector<uint8_t> data;
    };
    std::deque<VpuStaged> vpu_staged_;
    Cycle vpu_busy_until_ = 0;

    // ── DMA engine ──
    struct DmaDesc {
        uint32_t seq;
        Addr src, dst;
        uint32_t rows, row_bytes;
        int64_t src_stride, dst_stride;
        bool sync;                 ///< sequencer blocks until complete
        uint32_t cur_row = 0, cur_off = 0;
        int setup_remaining = 0;
        Cycle enq_cycle = 0;
        uint64_t total_bytes = 0;
    };
    std::deque<DmaDesc> dma_q_;
    bool dma_active_ = false;
    DmaDesc dma_cur_;
    struct DescState { uint32_t outstanding = 0; bool gen_done = false;
                       bool done = false; Cycle start = 0; uint64_t bytes = 0; };
    // Ordered by descriptor seq so descriptors retire in strict issue order
    // (a `DMA_FENCE keep=N` then sees a monotone oldest-first outstanding
    // set, which is what makes double-buffer prefetch correct).
    std::map<uint32_t, DescState> live_descs_;
    uint32_t inflight_chunks_ = 0;  ///< chip-bound sub-txns awaiting ack/data
    uint32_t next_seq_ = 1;
    uint32_t sync_wait_seq_ = 0;   ///< nonzero: sequencer blocked on this DMA

    // ── Ingress / remote read service ──
    std::deque<Flit> read_service_q_;

    // ── Sync ──
    std::unordered_map<uint32_t, uint32_t> join_cnt_;

    // ── Per-cycle port budgets ──
    int rd_budget_ = 0, wr_budget_ = 0;

    TileStats stats_;
    bool progressed_ = false;

    // ── Stages ──
    void commit_mxu(Cycle now);
    void commit_vpu(Cycle now);
    void process_vertical_down(Cycle now);
    void process_ingress(Cycle now);
    void process_ingress_queue(FlitQueue& q, Cycle now);
    void process_read_service(Cycle now);
    void dma_engine_tick(Cycle now);
    void sequencer_tick(Cycle now);

    // ── Helpers ──
    bool issue_instr(const Instr& in, Cycle now); ///< true if issued (pc++)
    void enqueue_dma(const Instr& in, bool sync);
    bool emit_dma_chunk(Cycle now);   ///< one 64B sub-transaction
    void desc_dec_outstanding(uint32_t seq, Cycle now);
    void desc_check_done(uint32_t seq, Cycle now);
    bool dma_all_done() const;

    void mxu_issue(const Instr& in, Cycle now);
    void vpu_issue(const Instr& in, Cycle now);

    uint8_t* local_bytes(uint32_t off, size_t size);
};

} // namespace mobol::cycle
