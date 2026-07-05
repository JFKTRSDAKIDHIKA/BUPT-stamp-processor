/// @file cycle_config.h
/// @brief Configuration knobs for the cycle-accurate MOBOL simulator.
///
/// Every microarchitectural parameter of the chip is a knob here.
/// Defaults follow the approved milestone parameters (UNDERSTANDING.md Q1-Q10):
///   MXU: latency 4, initiation interval 1
///   LOCAL SRAM: 2 read ports + 1 write port, 64 B/port/cycle
///   NoC: bidirectional ring, 1 flit (64 B) per link per cycle, 1 cycle/hop
///   DMA: 1 channel/tile, 2-cycle setup
///   Clock: logic : DRAM-controller = 1 : 3 (e.g. 400 MHz logic, DDR4-2400)
#pragma once

#include "common/types.h"
#include <string>

namespace mobol::cycle {

struct CycleConfig {
    // ─── Tile compute ────────────────────────────────────────
    int mxu_latency = 4;          ///< cycles from issue to result commit
    int mxu_issue_interval = 1;   ///< min cycles between MXU issues (II)

    // The MXU/VPU stream operands through dedicated latch/double-buffer
    // paths (TPU-style). When true, they instead contend on the scratchpad
    // ports below (much slower, for sensitivity studies).
    bool compute_uses_sram_ports = false;

    // VPU (vector unit) latency table, cycles per 16x16 block op.
    // The VPU is 16 f32 lanes; one row (16 elems) per cycle for simple ops.
    int vpu_add_cycles       = 16;   ///< elementwise add, 16 rows
    int vpu_convert_cycles   = 16;   ///< f32<->f16 block convert
    int vpu_transpose_cycles = 16;   ///< 16x16 block transpose
    int vpu_scale_cycles     = 16;   ///< multiply by scalar
    int vpu_softmax_cycles   = 128;  ///< fused row-wise softmax (max/exp/sum/div)
    int vpu_gelu_cycles      = 32;   ///< tanh-approx GELU
    int vpu_layernorm_cycles = 64;   ///< per 16x16 sub-block of a fused LN
    int vpu_mul_cycles       = 16;   ///< elementwise multiply
    int vpu_silu_cycles      = 32;   ///< SiLU / swish
    int vpu_rmsnorm_cycles   = 48;   ///< per block of a fused RMSNorm
    int vpu_rope_cycles      = 24;   ///< RoPE per d_head=32 head

    // ─── Local scratchpad (per tile) ─────────────────────────
    int local_rd_ports = 2;       ///< 64B reads per cycle
    int local_wr_ports = 1;       ///< 64B writes per cycle
    int sram_port_width = 64;     ///< bytes per port access

    // ─── Shared scratchpad (buffer die banks) ────────────────
    // Q2: contention modeling defaults off (enough ports), but the
    // toggle exists and access counts are always collected.
    bool shared_port_contention = false;
    int shared_rw_ports = 4;      ///< accesses/cycle when contention enabled

    // ─── NoC ring ────────────────────────────────────────────
    int noc_flit_bytes = 64;      ///< payload bytes per flit == link width
    int noc_hop_latency = 1;      ///< cycles per hop
    int noc_buffer_depth = 4;     ///< flits per router input buffer
    int noc_inject_queue_depth = 16;

    // ─── Vertical hybrid-bond links (3D) ─────────────────────
    // One tile<->bank link per tile; one bank<->DRAM column per group.
    int vbond_latency = 1;        ///< cycles per vertical hop
    int vbond_queue_depth = 8;    ///< flits buffered per tile-link endpoint
    int vbond_flits_per_cycle = 1;///< 64 B/cycle/direction per tile link

    // Hybrid-bond density of one bank<->DRAM column, the key 3D knob.
    // The DRAM die sits directly on top: each group column is a dense
    // bond field aggregating many >=1024-bit bank interfaces. Expressed
    // as 64 B flits per logic cycle per direction:
    //   1 = 512 bits/cycle, 2 = 1024, 4 = 2048, 8 = 4096, 16 = 8192.
    int vbond_dram_flits_per_cycle = 4;
    int dram_col_queue_depth = 64; ///< buffering at each column endpoint

    // ─── DMA engine (per tile) ───────────────────────────────
    int dma_setup_cycles = 2;
    int dma_queue_depth = 16;     ///< pending descriptors
    int dma_max_inflight_chunks = 64; ///< bounded outstanding sub-transactions
    int dma_chunks_per_cycle = 1; ///< sub-transaction emission rate (base-die
                                  ///< consumption-port DSE, with wr ports &
                                  ///< tile-link width)

    // ─── DRAM (3D-stacked die on top, driven by Ramulator2) ──
    // The logic:DRAM clock ratio is taken from the Ramulator2 YAML
    // (MemorySystem.clock_ratio); this field is only the fallback used
    // for printing before init.
    int dram_ticks_per_logic = 3;
    int dram_txn_bytes = 32;      ///< device access granularity (HBM3 core:
                                  ///< 128-bit x 2n prefetch = 32 B; use 64
                                  ///< for the legacy DDR4 comparison config)
    int dram_ctrl_queue_depth = 64;
    int dram_issue_width = 8;     ///< parallel channel issues per DRAM tick
    std::string ramulator_config; ///< Ramulator2 YAML path (required)

    // ─── Near-memory compute on the buffer die (paradigm C) ──
    // A small vector engine per bank: fused LayerNorm+convert and
    // fixed-order f32 block reduction, command-driven (NMC_CMD flit),
    // completion signalled with a release token. Same 16-lane class as
    // the tile VPU unless overridden.
    bool nmc_enable = false;
    int nmc_reduce_cycles_per_block = 16;  ///< f32 add, 16x16 block
    int nmc_ln_cycles_per_block = 64;      ///< layernorm slice
    int nmc_cvt_cycles_per_block = 16;     ///< f32->f16 block
    int nmc_queue_depth = 4;

    // ─── Sync ────────────────────────────────────────────────
    int max_live_tags = 1024;     ///< join counter table capacity per tile

    // ─── Simulation control ──────────────────────────────────
    Cycle max_cycles = 200'000'000;
    Cycle deadlock_window = 1'000'000; ///< abort if no event for this many cycles
    bool collect_trace = false;   ///< emit per-event trace records
};

} // namespace mobol::cycle
