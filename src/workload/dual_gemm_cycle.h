/// @file dual_gemm_cycle.h
/// @brief Dual-GEMM (E = (A x B) x D, 32x32 f16) as per-tile programs for
///        the cycle-accurate simulator.
///
/// Same milestone-1 schedule as the functional engine, expressed as real
/// tile instruction streams with explicit DMA descriptors, fences and
/// release/acquire sync:
///   GEMM1: tiles 0-3 compute k=0 and reduce; tiles 4-7 compute k=1 and
///          push psums over the NoC. C (f32) lives in SHARED bank 0.
///   GEMM2: same split; E (f16) is written back to DRAM.
///
/// Note on the local memory map: unlike the sequential functional model,
/// tiles here run concurrently, so the received-psum buffer must not alias
/// the A operand (a fast producer could overwrite a slow consumer's A).
/// Every buffer gets a private range.
#pragma once

#include "cycle/isa.h"

namespace mobol::cycle {

/// Local scratchpad map for this workload (per tile, disjoint buffers).
struct DGLayout {
    static constexpr uint32_t A_OFF   = 0x0000; // 16x16 f16, 512 B
    static constexpr uint32_t B_OFF   = 0x0200; // 16x16 f16, 512 B
    static constexpr uint32_t P_OFF   = 0x0400; // psum k=own, f32, 1 KB
    static constexpr uint32_t RP_OFF  = 0x0800; // psum received, f32, 1 KB
    static constexpr uint32_t C_OFF   = 0x0C00; // C block, f32, 1 KB
    static constexpr uint32_t D_OFF   = 0x1000; // D block, f16, 512 B
    static constexpr uint32_t P2_OFF  = 0x1200; // psum2 k=own, f32, 1 KB
    static constexpr uint32_t RP2_OFF = 0x1600; // psum2 received, f32, 1 KB
    static constexpr uint32_t E32_OFF = 0x1A00; // E block f32, 1 KB
    static constexpr uint32_t E16_OFF = 0x1E00; // E block f16, 512 B
};

/// Sync tags (indices are scoped per consumer tile).
struct DGTags {
    static constexpr uint32_t PSUM1 = 0; ///< GEMM1 psum join (arity 2)
    static constexpr uint32_t C_RDY = 1; ///< C block ready (arity 1)
    static constexpr uint32_t PSUM2 = 2; ///< GEMM2 psum join (arity 2)
};

/// Build the per-tile programs. Uses the same DRAM/SHARED layout as the
/// functional workload (workload/dual_gemm.h DramLayout/SharedLayout).
/// use_shared=false models the 2-die (no buffer die) paradigm: the fused
/// C matrix round-trips DRAM instead of living in SHARED bank 0.
Program build_dual_gemm_program(bool use_shared = true);

/// DRAM offset of C in the no-buffer-die variant.
constexpr Addr DG_C_DRAM_OFF = 0x10000;

} // namespace mobol::cycle
