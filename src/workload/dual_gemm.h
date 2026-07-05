/// @file dual_gemm.h
/// @brief Hardcoded dual GEMM schedule for milestone 1.
///
/// E = (A × B) × D, all 32×32 f16.
/// GEMM1: C = A × B, C stored in SHARED bank 0 as f32.
/// GEMM2: E = C × D, E written back to DRAM as f16.
///
/// Tile assignment (per UNDERSTANDING.md §8.2):
///   GEMM1: tiles 0-3 (group 0, k=0 + reducer), tiles 4-7 (group 1, k=1)
///   GEMM2: tiles 0-3 (group 0, k=0 + reducer), tiles 4-7 (group 1, k=1)
#pragma once

#include "common/types.h"
#include "common/memory_model.h"
#include "common/sync_manager.h"
#include "common/functional_engine.h"

namespace mobol {

/// DRAM layout for the dual GEMM workload.
struct DramLayout {
    static constexpr Addr A_BASE = 0x000000;  // 32×32 f16 = 2048 bytes
    static constexpr Addr B_BASE = 0x001000;  // 32×32 f16 = 2048 bytes
    static constexpr Addr D_BASE = 0x002000;  // 32×32 f16 = 2048 bytes
    static constexpr Addr E_BASE = 0x003000;  // 32×32 f16 = 2048 bytes
};

/// SHARED layout for intermediate C matrix.
struct SharedLayout {
    static constexpr BankId C_BANK = 0;
    static constexpr Addr   C_BASE_OFF = 0x000000;  // 32×32 f32 = 4096 bytes
};

/// LOCAL scratchpad layout within each tile.
struct LocalLayout {
    // GEMM1 compute tiles (k=0 and k=1)
    static constexpr Addr A_OFF  = 0x0000;  // 16×16 f16 = 512 B
    static constexpr Addr B_OFF  = 0x0200;  // 16×16 f16 = 512 B
    static constexpr Addr P_OFF  = 0x0400;  // 16×16 f32 = 1024 B (psum output)

    // GEMM1 reducer tiles
    static constexpr Addr RP_OFF = 0x0000;  // 16×16 f32 = 1024 B (received psum from k=1)
    static constexpr Addr RK0_OFF = 0x0400; // alias: own psum k=0 (same as P_OFF for k=0 tiles)
    // Note: for reducer tiles that are also k=0 compute tiles,
    // RK0 and P_OFF overlap. After k=0 MXU, psum is at 0x0400.
    // Then RP (received psum) is at 0x0000. Reduction reads both.

    // GEMM2 compute tiles
    static constexpr Addr C_OFF  = 0x0000;  // 16×16 f32 = 1024 B (C from SHARED)
    static constexpr Addr D_OFF  = 0x0400;  // 16×16 f16 = 512 B  (D from DRAM)
    static constexpr Addr P2_OFF = 0x0600;  // 16×16 f32 = 1024 B (psum output)

    // GEMM2 reducer tiles
    static constexpr Addr RP2_OFF = 0x0000; // 16×16 f32 = 1024 B (received psum k=1)
    static constexpr Addr E_F32_OFF = 0x0400; // 16×16 f32 = 1024 B (reduced)
    static constexpr Addr E_F16_OFF = 0x0800; // 16×16 f16 = 512 B  (converted)
};

/// Execute the full dual GEMM schedule on the functional engine.
/// Returns the final E matrix (32×32 f16, 2048 bytes).
/// Also writes intermediate C (32×32 f32) to SHARED bank 0.
void execute_dual_gemm(
    MemoryModel& mem,
    SyncManager& sync,
    FunctionalEngine& engine);

/// Load A, B, D from host arrays into DRAM.
void load_inputs_to_dram(
    MemoryModel& mem,
    const f16* A, const f16* B, const f16* D);

/// Read E from DRAM into a host array.
void read_output_from_dram(
    MemoryModel& mem,
    f16* E);

/// Read C from SHARED into a host array.
void read_C_from_shared(
    MemoryModel& mem,
    float* C);

// ─── Timing schedule builder ──────────────────────────────────

} // namespace mobol

#include "timing/timing_model.h"

namespace mobol {

/// Build the timing schedule (same operations as execute_dual_gemm, but as
/// a list of ScheduleEntry for the TimingSimulator).
void build_timing_schedule(std::vector<ScheduleEntry>& schedule);

} // namespace mobol
