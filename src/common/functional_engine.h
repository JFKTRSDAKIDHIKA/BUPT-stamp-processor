/// @file functional_engine.h
/// @brief Functional Engine — pure numerical operations, no timing.
///
/// Provides DMA, MXU, and reduce operations over the MemoryModel.
/// All operations are instantaneous (no cycle concept).
/// Numerical precision: f16 I/O, f32 accumulation, f32 intermediate results.
#pragma once

#include "common/types.h"
#include "common/f16.h"
#include "common/memory_model.h"
#include "common/sync_manager.h"
#include <cstdint>

namespace mobol {

/// Functional Engine: stateless operations over a MemoryModel.
class FunctionalEngine {
public:
    explicit FunctionalEngine(MemoryModel& mem) : mem_(mem) {}

    // ─── DMA ─────────────────────────────────────────────────
    /// Copy `size` bytes from src_addr to dst_addr (any PGAS segments).
    void dma_copy(Addr src_addr, Addr dst_addr, size_t size);

    // ─── MXU ─────────────────────────────────────────────────
    /// 16×16 × 16×16 matrix multiply with f32 accumulation.
    /// Reads A block (f16, row-major) from a_addr.
    /// Reads B block (f16, row-major) from b_addr.
    /// Writes C block (f32, row-major) to c_addr.
    /// All addresses point to LOCAL scratchpad.
    void mxu_f16xf16_to_f32(Addr a_addr, Addr b_addr, Addr c_addr);

    /// 16×16 × 16×16 matrix multiply: A is f32, B is f16, output is f32.
    /// Used for GEMM2 where C (from SHARED) is f32.
    void mxu_f32xf16_to_f32(Addr a_addr, Addr b_addr, Addr c_addr);

    // ─── Reduce ──────────────────────────────────────────────
    /// Element-wise f32 addition of two 16×16 blocks.
    /// result[i][j] = block_a[i][j] + block_b[i][j]  (f32)
    /// Writes f32 result to dst_addr.
    void reduce_f32(Addr src_a_addr, Addr src_b_addr, Addr dst_addr);

    // ─── Conversion ──────────────────────────────────────────
    /// Convert a 16×16 f32 block at src to f16 and write to dst.
    void convert_f32_to_f16(Addr src_f32_addr, Addr dst_f16_addr);

private:
    MemoryModel& mem_;

    static constexpr int TILE = 16; // MXU dimension
};

} // namespace mobol
