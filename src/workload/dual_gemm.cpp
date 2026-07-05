/// @file dual_gemm.cpp
/// @brief Hardcoded dual GEMM tile schedule execution.

#include "workload/dual_gemm.h"
#include <cassert>
#include <cstring>
#include <iostream>

namespace mobol {

// ─── Helper: copy matrix blocks between strided and contiguous ──

/// Copy a 16×16 block from row-major matrix in DRAM to contiguous LOCAL.
/// The source is strided (stride = 32 elements), dest is contiguous (stride = 16).
static void dma_matrix_block_to_local(
    FunctionalEngine& engine, MemoryModel& mem,
    Addr dram_base, int block_row, int block_col,
    TileId dst_tile, Addr dst_local_off,
    size_t elem_size)
{
    for (int r = 0; r < 16; r++) {
        Addr src = make_dram_addr(dram_base +
            static_cast<Addr>(((block_row * 16 + r) * 32 + block_col * 16) * elem_size));
        Addr dst = make_local_addr(dst_tile, dst_local_off + static_cast<Addr>(r * 16 * elem_size));
        engine.dma_copy(src, dst, 16 * elem_size);
    }
}

/// Copy a contiguous 16×16 block from LOCAL to row-major matrix in DRAM.
static void dma_local_to_matrix_block(
    FunctionalEngine& engine, MemoryModel& mem,
    TileId src_tile, Addr src_local_off,
    Addr dram_base, int block_row, int block_col,
    size_t elem_size)
{
    for (int r = 0; r < 16; r++) {
        Addr src = make_local_addr(src_tile, src_local_off + static_cast<Addr>(r * 16 * elem_size));
        Addr dst = make_dram_addr(dram_base +
            static_cast<Addr>(((block_row * 16 + r) * 32 + block_col * 16) * elem_size));
        engine.dma_copy(src, dst, 16 * elem_size);
    }
}

/// Copy a contiguous 16×16 f32 block from LOCAL to row-major matrix in SHARED.
static void dma_local_to_shared_block(
    FunctionalEngine& engine,
    TileId src_tile, Addr src_local_off,
    BankId bank, Addr shared_base_off, int block_row, int block_col)
{
    for (int r = 0; r < 16; r++) {
        Addr src = make_local_addr(src_tile, src_local_off + static_cast<Addr>(r * 16 * 4));
        Addr dst = make_shared_addr(bank,
            shared_base_off + static_cast<Addr>(((block_row * 16 + r) * 32 + block_col * 16) * 4));
        engine.dma_copy(src, dst, 16 * 4);
    }
}

/// Copy a 16×16 f32 block from row-major SHARED to contiguous LOCAL.
static void dma_shared_block_to_local(
    FunctionalEngine& engine,
    BankId bank, Addr shared_base_off, int block_row, int block_col,
    TileId dst_tile, Addr dst_local_off)
{
    for (int r = 0; r < 16; r++) {
        Addr src = make_shared_addr(bank,
            shared_base_off + static_cast<Addr>(((block_row * 16 + r) * 32 + block_col * 16) * 4));
        Addr dst = make_local_addr(dst_tile, dst_local_off + static_cast<Addr>(r * 16 * 4));
        engine.dma_copy(src, dst, 16 * 4);
    }
}

// ─── Main schedule ───────────────────────────────────────────

void execute_dual_gemm(
    MemoryModel& mem,
    SyncManager& sync,
    FunctionalEngine& engine)
{
    constexpr int TILES_M = 2, TILES_N = 2, TILES_K = 2;

    // ════════════════════════════════════════════════════════════
    // GEMM1: C = A × B
    // ════════════════════════════════════════════════════════════

    // ── Phase 1: k=0 compute (tiles 0-3) ──
    for (int mn = 0; mn < TILES_M * TILES_N; mn++) {
        int m = mn / TILES_N;  // 0 or 1
        int n = mn % TILES_N;  // 0 or 1
        TileId tile = static_cast<TileId>(mn); // 0-3

        // DMA A[m,0] from DRAM to local
        dma_matrix_block_to_local(engine, mem,
            DramLayout::A_BASE, m, 0, tile, LocalLayout::A_OFF, 2);

        // DMA B[0,n] from DRAM to local
        dma_matrix_block_to_local(engine, mem,
            DramLayout::B_BASE, 0, n, tile, LocalLayout::B_OFF, 2);

        // MXU: psum_k0 = A × B (f16×f16 → f32)
        engine.mxu_f16xf16_to_f32(
            make_local_addr(tile, LocalLayout::A_OFF),
            make_local_addr(tile, LocalLayout::B_OFF),
            make_local_addr(tile, LocalLayout::P_OFF));

        // Local release (reducer is self for k=0)
        SyncTag psum_tag{tile, 0}; // tag index 0 for GEMM1 psum
        sync.release(psum_tag);    // cnt = 1
    }

    // ── Phase 2: k=1 compute (tiles 4-7) ──
    for (int mn = 0; mn < TILES_M * TILES_N; mn++) {
        int m = mn / TILES_N;
        int n = mn % TILES_N;
        TileId tile = static_cast<TileId>(4 + mn); // 4-7
        TileId reducer = static_cast<TileId>(mn);  // 0-3

        // DMA A[m,1] from DRAM to local
        dma_matrix_block_to_local(engine, mem,
            DramLayout::A_BASE, m, 1, tile, LocalLayout::A_OFF, 2);

        // DMA B[1,n] from DRAM to local
        dma_matrix_block_to_local(engine, mem,
            DramLayout::B_BASE, 1, n, tile, LocalLayout::B_OFF, 2);

        // MXU: psum_k1 = A × B (f16×f16 → f32)
        engine.mxu_f16xf16_to_f32(
            make_local_addr(tile, LocalLayout::A_OFF),
            make_local_addr(tile, LocalLayout::B_OFF),
            make_local_addr(tile, LocalLayout::P_OFF));

        // DMA psum_k1 (f32, 1024 bytes) to reducer's LOCAL[RP_OFF]
        engine.dma_copy(
            make_local_addr(tile, LocalLayout::P_OFF),
            make_local_addr(reducer, LocalLayout::RP_OFF),
            16 * 16 * sizeof(float));

        // Remote release → increments reducer's join counter
        SyncTag psum_tag{reducer, 0};
        sync.release(psum_tag); // cnt = 2
    }

    // ── Phase 3: Reduce & write C (tiles 0-3) ──
    for (int mn = 0; mn < TILES_M * TILES_N; mn++) {
        int m = mn / TILES_N;
        int n = mn % TILES_N;
        TileId tile = static_cast<TileId>(mn); // 0-3

        SyncTag psum_tag{tile, 0};

        // Acquire: wait for both k-slices
        bool ok = sync.acquire(psum_tag, TILES_K);
        assert(ok && "GEMM1 psum acquire failed — scheduling error");
        (void)ok;

        // Reduce: psum_k0 (at P_OFF = 0x0400) + psum_k1 (at RP_OFF = 0x0000)
        engine.reduce_f32(
            make_local_addr(tile, LocalLayout::RK0_OFF),  // psum k=0 (own)
            make_local_addr(tile, LocalLayout::RP_OFF),   // psum k=1 (received)
            make_local_addr(tile, LocalLayout::P_OFF));   // result overwrites k=0

        // Write C block (f32) to SHARED bank 0
        dma_local_to_shared_block(engine,
            tile, LocalLayout::P_OFF,
            SharedLayout::C_BANK, SharedLayout::C_BASE_OFF, m, n);

        // Release C block for GEMM2
        SyncTag c_tag{tile, 1}; // tag index 1 for C block
        sync.release(c_tag);

        // Reset psum counter
        sync.reset_counter(psum_tag);
    }

    // ════════════════════════════════════════════════════════════
    // GEMM2: E = C × D
    // ════════════════════════════════════════════════════════════

    // ── Phase 4: k=0 compute (tiles 0-3) ──
    for (int mn = 0; mn < TILES_M * TILES_N; mn++) {
        int m = mn / TILES_N;
        int n = mn % TILES_N;
        TileId tile = static_cast<TileId>(mn); // 0-3

        SyncTag c_tag{tile, 1};
        bool ok = sync.acquire(c_tag, 1);
        assert(ok && "GEMM2 C acquire failed");
        (void)ok;

        // DMA C[m,0] from SHARED to local (f32, 1024 bytes per row-strip)
        dma_shared_block_to_local(engine,
            SharedLayout::C_BANK, SharedLayout::C_BASE_OFF, m, 0,
            tile, LocalLayout::C_OFF);

        // DMA D[0,n] from DRAM to local (f16)
        dma_matrix_block_to_local(engine, mem,
            DramLayout::D_BASE, 0, n, tile, LocalLayout::D_OFF, 2);

        // MXU: psum_k0 = C × D (f32×f16 → f32)
        engine.mxu_f32xf16_to_f32(
            make_local_addr(tile, LocalLayout::C_OFF),
            make_local_addr(tile, LocalLayout::D_OFF),
            make_local_addr(tile, LocalLayout::P2_OFF));

        // Local release for GEMM2 psum
        SyncTag psum2_tag{tile, 2}; // tag index 2 for GEMM2 psum
        sync.release(psum2_tag);    // cnt = 1
    }

    // ── Phase 5: k=1 compute (tiles 4-7) ──
    for (int mn = 0; mn < TILES_M * TILES_N; mn++) {
        int m = mn / TILES_N;
        int n = mn % TILES_N;
        TileId tile = static_cast<TileId>(4 + mn); // 4-7
        TileId reducer = static_cast<TileId>(mn);  // 0-3

        // DMA C[m,1] from SHARED to local
        dma_shared_block_to_local(engine,
            SharedLayout::C_BANK, SharedLayout::C_BASE_OFF, m, 1,
            tile, LocalLayout::C_OFF);

        // DMA D[1,n] from DRAM to local
        dma_matrix_block_to_local(engine, mem,
            DramLayout::D_BASE, 1, n, tile, LocalLayout::D_OFF, 2);

        // MXU: psum_k1 = C × D (f32×f16 → f32)
        engine.mxu_f32xf16_to_f32(
            make_local_addr(tile, LocalLayout::C_OFF),
            make_local_addr(tile, LocalLayout::D_OFF),
            make_local_addr(tile, LocalLayout::P2_OFF));

        // DMA psum to reducer's LOCAL[RP2_OFF]
        engine.dma_copy(
            make_local_addr(tile, LocalLayout::P2_OFF),
            make_local_addr(reducer, LocalLayout::RP2_OFF),
            16 * 16 * sizeof(float));

        // Remote release
        SyncTag psum2_tag{reducer, 2};
        sync.release(psum2_tag); // cnt = 2
    }

    // ── Phase 6: Reduce & write E (tiles 0-3) ──
    for (int mn = 0; mn < TILES_M * TILES_N; mn++) {
        int m = mn / TILES_N;
        int n = mn % TILES_N;
        TileId tile = static_cast<TileId>(mn); // 0-3

        SyncTag psum2_tag{tile, 2};
        bool ok = sync.acquire(psum2_tag, TILES_K);
        assert(ok && "GEMM2 psum acquire failed");
        (void)ok;

        // Reduce: psum_k0 (at P2_OFF = 0x0600) + psum_k1 (at RP2_OFF = 0x0000)
        engine.reduce_f32(
            make_local_addr(tile, LocalLayout::P2_OFF),   // psum k=0 (own)
            make_local_addr(tile, LocalLayout::RP2_OFF),  // psum k=1 (received)
            make_local_addr(tile, LocalLayout::E_F32_OFF)); // f32 result

        // Convert f32 → f16
        engine.convert_f32_to_f16(
            make_local_addr(tile, LocalLayout::E_F32_OFF),
            make_local_addr(tile, LocalLayout::E_F16_OFF));

        // DMA E block (f16) to DRAM
        dma_local_to_matrix_block(engine, mem,
            tile, LocalLayout::E_F16_OFF,
            DramLayout::E_BASE, m, n, 2);

        sync.reset_counter(psum2_tag);
    }
}

// ─── Data loading / reading ──────────────────────────────────

void load_inputs_to_dram(MemoryModel& mem, const f16* A, const f16* B, const f16* D) {
    mem.write(make_dram_addr(DramLayout::A_BASE), A, 32 * 32 * sizeof(f16));
    mem.write(make_dram_addr(DramLayout::B_BASE), B, 32 * 32 * sizeof(f16));
    mem.write(make_dram_addr(DramLayout::D_BASE), D, 32 * 32 * sizeof(f16));
}

void read_output_from_dram(MemoryModel& mem, f16* E) {
    mem.read(make_dram_addr(DramLayout::E_BASE), E, 32 * 32 * sizeof(f16));
}

void read_C_from_shared(MemoryModel& mem, float* C) {
    mem.read(make_shared_addr(SharedLayout::C_BANK, SharedLayout::C_BASE_OFF),
             C, 32 * 32 * sizeof(float));
}

} // namespace mobol

// ─── Timing Schedule Builder ─────────────────────────────────
// Mirrors execute_dual_gemm() exactly, producing ScheduleEntry objects.

#include "timing/timing_model.h"

namespace mobol {

static ScheduleEntry make_dma_dram_to_local(TileId tid, Addr dram_base,
    int brow, int bcol, Addr local_off, size_t esz, const std::string& note) {
    ScheduleEntry e;
    e.op = SchedOp::DMA_DRAM_TO_LOCAL;
    e.tile_id = tid;
    e.is_strided = true;
    e.base_addr = dram_base;
    e.block_row = brow;
    e.block_col = bcol;
    e.matrix_dim = 32;
    e.elem_size = esz;
    e.size_bytes = 16 * 16 * esz;
    e.src_addr = make_dram_addr(dram_base);
    e.dst_addr = make_local_addr(tid, local_off);
    e.note = note;
    return e;
}

static ScheduleEntry make_dma_shared_to_local(TileId tid,
    BankId bank, Addr base_off, int brow, int bcol, Addr local_off, const std::string& note) {
    ScheduleEntry e;
    e.op = SchedOp::DMA_SHARED_TO_LOCAL;
    e.tile_id = tid;
    e.is_strided = true;
    e.src_addr = make_shared_addr(bank, base_off);
    e.dst_addr = make_local_addr(tid, local_off);
    e.size_bytes = 16 * 16 * 4; // f32
    e.note = note;
    return e;
}

static ScheduleEntry make_dma_local_to_shared(TileId tid,
    Addr local_off, BankId bank, Addr base_off, int brow, int bcol, const std::string& note) {
    ScheduleEntry e;
    e.op = SchedOp::DMA_LOCAL_TO_SHARED;
    e.tile_id = tid;
    e.is_strided = true;
    e.src_addr = make_local_addr(tid, local_off);
    e.dst_addr = make_shared_addr(bank, base_off);
    e.size_bytes = 16 * 16 * 4;
    e.note = note;
    return e;
}

static ScheduleEntry make_dma_local_to_local(TileId src_tid, Addr src_off,
    TileId dst_tid, Addr dst_off, size_t bytes, const std::string& note) {
    ScheduleEntry e;
    e.op = SchedOp::DMA_LOCAL_TO_LOCAL;
    e.tile_id = src_tid;
    e.src_addr = make_local_addr(src_tid, src_off);
    e.dst_addr = make_local_addr(dst_tid, dst_off);
    e.size_bytes = bytes;
    e.note = note;
    return e;
}

static ScheduleEntry make_dma_local_to_dram(TileId tid, Addr local_off,
    Addr dram_base, int brow, int bcol, size_t esz, const std::string& note) {
    ScheduleEntry e;
    e.op = SchedOp::DMA_LOCAL_TO_DRAM;
    e.tile_id = tid;
    e.is_strided = true;
    e.src_addr = make_local_addr(tid, local_off);
    e.dst_addr = make_dram_addr(dram_base);
    e.elem_size = esz;
    e.size_bytes = 16 * 16 * esz;
    e.note = note;
    return e;
}

static ScheduleEntry make_mxu(TileId tid, SchedOp mxu_op, const std::string& note) {
    ScheduleEntry e;
    e.op = mxu_op;
    e.tile_id = tid;
    e.note = note;
    return e;
}

static ScheduleEntry make_sync(TileId tid, SchedOp sync_op, uint32_t tag, uint32_t arity,
    const std::string& note) {
    ScheduleEntry e;
    e.op = sync_op;
    e.tile_id = tid;
    e.tag_index = tag;
    e.arity = arity;
    e.note = note;
    return e;
}

void build_timing_schedule(std::vector<ScheduleEntry>& schedule) {
    // Per-tile schedules
    std::map<TileId, std::vector<ScheduleEntry>> per_tile;

    constexpr int TM = 2, TN = 2;

    // ═══ GEMM1 Phase 1: k=0 compute (tiles 0-3) ═══
    for (int mn = 0; mn < TM * TN; mn++) {
        int m = mn / TN, n = mn % TN;
        TileId t = static_cast<TileId>(mn);
        auto& s = per_tile[t];

        s.push_back(make_dma_dram_to_local(t, DramLayout::A_BASE, m, 0, LocalLayout::A_OFF, 2, "A[m,0]"));
        s.push_back(make_dma_dram_to_local(t, DramLayout::B_BASE, 0, n, LocalLayout::B_OFF, 2, "B[0,n]"));
        s.push_back(make_mxu(t, SchedOp::MXU_F16xF16, "psum_k0"));
        s.push_back(make_sync(t, SchedOp::RELEASE_SYNC, 0, 0, "psum_gemm1"));
    }

    // ═══ GEMM1 Phase 2: k=1 compute (tiles 4-7) ═══
    for (int mn = 0; mn < TM * TN; mn++) {
        int m = mn / TN, n = mn % TN;
        TileId t = static_cast<TileId>(4 + mn);
        TileId reducer = static_cast<TileId>(mn);
        auto& s = per_tile[t];

        s.push_back(make_dma_dram_to_local(t, DramLayout::A_BASE, m, 1, LocalLayout::A_OFF, 2, "A[m,1]"));
        s.push_back(make_dma_dram_to_local(t, DramLayout::B_BASE, 1, n, LocalLayout::B_OFF, 2, "B[1,n]"));
        s.push_back(make_mxu(t, SchedOp::MXU_F16xF16, "psum_k1"));
        s.push_back(make_dma_local_to_local(t, LocalLayout::P_OFF, reducer, LocalLayout::RP_OFF,
            16*16*sizeof(float), "psum→reducer"));
        // Release targets the REDUCER tile's counter
        auto rel = make_sync(reducer, SchedOp::RELEASE_SYNC, 0, 0, "remote_release_gemm1");
        rel.tile_id = t; // Operation runs on tile t but targets reducer's counter
        per_tile[t].push_back(rel);
    }

    // ═══ GEMM1 Phase 3: Reduce & write C (tiles 0-3) ═══
    for (int mn = 0; mn < TM * TN; mn++) {
        int m = mn / TN, n = mn % TN;
        TileId t = static_cast<TileId>(mn);
        auto& s = per_tile[t];

        s.push_back(make_sync(t, SchedOp::ACQUIRE_SYNC, 0, 2, "acquire_psum_gemm1"));

        ScheduleEntry reduce;
        reduce.op = SchedOp::REDUCE_F32;
        reduce.tile_id = t;
        reduce.note = "reduce k0+k1→C";
        s.push_back(reduce);

        s.push_back(make_dma_local_to_shared(t, LocalLayout::P_OFF,
            SharedLayout::C_BANK, SharedLayout::C_BASE_OFF, m, n, "C→SHARED"));

        s.push_back(make_sync(t, SchedOp::RELEASE_SYNC, 1, 0, "release_C"));
    }

    // ═══ GEMM2 Phase 4: k=0 compute (tiles 0-3) ═══
    for (int mn = 0; mn < TM * TN; mn++) {
        int m = mn / TN, n = mn % TN;
        TileId t = static_cast<TileId>(mn);
        auto& s = per_tile[t];

        s.push_back(make_sync(t, SchedOp::ACQUIRE_SYNC, 1, 1, "acquire_C"));
        s.push_back(make_dma_shared_to_local(t, SharedLayout::C_BANK, SharedLayout::C_BASE_OFF,
            m, 0, LocalLayout::C_OFF, "C[m,0]"));
        s.push_back(make_dma_dram_to_local(t, DramLayout::D_BASE, 0, n, LocalLayout::D_OFF, 2, "D[0,n]"));
        s.push_back(make_mxu(t, SchedOp::MXU_F32xF16, "psum2_k0"));
        s.push_back(make_sync(t, SchedOp::RELEASE_SYNC, 2, 0, "psum2_gemm2"));
    }

    // ═══ GEMM2 Phase 5: k=1 compute (tiles 4-7) ═══
    for (int mn = 0; mn < TM * TN; mn++) {
        int m = mn / TN, n = mn % TN;
        TileId t = static_cast<TileId>(4 + mn);
        TileId reducer = static_cast<TileId>(mn);
        auto& s = per_tile[t];

        s.push_back(make_dma_shared_to_local(t, SharedLayout::C_BANK, SharedLayout::C_BASE_OFF,
            m, 1, LocalLayout::C_OFF, "C[m,1]"));
        s.push_back(make_dma_dram_to_local(t, DramLayout::D_BASE, 1, n, LocalLayout::D_OFF, 2, "D[1,n]"));
        s.push_back(make_mxu(t, SchedOp::MXU_F32xF16, "psum2_k1"));
        s.push_back(make_dma_local_to_local(t, LocalLayout::P2_OFF, reducer, LocalLayout::RP2_OFF,
            16*16*sizeof(float), "psum2→reducer"));
        auto rel = make_sync(reducer, SchedOp::RELEASE_SYNC, 2, 0, "remote_release_gemm2");
        rel.tile_id = t;
        per_tile[t].push_back(rel);
    }

    // ═══ GEMM2 Phase 6: Reduce & write E (tiles 0-3) ═══
    for (int mn = 0; mn < TM * TN; mn++) {
        int m = mn / TN, n = mn % TN;
        TileId t = static_cast<TileId>(mn);
        auto& s = per_tile[t];

        s.push_back(make_sync(t, SchedOp::ACQUIRE_SYNC, 2, 2, "acquire_psum2_gemm2"));

        ScheduleEntry reduce;
        reduce.op = SchedOp::REDUCE_F32;
        reduce.tile_id = t;
        reduce.note = "reduce2 k0+k1→E";
        s.push_back(reduce);

        ScheduleEntry conv;
        conv.op = SchedOp::CONVERT_F32_F16;
        conv.tile_id = t;
        conv.note = "E f32→f16";
        s.push_back(conv);

        s.push_back(make_dma_local_to_dram(t, LocalLayout::E_F16_OFF,
            DramLayout::E_BASE, m, n, 2, "E→DRAM"));
    }

    // Flatten per-tile schedules into the output
    // Interleave: process tiles in order 0,1,2,3,4,5,6,7
    for (int t = 0; t < 8; t++) {
        for (auto& entry : per_tile[static_cast<TileId>(t)]) {
            schedule.push_back(entry);
        }
    }
}

} // namespace mobol
