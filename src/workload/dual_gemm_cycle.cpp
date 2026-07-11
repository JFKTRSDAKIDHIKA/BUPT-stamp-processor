/// @file dual_gemm_cycle.cpp
/// @brief Per-tile instruction streams for the dual-GEMM workload.

#include "workload/dual_gemm_cycle.h"
#include "workload/dual_gemm.h"
#include "common/address.h"

namespace mobol::cycle {

namespace {

constexpr int DIM = 32;       // matrix dimension
constexpr int BN = 16;        // block dimension

/// DRAM address of block (r, c) of a row-major DIM x DIM matrix.
Addr dram_block(Addr base, int r, int c, size_t esz) {
    return make_dram_addr(base + static_cast<Addr>((r * BN * DIM + c * BN) * esz));
}

/// Address of block (r, c) of the row-major f32 C matrix: SHARED bank 0
/// (3-die) or a DRAM scratch region (2-die, no buffer die).
Addr c_block(bool use_shared, int r, int c) {
    Addr off = static_cast<Addr>((r * BN * DIM + c * BN) * 4);
    return use_shared
        ? make_shared_addr(SharedLayout::C_BANK, SharedLayout::C_BASE_OFF + off)
        : make_dram_addr(DG_C_DRAM_OFF + off);
}

/// 16-row strided DMA: DRAM matrix block -> contiguous LOCAL block.
Instr load_block(TileId t, Addr base, int r, int c, uint32_t local_off,
                 size_t esz, std::string note) {
    return mk_dma(dram_block(base, r, c, esz), make_local_addr(t, local_off),
                  BN, static_cast<uint32_t>(BN * esz),
                  static_cast<int64_t>(DIM * esz), static_cast<int64_t>(BN * esz),
                  std::move(note));
}

/// Contiguous LOCAL block -> DRAM matrix block.
Instr store_block(TileId t, uint32_t local_off, Addr base, int r, int c,
                  size_t esz, std::string note) {
    return mk_dma(make_local_addr(t, local_off), dram_block(base, r, c, esz),
                  BN, static_cast<uint32_t>(BN * esz),
                  static_cast<int64_t>(BN * esz), static_cast<int64_t>(DIM * esz),
                  std::move(note));
}

} // namespace

Program build_dual_gemm_program(bool use_shared) {
    using L = DGLayout;
    if (NUM_TILES < 8)
        throw std::runtime_error(
            "dual_gemm workload encodes a fixed 8-tile dataflow");
    Program prog;

    // ═══ Tiles 0-3: k=0 compute + reduce (both GEMMs) ═══
    for (int mn = 0; mn < 4; mn++) {
        int m = mn / 2, n = mn % 2;
        TileId t = static_cast<TileId>(mn);
        auto add = [&](Instr i) { prog.add(t, std::move(i)); };

        // ── GEMM1, k = 0 ──
        add(load_block(t, DramLayout::A_BASE, m, 0, L::A_OFF, 2, "A[m,0]"));
        add(load_block(t, DramLayout::B_BASE, 0, n, L::B_OFF, 2, "B[0,n]"));
        add(mk_dma_fence());
        add(mk_mxu(Op::MXU_F16F16, L::A_OFF, L::B_OFF, L::P_OFF, false, "psum1_k0"));
        add(mk_wait_mxu());
        add(mk_release(t, DGTags::PSUM1, "psum1 self"));

        // ── Reduce k0 + k1 (k1 psum arrives at RP_OFF from tile 4+mn) ──
        add(mk_acquire(DGTags::PSUM1, 2, "join psum1"));
        add(mk_vpu(Op::VPU_ADD_F32, L::P_OFF, L::RP_OFF, L::P_OFF));

        // ── C block (f32) -> SHARED bank 0 (near for group 0) ──
        add(mk_dma(make_local_addr(t, L::P_OFF), c_block(use_shared, m, n),
                   BN, BN * 4, BN * 4, DIM * 4, "C->SHARED"));
        add(mk_dma_fence());

        // C consumers: k=0 readers need C[m,0] (written by tile 2m);
        // k=1 readers need C[m,1] (written by tile 2m+1).
        if (n == 0) {
            add(mk_release(static_cast<TileId>(2 * m),     DGTags::C_RDY, "C[m,0] rdy"));
            add(mk_release(static_cast<TileId>(2 * m + 1), DGTags::C_RDY, "C[m,0] rdy"));
        } else {
            add(mk_release(static_cast<TileId>(4 + 2 * m),     DGTags::C_RDY, "C[m,1] rdy"));
            add(mk_release(static_cast<TileId>(4 + 2 * m + 1), DGTags::C_RDY, "C[m,1] rdy"));
        }

        // ── GEMM2, k = 0 ──
        add(mk_acquire(DGTags::C_RDY, 1, "wait C[m,0]"));
        add(mk_dma(c_block(use_shared, m, 0), make_local_addr(t, L::C_OFF),
                   BN, BN * 4, DIM * 4, BN * 4, "C[m,0]<-SHARED"));
        add(load_block(t, DramLayout::D_BASE, 0, n, L::D_OFF, 2, "D[0,n]"));
        add(mk_dma_fence());
        add(mk_mxu(Op::MXU_F32F16, L::C_OFF, L::D_OFF, L::P2_OFF, false, "psum2_k0"));
        add(mk_wait_mxu());
        add(mk_release(t, DGTags::PSUM2, "psum2 self"));

        // ── Reduce, convert, write E back to DRAM ──
        add(mk_acquire(DGTags::PSUM2, 2, "join psum2"));
        add(mk_vpu(Op::VPU_ADD_F32, L::P2_OFF, L::RP2_OFF, L::E32_OFF));
        add(mk_vpu(Op::VPU_CVT_F32_F16, L::E32_OFF, 0, L::E16_OFF));
        add(store_block(t, L::E16_OFF, DramLayout::E_BASE, m, n, 2, "E->DRAM"));
        add(mk_dma_fence());
        add(mk_halt());
    }

    // ═══ Tiles 4-7: k=1 compute, psum push to reducers ═══
    for (int mn = 0; mn < 4; mn++) {
        int m = mn / 2, n = mn % 2;
        TileId t = static_cast<TileId>(4 + mn);
        TileId red = static_cast<TileId>(mn);
        auto add = [&](Instr i) { prog.add(t, std::move(i)); };

        // ── GEMM1, k = 1 ──
        add(load_block(t, DramLayout::A_BASE, m, 1, L::A_OFF, 2, "A[m,1]"));
        add(load_block(t, DramLayout::B_BASE, 1, n, L::B_OFF, 2, "B[1,n]"));
        add(mk_dma_fence());
        add(mk_mxu(Op::MXU_F16F16, L::A_OFF, L::B_OFF, L::P_OFF, false, "psum1_k1"));
        add(mk_wait_mxu());
        add(mk_dma_linear(make_local_addr(t, L::P_OFF), make_local_addr(red, L::RP_OFF),
                          1024, "psum1 push"));
        add(mk_dma_fence());  // release only after the data is committed
        add(mk_release(red, DGTags::PSUM1, "psum1 remote"));

        // ── GEMM2, k = 1: needs C[m,1] (released by tile 2m+1) ──
        add(mk_acquire(DGTags::C_RDY, 1, "wait C[m,1]"));
        add(mk_dma(c_block(use_shared, m, 1), make_local_addr(t, L::C_OFF),
                   BN, BN * 4, DIM * 4, BN * 4, "C[m,1]<-SHARED (far)"));
        add(load_block(t, DramLayout::D_BASE, 1, n, L::D_OFF, 2, "D[1,n]"));
        add(mk_dma_fence());
        add(mk_mxu(Op::MXU_F32F16, L::C_OFF, L::D_OFF, L::P2_OFF, false, "psum2_k1"));
        add(mk_wait_mxu());
        add(mk_dma_linear(make_local_addr(t, L::P2_OFF), make_local_addr(red, L::RP2_OFF),
                          1024, "psum2 push"));
        add(mk_dma_fence());
        add(mk_release(red, DGTags::PSUM2, "psum2 remote"));
        add(mk_halt());
    }

    // ═══ Tiles 8-15: unused in this workload ═══
    for (int t = 8; t < NUM_TILES; t++)
        prog.add(static_cast<TileId>(t), mk_halt());

    return prog;
}

} // namespace mobol::cycle
