/// @file llm_layer.cpp
/// @brief Transformer layer: program builder + CPU references.

#include "workload/llm_layer.h"
#include "cycle/blockops.h"
#include "cycle/flit.h"
#include "common/address.h"

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>

namespace mobol::cycle {

namespace {

constexpr int BN = 16;

// ─── Local scratchpad map (per tile, disjoint) ───────────────
struct L {
    // Phase A: projections (tiles 0-11)
    static constexpr uint32_t XB     = 0x0000; // 4 x 512 B f16 X blocks
    static constexpr uint32_t WB     = 0x0800; // 4 x 512 B f16 weight blocks
    static constexpr uint32_t PROJ32 = 0x1000; // 1 KB f32
    static constexpr uint32_t PROJ16 = 0x1400; // 512 B f16
    static constexpr uint32_t KT16   = 0x1600; // 512 B f16 (K^T on K tiles)
    // Head tiles 0-3 receive:
    static constexpr uint32_t KT_IN  = 0x1800; // 512 B
    static constexpr uint32_t V_IN   = 0x1A00; // 512 B
    // Phase B: attention (tiles 0-3)
    static constexpr uint32_t S32    = 0x1C00; // 1 KB
    static constexpr uint32_t P32    = 0x2000; // 1 KB
    static constexpr uint32_t P16    = 0x2400; // 512 B
    static constexpr uint32_t O32    = 0x2600; // 1 KB
    static constexpr uint32_t O16    = 0x2A00; // 512 B
    static constexpr uint32_t O_ALL  = 0x2C00; // 4 x 512 B (h-indexed)
    static constexpr uint32_t WO_B   = 0x3400; // 4 x 512 B
    static constexpr uint32_t ATTN32 = 0x3C00; // 1 KB
    static constexpr uint32_t XN16   = 0x4000; // 512 B
    static constexpr uint32_t X32    = 0x4200; // 1 KB
    static constexpr uint32_t R32    = 0x4600; // 1 KB — lives until FFN2!
    // LayerNorm (tile 0 only)
    static constexpr uint32_t LN_G   = 0x5000; // 16x64 f32 row-major, 4 KB
    static constexpr uint32_t LN_O32 = 0x6000; // 4 KB
    static constexpr uint32_t LN_T32 = 0x7000; // 1 KB
    static constexpr uint32_t LN_T16 = 0x7400; // 512 B
    static constexpr uint32_t LN16   = 0x7600; // 16x64 f16 row-major, 2 KB
    // Phase C: FFN1 (all tiles)
    static constexpr uint32_t LNB    = 0x8000; // 4 x 512 B
    static constexpr uint32_t W1B    = 0x8800; // 4 x 512 B
    static constexpr uint32_t H32    = 0x9000; // 1 KB
    static constexpr uint32_t H16    = 0x9400; // 512 B
    static constexpr uint32_t W2B    = 0x9600; // 4 x 512 B
    static constexpr uint32_t PART32 = 0xA000; // 4 x 1 KB (n-indexed)
    // Phase D: reduction (tiles 0-3)
    static constexpr uint32_t SLOTS  = 0xC000; // 16 x 1 KB (j-indexed)
    static constexpr uint32_t ACC32  = 0x10000;
    static constexpr uint32_t RES32  = 0x10400;
    static constexpr uint32_t OUT16  = 0x10800;
};

// Sync tags (indices scoped per consumer tile; generation counters
// self-reset on acquire, so reuse across layers is safe).
struct T {
    static constexpr uint32_t KV    = 0; ///< heads: K^T+V arrived (arity 2)
    static constexpr uint32_t O_RDY = 1; ///< proj rows: all O_h arrived (arity 4)
    static constexpr uint32_t LN_G  = 2; ///< tile0: R blocks gathered (arity 4)
    static constexpr uint32_t LN_P  = 3; ///< LN published to SHARED (arity 1)
    static constexpr uint32_t PSUM  = 4; ///< reducers: FFN partials (arity 16)
    static constexpr uint32_t X_RDY = 5; ///< next-layer input in DRAM (arity 4)
    static constexpr uint32_t NMC_D = 6; ///< NMC command completed (arity 1)
    static constexpr uint32_t WPREF_D = 7; ///< leader: prefetch cmds done
    static constexpr uint32_t WPREF = 8;   ///< consumers: layer weights in bank
};

constexpr Addr LN_SH_OFF = 0x0; ///< LN activation offset inside each bank

// SHARED regions for the NMC paradigm. The FFN reduction for output
// block n lives entirely in bank n (slots, engine, result) so the four
// bank engines reduce in parallel — one engine per bank matches the
// dataflow instead of serializing on bank 0. LN stays on bank 0 (its
// input R is gathered once). Per-layer slot regions avoid re-zeroing;
// generation safety comes from the X_RDY layer barrier.
struct SH {
    static constexpr Addr LN_G  = 0x10000;                  ///< 16x64 f32 gather
    static Addr slots(int l) { return 0x20000 + l * 0x8000; } ///< 16 KB in bank n
    static Addr accr(int l) { return 0x100000 + l * 0x400; }  ///< 1 KB in bank n
};

/// NO_BUFFER paradigm: LN activations round-trip DRAM instead of SRAM.
Addr lnp_dram(int layer) { return 0x080000 + layer * 0x1000; }

// Weight-streaming regions per bank, double-buffered by layer parity.
// Each bank holds exactly the weight slices its group's tiles consume
// (no replication — total DRAM traffic equals the baseline's):
//   bank0: Wq | bank1: Wk | bank2: Wv   (whole 64x64, ld 64)
//   bank0 additionally: Wo
//   every bank: W1[:, 64g:64g+64] and W2[64g:64g+64, :]  (64x64, ld 64)
struct WS {
    static Addr base(int l)  { return 0x200000 + (l & 1) * 0x20000; }
    static Addr wrole(int l) { return base(l) + 0x0000; }
    static Addr wo(int l)    { return base(l) + 0x2000; }
    static Addr w1(int l)    { return base(l) + 0x4000; }
    static Addr w2(int l)    { return base(l) + 0x6000; }
};

/// Block (r, c) pull from a bank-resident 64x64 f16 weight region.
Instr sh_blk(TileId t, BankId b, Addr region, int r, int c, uint32_t off,
             std::string note) {
    return mk_dma(make_shared_addr(b, region + (r * BN * 64 + c * BN) * 2),
                  make_local_addr(t, off),
                  BN, BN * 2, 64 * 2, BN * 2, std::move(note));
}

Addr act_addr(int layer) { return LlmDram::ACT_BASE + layer * LlmDram::ACT_STRIDE; }
Addr wbase(int layer) { return LlmDram::L_BASE + layer * LlmDram::L_STRIDE; }

/// DRAM block (r,c) of a row-major matrix with leading dimension ld (f16).
Addr dblk(Addr base, int ld, int r, int c) {
    return make_dram_addr(base + static_cast<Addr>((r * BN * ld + c * BN) * 2));
}

/// Load a f16 block: DRAM (ld columns) -> contiguous LOCAL.
Instr ld_blk(TileId t, Addr base, int ld, int r, int c, uint32_t off, std::string note) {
    return mk_dma(dblk(base, ld, r, c), make_local_addr(t, off),
                  BN, BN * 2, ld * 2, BN * 2, std::move(note));
}

/// Store a f16 block: contiguous LOCAL -> DRAM (ld columns).
Instr st_blk(TileId t, uint32_t off, Addr base, int ld, int r, int c, std::string note) {
    return mk_dma(make_local_addr(t, off), dblk(base, ld, r, c),
                  BN, BN * 2, BN * 2, ld * 2, std::move(note));
}

} // namespace

// ═══ Model generation / DRAM loading ═════════════════════════

LlmData llm_generate(const LlmParams& p) {
    std::mt19937 rng(p.seed);
    auto gen = [&](size_t n, float scale) {
        std::uniform_real_distribution<float> dist(-scale, scale);
        std::vector<f16> v(n);
        for (auto& x : v) x = f16(dist(rng));
        return v;
    };
    LlmData d;
    d.X = gen(LlmParams::SEQ * LlmParams::DM, 0.5f);
    d.w.resize(p.layers);
    for (auto& w : d.w) {
        w.Wq = gen(LlmParams::DM * LlmParams::DM, 0.125f);
        w.Wk = gen(LlmParams::DM * LlmParams::DM, 0.125f);
        w.Wv = gen(LlmParams::DM * LlmParams::DM, 0.125f);
        w.Wo = gen(LlmParams::DM * LlmParams::DM, 0.125f);
        w.W1 = gen(LlmParams::DM * LlmParams::FF, 0.125f);
        w.W2 = gen(LlmParams::FF * LlmParams::DM, 0.0625f);
    }
    return d;
}

void llm_load_dram(MemoryModel& mem, const LlmParams& p, const LlmData& d) {
    mem.write(make_dram_addr(act_addr(0)), d.X.data(), d.X.size() * 2);
    for (int l = 0; l < p.layers; l++) {
        const auto& w = d.w[l];
        Addr b = wbase(l);
        mem.write(make_dram_addr(b + LlmDram::WQ), w.Wq.data(), w.Wq.size() * 2);
        mem.write(make_dram_addr(b + LlmDram::WK), w.Wk.data(), w.Wk.size() * 2);
        mem.write(make_dram_addr(b + LlmDram::WV), w.Wv.data(), w.Wv.size() * 2);
        mem.write(make_dram_addr(b + LlmDram::WO), w.Wo.data(), w.Wo.size() * 2);
        mem.write(make_dram_addr(b + LlmDram::W1), w.W1.data(), w.W1.size() * 2);
        mem.write(make_dram_addr(b + LlmDram::W2), w.W2.data(), w.W2.size() * 2);
    }
}

// ═══ Program builder ═════════════════════════════════════════

Program build_llm_program(const LlmParams& p) {
    constexpr int DM = LlmParams::DM, FF = LlmParams::FF;
    constexpr int KB = DM / BN;         // 4 K-blocks over d_model
    constexpr int NH = LlmParams::HEADS;
    constexpr int FJ = FF / BN;         // 16 FFN blocks

    if (p.stream_weights && p.arch == LlmArch::NO_BUFFER)
        throw std::runtime_error("weight streaming requires the buffer die");

    Program prog;
    auto emit = [&](int t, Instr i) { prog.add(static_cast<TileId>(t), std::move(i)); };

    // Leader tile 4g issues the prefetch commands for bank g; command
    // counts per group (bank0 carries Wq+Wo, banks1/2 Wk/Wv, bank3 FFN only).
    static const int PF_CMDS[4] = {4, 3, 3, 2};
    auto emit_prefetch_batch = [&](int l, int g) {
        if (!p.stream_weights || l >= p.layers) return;
        int t = 4 * g;
        Addr wb = wbase(l);
        if (g < 3) {
            Addr role_src = wb + (g == 0 ? LlmDram::WQ : g == 1 ? LlmDram::WK
                                                                : LlmDram::WV);
            emit(t, mk_nmc_prefetch(make_dram_addr(role_src),
                    make_shared_addr(static_cast<BankId>(g), WS::wrole(l)),
                    64, 128, 128, 128, T::WPREF_D, "pf Wrole"));
        }
        if (g == 0)
            emit(t, mk_nmc_prefetch(make_dram_addr(wb + LlmDram::WO),
                    make_shared_addr(0, WS::wo(l)),
                    64, 128, 128, 128, T::WPREF_D, "pf Wo"));
        emit(t, mk_nmc_prefetch(make_dram_addr(wb + LlmDram::W1 + g * 64 * 2),
                make_shared_addr(static_cast<BankId>(g), WS::w1(l)),
                64, 128, FF * 2, 128, T::WPREF_D, "pf W1 sub"));
        emit(t, mk_nmc_prefetch(make_dram_addr(wb + LlmDram::W2 + g * 64 * DM * 2),
                make_shared_addr(static_cast<BankId>(g), WS::w2(l)),
                64, 128, 128, 128, T::WPREF_D, "pf W2 sub"));
    };

    // Layer-0 weights start streaming before any compute.
    if (p.stream_weights)
        for (int g = 0; g < NUM_BANKS; g++) emit_prefetch_batch(0, g);

    for (int l = 0; l < p.layers; l++) {
        // ── Weight-streaming leaders: collect this layer's completion
        //    tokens, publish WPREF to the group, then launch the NEXT
        //    layer's prefetch (overlapped with this layer's compute).
        //    The X_RDY gate keeps the parity-(l+1) double buffer safe:
        //    its previous readers (layer l-1) finished before X_RDY(l-1).
        if (p.stream_weights) {
            for (int g = 0; g < NUM_BANKS; g++) {
                int t = 4 * g;
                if (l > 0)
                    emit(t, mk_acquire(T::X_RDY, 4, "stream gate"));
                emit(t, mk_acquire(T::WPREF_D, PF_CMDS[g], "weights landed"));
                for (int c = 0; c < TILES_PER_GROUP; c++)
                    emit(t, mk_release(static_cast<TileId>(4 * g + c),
                                       T::WPREF, "w rdy"));
                emit_prefetch_batch(l + 1, g);
            }
        }
        Addr X = act_addr(l);
        Addr XN = act_addr(l + 1);
        Addr WQ = wbase(l) + LlmDram::WQ, WK = wbase(l) + LlmDram::WK;
        Addr WV = wbase(l) + LlmDram::WV, WO = wbase(l) + LlmDram::WO;
        Addr W1 = wbase(l) + LlmDram::W1, W2 = wbase(l) + LlmDram::W2;

        // ── Phase A: Q/K/V projections on tiles 0-11 ──
        for (int h = 0; h < NH; h++) {
            for (int role = 0; role < 3; role++) {   // 0=Q, 1=K, 2=V
                int t = role * NH + h;
                Addr W = role == 0 ? WQ : role == 1 ? WK : WV;

                if (l > 0)
                    emit(t, mk_acquire(T::X_RDY, 4, "layer input ready"));
                if (p.stream_weights)
                    emit(t, mk_acquire(T::WPREF, 1, "weights in bank"));
                for (int k = 0; k < KB; k++) {
                    emit(t, ld_blk(t, X, DM, 0, k, L::XB + k * 512, "X blk"));
                    if (p.stream_weights)
                        emit(t, sh_blk(t, static_cast<BankId>(role), WS::wrole(l),
                                       k, h, L::WB + k * 512, "W blk near"));
                    else
                        emit(t, ld_blk(t, W, DM, k, h, L::WB + k * 512, "W blk"));
                }
                emit(t, mk_dma_fence());
                for (int k = 0; k < KB; k++)
                    emit(t, mk_mxu(Op::MXU_F16F16, L::XB + k * 512, L::WB + k * 512,
                                   L::PROJ32, k > 0, "proj acc"));
                emit(t, mk_wait_mxu());
                emit(t, mk_vpu(Op::VPU_CVT_F32_F16, L::PROJ32, 0, L::PROJ16));

                if (role == 0) {
                    // Q stays on the head tile (t == h).
                } else if (role == 1) {
                    // K: transpose, push K^T to head tile h.
                    emit(t, mk_vpu(Op::VPU_TRANS_F16, L::PROJ16, 0, L::KT16));
                    emit(t, mk_dma_linear(make_local_addr(t, L::KT16),
                                          make_local_addr(h, L::KT_IN), 512, "K^T->head"));
                    emit(t, mk_dma_fence());
                    emit(t, mk_release(static_cast<TileId>(h), T::KV, "K^T rdy"));
                } else {
                    // V: push to head tile h.
                    emit(t, mk_dma_linear(make_local_addr(t, L::PROJ16),
                                          make_local_addr(h, L::V_IN), 512, "V->head"));
                    emit(t, mk_dma_fence());
                    emit(t, mk_release(static_cast<TileId>(h), T::KV, "V rdy"));
                }
            }
        }

        // ── Phase B: attention on head tiles 0-3 (Q is local PROJ16) ──
        for (int h = 0; h < NH; h++) {
            int t = h;
            emit(t, mk_acquire(T::KV, 2, "wait K^T,V"));
            emit(t, mk_mxu(Op::MXU_F16F16, L::PROJ16, L::KT_IN, L::S32, false, "S=QK^T"));
            emit(t, mk_wait_mxu());
            emit(t, mk_vpu(Op::VPU_SOFTMAX_F32, L::S32, 0, L::P32, LlmParams::ATT_SCALE));
            emit(t, mk_vpu(Op::VPU_CVT_F32_F16, L::P32, 0, L::P16));
            emit(t, mk_mxu(Op::MXU_F16F16, L::P16, L::V_IN, L::O32, false, "O=PV"));
            emit(t, mk_wait_mxu());
            emit(t, mk_vpu(Op::VPU_CVT_F32_F16, L::O32, 0, L::O16));
            // All-to-all: O_h to every projection-row tile (0-3).
            for (int n = 0; n < NH; n++) {
                emit(t, mk_dma_linear(make_local_addr(t, L::O16),
                                      make_local_addr(n, L::O_ALL + h * 512), 512, "O_h"));
            }
            emit(t, mk_dma_fence());
            for (int n = 0; n < NH; n++)
                emit(t, mk_release(static_cast<TileId>(n), T::O_RDY, "O_h rdy"));
        }

        // ── Output projection + residual on tiles 0-3 ──
        for (int n = 0; n < NH; n++) {
            int t = n;
            emit(t, mk_acquire(T::O_RDY, 4, "all O_h"));
            for (int h = 0; h < NH; h++) {
                if (p.stream_weights)
                    emit(t, sh_blk(t, 0, WS::wo(l), h, n, L::WO_B + h * 512,
                                   "Wo blk near"));
                else
                    emit(t, ld_blk(t, WO, DM, h, n, L::WO_B + h * 512, "Wo blk"));
            }
            emit(t, ld_blk(t, X, DM, 0, n, L::XN16, "X resid"));
            emit(t, mk_dma_fence());
            for (int h = 0; h < NH; h++)
                emit(t, mk_mxu(Op::MXU_F16F16, L::O_ALL + h * 512, L::WO_B + h * 512,
                               L::ATTN32, h > 0, "attn acc"));
            emit(t, mk_wait_mxu());
            emit(t, mk_vpu(Op::VPU_CVT_F16_F32, L::XN16, 0, L::X32));
            emit(t, mk_vpu(Op::VPU_ADD_F32, L::ATTN32, L::X32, L::R32));
            // Gather R into a row-major LN buffer (strided scatter):
            // on tile 0's LOCAL for base-die LN, or directly into bank 0
            // when the buffer-die NMC engine does the LN.
            Addr ln_dst = (p.arch == LlmArch::NMC)
                ? make_shared_addr(0, SH::LN_G + n * 64)
                : make_local_addr(0, L::LN_G + n * 64);
            emit(t, mk_dma(make_local_addr(t, L::R32), ln_dst,
                           BN, 64, 64, 256, "R->LN gather"));
            emit(t, mk_dma_fence());
            emit(t, mk_release(0, T::LN_G, "R gathered"));
        }

        // ── LayerNorm + publication (paradigm-dependent) ──
        {
            int t = 0;
            emit(t, mk_acquire(T::LN_G, 4, "R complete"));

            if (p.arch == LlmArch::NMC) {
                // Buffer-die NMC: fused LN+convert executes beside the
                // SRAM; tile 0 only replicates the 2 KB result to the
                // other banks. No gather into LOCAL, no repack dance,
                // tile 0's VPU stays free.
                emit(t, mk_nmc(static_cast<uint32_t>(NmcOp::LN_F16),
                               make_shared_addr(0, SH::LN_G),
                               make_shared_addr(0, LN_SH_OFF),
                               KB, T::NMC_D, LlmParams::LN_EPS, "NMC LN"));
                emit(t, mk_acquire(T::NMC_D, 1, "LN done"));
                emit(t, mk_dma(make_shared_addr(0, LN_SH_OFF),
                               make_local_addr(t, L::LN16),
                               1, LlmParams::SEQ * DM * 2, 0, 0, "LN16 near pull"));
                emit(t, mk_dma_fence());
                for (int b = 1; b < NUM_BANKS; b++)
                    emit(t, mk_dma_linear(make_local_addr(t, L::LN16),
                                          make_shared_addr(static_cast<BankId>(b), LN_SH_OFF),
                                          LlmParams::SEQ * DM * 2, "LN->bank"));
                emit(t, mk_dma_fence());
            } else {
                // Base-die LN on tile 0's VPU, then repack
                // row-major f32 -> per-block f16 -> row-major f16.
                emit(t, mk_vpu(Op::VPU_LAYERNORM_F32, L::LN_G, 0, L::LN_O32,
                               LlmParams::LN_EPS, KB));
                for (int k = 0; k < KB; k++) {
                    emit(t, mk_dma(make_local_addr(t, L::LN_O32 + k * 64),
                                   make_local_addr(t, L::LN_T32),
                                   BN, 64, 256, 64, "LN extract"));
                    emit(t, mk_dma_fence());
                    emit(t, mk_vpu(Op::VPU_CVT_F32_F16, L::LN_T32, 0, L::LN_T16));
                    emit(t, mk_dma(make_local_addr(t, L::LN_T16),
                                   make_local_addr(t, L::LN16 + k * 32),
                                   BN, 32, 32, 128, "LN insert"));
                    emit(t, mk_dma_fence());
                }
                if (p.arch == LlmArch::NO_BUFFER) {
                    // 2-die stack: no SRAM banks — publish once to DRAM.
                    emit(t, mk_dma_linear(make_local_addr(t, L::LN16),
                                          make_dram_addr(lnp_dram(l)),
                                          LlmParams::SEQ * DM * 2, "LN->DRAM"));
                } else {
                    // Replicate into every SHARED bank so each tile's FFN
                    // read is a near (own-bank) access.
                    for (int b = 0; b < NUM_BANKS; b++)
                        emit(t, mk_dma_linear(make_local_addr(t, L::LN16),
                                              make_shared_addr(static_cast<BankId>(b), LN_SH_OFF),
                                              LlmParams::SEQ * DM * 2, "LN->bank"));
                }
            }
            emit(t, mk_dma_fence());
            for (int j = 0; j < NUM_TILES; j++)
                emit(t, mk_release(static_cast<TileId>(j), T::LN_P, "LN published"));
        }

        // ── Phase C: FFN1 H = gelu(LN * W1) on all 16 tiles ──
        for (int j = 0; j < FJ; j++) {
            int t = j;
            emit(t, mk_acquire(T::LN_P, 1, "LN ready"));
            if (p.stream_weights && t >= 12)
                emit(t, mk_acquire(T::WPREF, 1, "weights in bank"));
            BankId nb = tile_group(static_cast<TileId>(t));  // near bank
            for (int k = 0; k < KB; k++) {
                // Pull LN block (0,k): from the near bank (row-major,
                // ld=DM), or from DRAM in the 2-die paradigm.
                Addr ln_src = (p.arch == LlmArch::NO_BUFFER)
                    ? make_dram_addr(lnp_dram(l) + k * BN * 2)
                    : make_shared_addr(nb, LN_SH_OFF + k * BN * 2);
                emit(t, mk_dma(ln_src, make_local_addr(t, L::LNB + k * 512),
                               BN, BN * 2, DM * 2, BN * 2, "LN blk"));
                if (p.stream_weights)
                    emit(t, sh_blk(t, nb, WS::w1(l), k, t % 4,
                                   L::W1B + k * 512, "W1 blk near"));
                else
                    emit(t, ld_blk(t, W1, FF, k, j, L::W1B + k * 512, "W1 blk"));
            }
            emit(t, mk_dma_fence());
            for (int k = 0; k < KB; k++)
                emit(t, mk_mxu(Op::MXU_F16F16, L::LNB + k * 512, L::W1B + k * 512,
                               L::H32, k > 0, "H acc"));
            emit(t, mk_wait_mxu());
            emit(t, mk_vpu(Op::VPU_GELU_F32, L::H32, 0, L::H32));
            emit(t, mk_vpu(Op::VPU_CVT_F32_F16, L::H32, 0, L::H16));

            // ── Phase D producer: partials for each output block n ──
            for (int n = 0; n < KB; n++) {
                if (p.stream_weights)
                    emit(t, sh_blk(t, nb, WS::w2(l), t % 4, n,
                                   L::W2B + n * 512, "W2 blk near"));
                else
                    emit(t, ld_blk(t, W2, DM, j, n, L::W2B + n * 512, "W2 blk"));
            }
            emit(t, mk_dma_fence());
            for (int n = 0; n < KB; n++)
                emit(t, mk_mxu(Op::MXU_F16F16, L::H16, L::W2B + n * 512,
                               L::PART32 + n * 1024, false, "FFN2 partial"));
            emit(t, mk_wait_mxu());
            for (int n = 0; n < KB; n++) {
                // Partial destinations: reducer-tile LOCAL slots (base-die
                // reduction) or bank-n slot regions (buffer-die NMC,
                // reduction distributed across the four bank engines).
                Addr slot = (p.arch == LlmArch::NMC)
                    ? make_shared_addr(static_cast<BankId>(n),
                                       SH::slots(l) + j * 1024)
                    : make_local_addr(n, L::SLOTS + j * 1024);
                emit(t, mk_dma_linear(make_local_addr(t, L::PART32 + n * 1024),
                                      slot, 1024, "psum->slot"));
            }
            emit(t, mk_dma_fence());
            for (int n = 0; n < KB; n++)
                emit(t, mk_release(static_cast<TileId>(n), T::PSUM, "psum rdy"));
        }

        // ── Phase D reducers: tiles 0-3, arity-16 join + residual ──
        for (int n = 0; n < KB; n++) {
            int t = n;
            emit(t, mk_acquire(T::PSUM, FJ, "16-way join"));
            if (p.arch == LlmArch::NMC) {
                // Fixed-order reduction beside bank n's SRAM (the four
                // bank engines run in parallel); the tile then fetches
                // only the 1 KB result.
                BankId rb = static_cast<BankId>(n);
                emit(t, mk_nmc(static_cast<uint32_t>(NmcOp::REDUCE_F32),
                               make_shared_addr(rb, SH::slots(l)),
                               make_shared_addr(rb, SH::accr(l)),
                               FJ, T::NMC_D, 0.0f, "NMC reduce"));
                emit(t, mk_acquire(T::NMC_D, 1, "reduce done"));
                emit(t, mk_dma(make_shared_addr(rb, SH::accr(l)),
                               make_local_addr(t, L::ACC32),
                               1, 1024, 0, 0, "acc pull"));
                emit(t, mk_dma_fence());
            } else {
                emit(t, mk_vpu(Op::VPU_ADD_F32, L::SLOTS + 0 * 1024,
                               L::SLOTS + 1 * 1024, L::ACC32));
                for (int j = 2; j < FJ; j++)
                    emit(t, mk_vpu(Op::VPU_ADD_F32, L::ACC32, L::SLOTS + j * 1024,
                                   L::ACC32));
            }
            emit(t, mk_vpu(Op::VPU_ADD_F32, L::ACC32, L::R32, L::RES32));
            emit(t, mk_vpu(Op::VPU_CVT_F32_F16, L::RES32, 0, L::OUT16));
            emit(t, st_blk(t, L::OUT16, XN, DM, 0, n, "out->DRAM"));
            emit(t, mk_dma_fence());
            if (l + 1 < p.layers) {
                for (int c = 0; c < 12; c++)
                    emit(t, mk_release(static_cast<TileId>(c), T::X_RDY, "next layer"));
                if (p.stream_weights) {
                    // Extra tokens gate the leaders' next prefetch batch
                    // (double-buffer parity safety).
                    for (int g = 0; g < NUM_BANKS; g++)
                        emit(t, mk_release(static_cast<TileId>(4 * g),
                                           T::X_RDY, "stream gate"));
                }
            }
        }
    }

    for (int t = 0; t < NUM_TILES; t++)
        prog.add(static_cast<TileId>(t), mk_halt());
    return prog;
}

// ═══ CPU reference (schedule mirror, bit-exact) ══════════════

namespace {

void get_blk(const f16* M, int ld, int r, int c, f16* out) {
    for (int i = 0; i < BN; i++)
        std::memcpy(out + i * BN, M + (r * BN + i) * ld + c * BN, BN * 2);
}

void put_blk(f16* M, int ld, int r, int c, const f16* in) {
    for (int i = 0; i < BN; i++)
        std::memcpy(M + (r * BN + i) * ld + c * BN, in + i * BN, BN * 2);
}

} // namespace

std::vector<f16> llm_reference(const LlmParams& p, const LlmData& d) {
    namespace bo = blockops;
    constexpr int DM = LlmParams::DM, FF = LlmParams::FF;
    constexpr int KB = DM / BN, NH = LlmParams::HEADS, FJ = FF / BN;
    constexpr int SEQ = LlmParams::SEQ;

    std::vector<f16> X = d.X;   // SEQ x DM, updated per layer

    for (int l = 0; l < p.layers; l++) {
        const auto& w = d.w[l];
        f16 xb[256], wb[256];

        // Projections.
        f16 q16[NH][256], kt16[NH][256], v16[NH][256];
        for (int h = 0; h < NH; h++) {
            const f16* Ws[3] = {w.Wq.data(), w.Wk.data(), w.Wv.data()};
            for (int role = 0; role < 3; role++) {
                float acc[256];
                for (int k = 0; k < KB; k++) {
                    get_blk(X.data(), DM, 0, k, xb);
                    get_blk(Ws[role], DM, k, h, wb);
                    bo::mxu(reinterpret_cast<uint8_t*>(xb), false,
                            reinterpret_cast<uint8_t*>(wb), acc, k > 0);
                }
                f16 out16[256];
                bo::cvt_f32_f16(acc, out16);
                if (role == 0) std::memcpy(q16[h], out16, 512);
                else if (role == 1) bo::transpose_f16(out16, kt16[h]);
                else std::memcpy(v16[h], out16, 512);
            }
        }

        // Attention per head.
        f16 o16[NH][256];
        for (int h = 0; h < NH; h++) {
            float s32[256], p32[256], o32[256];
            f16 p16[256];
            bo::mxu(reinterpret_cast<uint8_t*>(q16[h]), false,
                    reinterpret_cast<uint8_t*>(kt16[h]), s32, false);
            bo::softmax_f32(s32, LlmParams::ATT_SCALE, p32);
            bo::cvt_f32_f16(p32, p16);
            bo::mxu(reinterpret_cast<uint8_t*>(p16), false,
                    reinterpret_cast<uint8_t*>(v16[h]), o32, false);
            bo::cvt_f32_f16(o32, o16[h]);
        }

        // Output projection + residual.
        float r32[NH][256];
        for (int n = 0; n < NH; n++) {
            float attn[256];
            for (int h = 0; h < NH; h++) {
                get_blk(w.Wo.data(), DM, h, n, wb);
                bo::mxu(reinterpret_cast<uint8_t*>(o16[h]), false,
                        reinterpret_cast<uint8_t*>(wb), attn, h > 0);
            }
            f16 xr[256];
            get_blk(X.data(), DM, 0, n, xr);
            float x32[256];
            bo::cvt_f16_f32(xr, x32);
            bo::add_f32(attn, x32, r32[n]);
        }

        // LayerNorm over full rows; repack to f16 blocks.
        std::vector<float> ln_in(SEQ * DM), ln_out(SEQ * DM);
        for (int n = 0; n < NH; n++)
            for (int i = 0; i < BN; i++)
                std::memcpy(&ln_in[i * DM + n * BN], &r32[n][i * BN], BN * 4);
        bo::layernorm_f32(ln_in.data(), KB, LlmParams::LN_EPS, ln_out.data());
        std::vector<f16> ln16(SEQ * DM);
        for (int k = 0; k < KB; k++) {
            float t32[256];
            f16 t16[256];
            for (int i = 0; i < BN; i++)
                std::memcpy(&t32[i * BN], &ln_out[i * DM + k * BN], BN * 4);
            bo::cvt_f32_f16(t32, t16);
            put_blk(ln16.data(), DM, 0, k, t16);
        }

        // FFN1 + FFN2 partials.
        float part[FJ][NH][256];
        for (int j = 0; j < FJ; j++) {
            float h32[256];
            for (int k = 0; k < KB; k++) {
                get_blk(ln16.data(), DM, 0, k, xb);
                get_blk(w.W1.data(), FF, k, j, wb);
                bo::mxu(reinterpret_cast<uint8_t*>(xb), false,
                        reinterpret_cast<uint8_t*>(wb), h32, k > 0);
            }
            bo::gelu_f32(h32, h32);
            f16 h16[256];
            bo::cvt_f32_f16(h32, h16);
            for (int n = 0; n < NH; n++) {
                get_blk(w.W2.data(), DM, j, n, wb);
                bo::mxu(reinterpret_cast<uint8_t*>(h16), false,
                        reinterpret_cast<uint8_t*>(wb), part[j][n], false);
            }
        }

        // Reduction (fixed j order) + residual + writeback.
        std::vector<f16> Xn(SEQ * DM);
        for (int n = 0; n < NH; n++) {
            float acc[256], res[256];
            bo::add_f32(part[0][n], part[1][n], acc);
            for (int j = 2; j < FJ; j++) bo::add_f32(acc, part[j][n], acc);
            bo::add_f32(acc, r32[n], res);
            f16 out16[256];
            bo::cvt_f32_f16(res, out16);
            put_blk(Xn.data(), DM, 0, n, out16);
        }
        X = std::move(Xn);
    }
    return X;
}

// ═══ Independent float64 reference (numerical sanity) ════════

std::vector<double> llm_reference_f64(const LlmParams& p, const LlmData& d) {
    constexpr int DM = LlmParams::DM, FF = LlmParams::FF;
    constexpr int NH = LlmParams::HEADS, DH = LlmParams::DH;
    constexpr int SEQ = LlmParams::SEQ;

    auto matmul = [](const std::vector<double>& A, const std::vector<double>& B,
                     int M, int K, int N) {
        std::vector<double> C(M * N, 0.0);
        for (int i = 0; i < M; i++)
            for (int k = 0; k < K; k++)
                for (int j = 0; j < N; j++)
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
        return C;
    };
    auto to_f64 = [](const std::vector<f16>& v) {
        std::vector<double> o(v.size());
        for (size_t i = 0; i < v.size(); i++) o[i] = v[i].to_f32();
        return o;
    };

    std::vector<double> X = to_f64(d.X);
    for (int l = 0; l < p.layers; l++) {
        const auto& w = d.w[l];
        auto Q = matmul(X, to_f64(w.Wq), SEQ, DM, DM);
        auto K = matmul(X, to_f64(w.Wk), SEQ, DM, DM);
        auto V = matmul(X, to_f64(w.Wv), SEQ, DM, DM);

        std::vector<double> O(SEQ * DM, 0.0);
        for (int h = 0; h < NH; h++) {
            for (int i = 0; i < SEQ; i++) {
                double s[SEQ];
                double mx = -1e300;
                for (int j2 = 0; j2 < SEQ; j2++) {
                    double dot = 0;
                    for (int k = 0; k < DH; k++)
                        dot += Q[i * DM + h * DH + k] * K[j2 * DM + h * DH + k];
                    s[j2] = dot * LlmParams::ATT_SCALE;
                    mx = std::max(mx, s[j2]);
                }
                double sum = 0;
                for (int j2 = 0; j2 < SEQ; j2++) { s[j2] = std::exp(s[j2] - mx); sum += s[j2]; }
                for (int k = 0; k < DH; k++) {
                    double acc = 0;
                    for (int j2 = 0; j2 < SEQ; j2++)
                        acc += (s[j2] / sum) * V[j2 * DM + h * DH + k];
                    O[i * DM + h * DH + k] = acc;
                }
            }
        }
        auto attn = matmul(O, to_f64(w.Wo), SEQ, DM, DM);
        std::vector<double> R(SEQ * DM);
        for (int i = 0; i < SEQ * DM; i++) R[i] = X[i] + attn[i];

        std::vector<double> LN(SEQ * DM);
        for (int i = 0; i < SEQ; i++) {
            double mean = 0, var = 0;
            for (int j = 0; j < DM; j++) mean += R[i * DM + j];
            mean /= DM;
            for (int j = 0; j < DM; j++) {
                double t = R[i * DM + j] - mean;
                var += t * t;
            }
            var /= DM;
            double inv = 1.0 / std::sqrt(var + LlmParams::LN_EPS);
            for (int j = 0; j < DM; j++) LN[i * DM + j] = (R[i * DM + j] - mean) * inv;
        }

        auto H = matmul(LN, to_f64(w.W1), SEQ, DM, FF);
        constexpr double kg = 0.7978845608028654;
        for (auto& x : H)
            x = 0.5 * x * (1.0 + std::tanh(kg * (x + 0.044715 * x * x * x)));
        auto F = matmul(H, to_f64(w.W2), SEQ, FF, DM);
        for (int i = 0; i < SEQ * DM; i++) X[i] = R[i] + F[i];
    }
    return X;
}

} // namespace mobol::cycle
