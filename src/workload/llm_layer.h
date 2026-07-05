/// @file llm_layer.h
/// @brief Transformer-decoder inference workload for the cycle simulator.
///
/// Model (per layer, prefill of one 16-token sequence):
///   d_model = 64, heads = 4, d_head = 16, FFN = 256, seq = 16, no affine LN.
///     QKV  : Q/K/V = X * Wq/Wk/Wv          (12 output blocks, tiles 0-11)
///     Attn : per head h on tile h: S = Q K^T, P = softmax(S/4), O = P V
///     Proj : attn = concat(O) * Wo         (4 blocks, tiles 0-3, all-to-all O push)
///     Res1 : R = X + attn                  (f32, stays resident on tiles 0-3)
///     LN   : gathered on tile 0 (rows span the whole d_model), result
///            replicated into all 4 SHARED banks => every tile reads its
///            NEAR bank (NUMA-aware placement per spec §5)
///     FFN1 : H = gelu(LN * W1)             (16 blocks, all 16 tiles)
///     FFN2 : out = H * W2 + R              (K=16 cross-tile reduction:
///            every tile pushes 4 partial blocks; reducers 0-3 join with
///            arity 16 and add in fixed j-order)
///   Layer boundary: reducers release X_RDY to the 12 projection tiles.
///
/// Precision chain: weights/activations f16, MXU accumulate f32, softmax/
/// LN/GELU in f32, activations converted back to f16 between ops.
///
/// NOTE (architecture gap): softmax/LayerNorm/GELU need a vector unit that
/// the MOBOL spec does not define. The simulator adds a per-tile VPU
/// (16 f32 lanes) as a spec extension — without it, an LLM cannot run on
/// this architecture at all.
#pragma once

#include "cycle/isa.h"
#include "common/f16.h"
#include "common/memory_model.h"
#include <vector>

namespace mobol::cycle {

/// Architecture paradigm under study.
enum class LlmArch {
    BASELINE,   ///< 3-die: buffer-die SRAM banks, all compute on base die
    NO_BUFFER,  ///< 2-die: no shared SRAM, intermediates round-trip DRAM
    NMC,        ///< 3-die + near-memory compute on the buffer die
                ///< (bank-side fused LayerNorm + fixed-order reduction)
};

struct LlmParams {
    int layers = 2;
    uint32_t seed = 42;
    LlmArch arch = LlmArch::BASELINE;
    /// Weight streaming: buffer-die bank DMA engines prefetch each
    /// layer's weights DRAM -> near-bank SRAM through their own vertical
    /// columns (double-buffered, overlapped with the previous layer),
    /// bypassing the base-die tile ports entirely. Orthogonal to `arch`
    /// (combinable with BASELINE and NMC; meaningless with NO_BUFFER).
    bool stream_weights = false;
    static constexpr int SEQ = 16;
    static constexpr int DM = 64;      ///< d_model
    static constexpr int HEADS = 4;
    static constexpr int DH = 16;      ///< d_head
    static constexpr int FF = 256;     ///< FFN hidden
    static constexpr float LN_EPS = 1e-5f;
    static constexpr float ATT_SCALE = 0.25f;  ///< 1/sqrt(d_head)
};

/// Host-side model data (also the input to the CPU reference).
struct LlmData {
    std::vector<f16> X;                 ///< SEQ x DM
    struct LayerW {
        std::vector<f16> Wq, Wk, Wv, Wo; ///< DM x DM
        std::vector<f16> W1;             ///< DM x FF
        std::vector<f16> W2;             ///< FF x DM
    };
    std::vector<LayerW> w;
};

/// DRAM placement.
struct LlmDram {
    static constexpr Addr ACT_BASE = 0x000000;     ///< X_l at ACT_BASE + l*ACT_STRIDE
    static constexpr Addr ACT_STRIDE = 0x1000;
    static constexpr Addr L_BASE = 0x100000;       ///< per-layer weights
    static constexpr Addr L_STRIDE = 0x40000;
    static constexpr Addr WQ = 0x0000, WK = 0x2000, WV = 0x4000, WO = 0x6000;
    static constexpr Addr W1 = 0x8000, W2 = 0x10000;
};

/// Deterministic model generation (seeded).
LlmData llm_generate(const LlmParams& p);

/// Write X and all layer weights into DRAM (via MemoryModel).
void llm_load_dram(MemoryModel& mem, const LlmParams& p, const LlmData& d);

/// Build the per-tile programs for `p.layers` decoder layers.
Program build_llm_program(const LlmParams& p);

/// CPU reference with the exact schedule/precision mirror (bit-exact
/// against the simulator). All three paradigms move the same values
/// through the same arithmetic in the same order — only data placement
/// and the executing engine differ — so ONE reference validates all of
/// them bit-for-bit. Returns the final activations (SEQ x DM f16).
std::vector<f16> llm_reference(const LlmParams& p, const LlmData& d);

/// Independent float64 model (naive, no tiling) for numerical sanity.
std::vector<double> llm_reference_f64(const LlmParams& p, const LlmData& d);

} // namespace mobol::cycle
