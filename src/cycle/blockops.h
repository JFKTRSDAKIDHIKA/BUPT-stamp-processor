/// @file blockops.h
/// @brief Functional 16x16-block primitives shared by the cycle simulator
///        (MXU/VPU datapaths) and CPU reference implementations.
///
/// Numerical policy (MOBOL spec §4 / UNDERSTANDING.md):
///   - MXU: f16 (or f32) x f16 inputs, f32 accumulation, fixed k order.
///   - VPU: 16 f32 lanes; f16 conversions round-to-nearest-even via f16.h.
/// Using the identical code on both sides makes bit-exact comparison a
/// test of the *simulator's dataflow*, not of libm.
#pragma once

#include "common/f16.h"
#include <cmath>
#include <cstdint>
#include <cstring>

namespace mobol::cycle::blockops {

constexpr int BN = 16; ///< native block dimension

/// C(16x16 f32) = [acc ? C : 0] + A(16x16) x B(16x16 f16), f32 accumulate.
/// A is f16 when a_is_f32 == false, else f32. Fixed accumulation order:
/// k ascending inside the dot product (bit-reproducible).
inline void mxu(const uint8_t* A, bool a_is_f32, const uint8_t* B,
                float* C, bool acc) {
    const f16* a16 = reinterpret_cast<const f16*>(A);
    const float* a32 = reinterpret_cast<const float*>(A);
    const f16* b16 = reinterpret_cast<const f16*>(B);
    for (int i = 0; i < BN; i++) {
        for (int j = 0; j < BN; j++) {
            float s = acc ? C[i * BN + j] : 0.0f;
            for (int k = 0; k < BN; k++) {
                float a = a_is_f32 ? a32[i * BN + k] : a16[i * BN + k].to_f32();
                float b = b16[k * BN + j].to_f32();
                s += a * b;
            }
            C[i * BN + j] = s;
        }
    }
}

inline void add_f32(const float* a, const float* b, float* d) {
    for (int i = 0; i < BN * BN; i++) d[i] = a[i] + b[i];
}

inline void add_f16(const f16* a, const f16* b, f16* d) {
    for (int i = 0; i < BN * BN; i++)
        d[i] = f16(a[i].to_f32() + b[i].to_f32());
}

inline void cvt_f32_f16(const float* a, f16* d) {
    for (int i = 0; i < BN * BN; i++) d[i] = f16(a[i]);
}

inline void cvt_f16_f32(const f16* a, float* d) {
    for (int i = 0; i < BN * BN; i++) d[i] = a[i].to_f32();
}

inline void transpose_f16(const f16* a, f16* d) {
    for (int i = 0; i < BN; i++)
        for (int j = 0; j < BN; j++)
            d[j * BN + i] = a[i * BN + j];
}

inline void scale_f32(const float* a, float s, float* d) {
    for (int i = 0; i < BN * BN; i++) d[i] = a[i] * s;
}

/// Row-wise softmax over a 16x16 f32 block: d = softmax(a * scale) with
/// the numerically stable max-subtraction form, fixed evaluation order.
inline void softmax_f32(const float* a, float scale, float* d) {
    for (int i = 0; i < BN; i++) {
        const float* row = a + i * BN;
        float m = row[0] * scale;
        for (int j = 1; j < BN; j++) m = std::max(m, row[j] * scale);
        float sum = 0.0f;
        float e[BN];
        for (int j = 0; j < BN; j++) {
            e[j] = std::exp(row[j] * scale - m);
            sum += e[j];
        }
        for (int j = 0; j < BN; j++) d[i * BN + j] = e[j] / sum;
    }
}

/// tanh-approximation GELU, elementwise f32.
inline void gelu_f32(const float* a, float* d) {
    constexpr float k = 0.7978845608028654f; // sqrt(2/pi)
    for (int i = 0; i < BN * BN; i++) {
        float x = a[i];
        d[i] = 0.5f * x * (1.0f + std::tanh(k * (x + 0.044715f * x * x * x)));
    }
}

/// LayerNorm (no affine) over rows of width 16*count: `a` is a row-major
/// 16 x (16*count) f32 matrix. eps passed via `eps`.
inline void layernorm_f32(const float* a, int count, float eps, float* d) {
    const int W = BN * count;
    for (int i = 0; i < BN; i++) {
        const float* row = a + i * W;
        float mean = 0.0f;
        for (int j = 0; j < W; j++) mean += row[j];
        mean /= static_cast<float>(W);
        float var = 0.0f;
        for (int j = 0; j < W; j++) {
            float t = row[j] - mean;
            var += t * t;
        }
        var /= static_cast<float>(W);
        float inv = 1.0f / std::sqrt(var + eps);
        for (int j = 0; j < W; j++) d[i * W + j] = (row[j] - mean) * inv;
    }
}

// ─── Compiler-target ops (BLOCKED layout) ────────────────────
// `count` consecutive 16x16 row-major blocks tile one 16 x (16*count)
// logical row group side by side: element (i, j) with j in [0,16*count)
// lives at block (j/16), local index (i*16 + j%16). This matches how the
// tile datapath keeps a wide activation across adjacent scratchpad blocks.
inline float blk_get(const float* a, int i, int j) {
    int b = j / BN, jj = j % BN;
    return a[b * BN * BN + i * BN + jj];
}
inline void blk_set(float* d, int i, int j, float v) {
    int b = j / BN, jj = j % BN;
    d[b * BN * BN + i * BN + jj] = v;
}

inline void mul_f32(const float* a, const float* b, float* d) {
    for (int i = 0; i < BN * BN; i++) d[i] = a[i] * b[i];
}

inline void silu_f32(const float* a, float* d) {
    for (int i = 0; i < BN * BN; i++) {
        float x = a[i];
        d[i] = x / (1.0f + std::exp(-x));
    }
}

/// RMSNorm (no affine) over blocked rows of width 16*count.
inline void rmsnorm_blk(const float* a, int count, float eps, float* d) {
    const int W = BN * count;
    for (int i = 0; i < BN; i++) {
        float ss = 0.0f;
        for (int j = 0; j < W; j++) { float v = blk_get(a, i, j); ss += v * v; }
        float inv = 1.0f / std::sqrt(ss / static_cast<float>(W) + eps);
        for (int j = 0; j < W; j++) blk_set(d, i, j, blk_get(a, i, j) * inv);
    }
}

/// Blocked row softmax with optional causal mask and sliding window.
///   valid_cols>0 limits the row to its first valid_cols entries.
///   row0_pos>=0 enables causal masking: query row i is at absolute
///   position row0_pos+i, key column j is masked if j>row0_pos+i, or (when
///   window>0) if j <= row0_pos+i-window.
inline void softmax_blk(const float* a, int count, float scale, int valid_cols,
                        int row0_pos, int window, float* d) {
    const int W = BN * count;
    const int cols = valid_cols > 0 ? valid_cols : W;
    for (int i = 0; i < BN; i++) {
        int hi = cols;
        if (row0_pos >= 0) hi = std::min(cols, row0_pos + i + 1);
        int lo = 0;
        if (row0_pos >= 0 && window > 0) lo = std::max(0, row0_pos + i - window + 1);
        float m = -std::numeric_limits<float>::infinity();
        for (int j = lo; j < hi; j++) m = std::max(m, blk_get(a, i, j) * scale);
        float sum = 0.0f;
        for (int j = 0; j < W; j++) {
            if (j >= lo && j < hi) {
                float e = std::exp(blk_get(a, i, j) * scale - m);
                blk_set(d, i, j, e);
                sum += e;
            } else {
                blk_set(d, i, j, 0.0f);
            }
        }
        float inv = sum > 0 ? 1.0f / sum : 0.0f;
        for (int j = 0; j < W; j++) blk_set(d, i, j, blk_get(d, i, j) * inv);
    }
}

/// Rotate-half RoPE on one 16-wide head block. `x` rows are tokens, its 16
/// columns are the head dims. `half` = head_dim/2; column k<half of cos/sin
/// holds cos/sin(pos, k). Pairs dim k with dim k+half.
inline void rope_block_f16(const f16* x, const float* cosb, const float* sinb,
                           int half, f16* out) {
    for (int i = 0; i < BN; i++) {
        for (int k = 0; k < half; k++) {
            float a = x[i * BN + k].to_f32();
            float b = x[i * BN + half + k].to_f32();
            float c = cosb[i * BN + k], s = sinb[i * BN + k];
            out[i * BN + k]        = f16(a * c - b * s);
            out[i * BN + half + k] = f16(b * c + a * s);
        }
    }
}

} // namespace mobol::cycle::blockops
