/// @file functional_engine.cpp

#include "common/functional_engine.h"
#include <cstring>
#include <cassert>

namespace mobol {

// ─── DMA ─────────────────────────────────────────────────────

void FunctionalEngine::dma_copy(Addr src_addr, Addr dst_addr, size_t size) {
    // Read from source, write to destination.
    // For overlapping LOCAL scratchpad regions, use a temp buffer.
    std::vector<uint8_t> buf(size);
    mem_.read(src_addr, buf.data(), size);
    mem_.write(dst_addr, buf.data(), size);
}

// ─── MXU: f16 × f16 → f32 ───────────────────────────────────

void FunctionalEngine::mxu_f16xf16_to_f32(Addr a_addr, Addr b_addr, Addr c_addr) {
    // A[i][k] at a_addr + (i*TILE + k) * sizeof(f16)
    // B[k][j] at b_addr + (k*TILE + j) * sizeof(f16)
    // C[i][j] at c_addr + (i*TILE + j) * sizeof(float)

    const auto* A = reinterpret_cast<const f16*>(mem_.local_ptr(a_addr));
    const auto* B = reinterpret_cast<const f16*>(mem_.local_ptr(b_addr));
    auto* C = reinterpret_cast<float*>(mem_.local_ptr(c_addr));

    for (int i = 0; i < TILE; i++) {
        for (int j = 0; j < TILE; j++) {
            float acc = 0.0f;
            for (int k = 0; k < TILE; k++) {
                float a_val = A[i * TILE + k].to_f32();
                float b_val = B[k * TILE + j].to_f32();
                acc += a_val * b_val;
            }
            C[i * TILE + j] = acc;
        }
    }
}

// ─── MXU: f32 × f16 → f32 ───────────────────────────────────

void FunctionalEngine::mxu_f32xf16_to_f32(Addr a_addr, Addr b_addr, Addr c_addr) {
    // A is f32 (C from GEMM1), B is f16 (D slice), output is f32
    const auto* A = reinterpret_cast<const float*>(mem_.local_ptr(a_addr));
    const auto* B = reinterpret_cast<const f16*>(mem_.local_ptr(b_addr));
    auto* C = reinterpret_cast<float*>(mem_.local_ptr(c_addr));

    for (int i = 0; i < TILE; i++) {
        for (int j = 0; j < TILE; j++) {
            float acc = 0.0f;
            for (int k = 0; k < TILE; k++) {
                float a_val = A[i * TILE + k];          // f32 directly
                float b_val = B[k * TILE + j].to_f32(); // f16 → f32
                acc += a_val * b_val;
            }
            C[i * TILE + j] = acc;
        }
    }
}

// ─── Reduce: f32 + f32 → f32 ────────────────────────────────

void FunctionalEngine::reduce_f32(Addr src_a, Addr src_b, Addr dst) {
    const auto* A = reinterpret_cast<const float*>(mem_.local_ptr(src_a));
    const auto* B = reinterpret_cast<const float*>(mem_.local_ptr(src_b));
    auto* C = reinterpret_cast<float*>(mem_.local_ptr(dst));

    for (int idx = 0; idx < TILE * TILE; idx++) {
        C[idx] = A[idx] + B[idx];
    }
}

// ─── Convert: f32 block → f16 block ─────────────────────────

void FunctionalEngine::convert_f32_to_f16(Addr src_f32, Addr dst_f16) {
    const auto* src = reinterpret_cast<const float*>(mem_.local_ptr(src_f32));
    auto* dst = reinterpret_cast<f16*>(mem_.local_ptr(dst_f16));

    for (int idx = 0; idx < TILE * TILE; idx++) {
        dst[idx] = f16(src[idx]);
    }
}

} // namespace mobol
