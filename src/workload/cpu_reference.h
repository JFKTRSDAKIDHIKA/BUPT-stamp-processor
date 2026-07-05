/// @file cpu_reference.h
/// @brief CPU reference implementation for dual GEMM verification.
///
/// Precision chain MUST match the simulator exactly:
///   GEMM1: f16(A) × f16(B) → f32 accumulate, K order 0→1, f32 reduce → f32 C
///   GEMM2: f32(C) × f16(D) → f32 accumulate, K order 0→1, f32 reduce → f16 E
#pragma once

#include "common/types.h"
#include "common/f16.h"

namespace mobol {

/// Compute E_ref = (A × B) × D on CPU with the exact same precision chain
/// as the simulator.
///
/// @param A, B, D  Input matrices (32×32 f16, row-major)
/// @param C_out    Output intermediate C (32×32 f32, row-major)
/// @param E_out    Output final E (32×32 f16, row-major)
void cpu_dual_gemm_reference(
    const f16* A, const f16* B, const f16* D,
    float* C_out, f16* E_out);

} // namespace mobol
