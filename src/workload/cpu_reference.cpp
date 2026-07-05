/// @file cpu_reference.cpp
/// @brief CPU reference implementation with exact precision matching.

#include "workload/cpu_reference.h"
#include <vector>
#include <cstring>

namespace mobol {

static constexpr int DIM = 32;
static constexpr int TILE = 16;
static constexpr int TILES_M = 2;
static constexpr int TILES_N = 2;

void cpu_dual_gemm_reference(
    const f16* A, const f16* B, const f16* D,
    float* C_out, f16* E_out)
{
    // ─── GEMM1: C = A × B ────────────────────────────────────
    // Same tiling and K order as the simulator.

    for (int m = 0; m < TILES_M; m++) {
        for (int n = 0; n < TILES_N; n++) {
            // f32 accumulator for the output tile
            float acc[TILE][TILE] = {};

            // K slice 0: A[m,0] × B[0,n]
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < TILE; kk++) {
                        float a_val = A[(m * TILE + i) * DIM + (0 * TILE + kk)].to_f32();
                        float b_val = B[(0 * TILE + kk) * DIM + (n * TILE + j)].to_f32();
                        sum += a_val * b_val;
                    }
                    acc[i][j] += sum;
                }
            }

            // K slice 1: A[m,1] × B[1,n]
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < TILE; kk++) {
                        float a_val = A[(m * TILE + i) * DIM + (1 * TILE + kk)].to_f32();
                        float b_val = B[(1 * TILE + kk) * DIM + (n * TILE + j)].to_f32();
                        sum += a_val * b_val;
                    }
                    acc[i][j] += sum;
                }
            }

            // Write C block as f32 (NO truncation)
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
                    C_out[(m * TILE + i) * DIM + (n * TILE + j)] = acc[i][j];
                }
            }
        }
    }

    // ─── GEMM2: E = C × D ────────────────────────────────────
    // C is f32, D is f16, E output is f16.

    for (int m = 0; m < TILES_M; m++) {
        for (int n = 0; n < TILES_N; n++) {
            float acc[TILE][TILE] = {};

            // K slice 0: C[m,0] × D[0,n]
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < TILE; kk++) {
                        float c_val = C_out[(m * TILE + i) * DIM + (0 * TILE + kk)];
                        float d_val = D[(0 * TILE + kk) * DIM + (n * TILE + j)].to_f32();
                        sum += c_val * d_val;
                    }
                    acc[i][j] += sum;
                }
            }

            // K slice 1: C[m,1] × D[1,n]
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
                    float sum = 0.0f;
                    for (int kk = 0; kk < TILE; kk++) {
                        float c_val = C_out[(m * TILE + i) * DIM + (1 * TILE + kk)];
                        float d_val = D[(1 * TILE + kk) * DIM + (n * TILE + j)].to_f32();
                        sum += c_val * d_val;
                    }
                    acc[i][j] += sum;
                }
            }

            // Write E block as f16 (truncation happens here)
            for (int i = 0; i < TILE; i++) {
                for (int j = 0; j < TILE; j++) {
                    E_out[(m * TILE + i) * DIM + (n * TILE + j)] = f16(acc[i][j]);
                }
            }
        }
    }
}

} // namespace mobol
