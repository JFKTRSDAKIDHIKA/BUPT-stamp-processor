/// @file test_functional.cpp
/// @brief §5.1 Functional correctness verification.
///
/// Generates A, B, D with fixed seed, runs both:
///   1. Functional Engine (simulator path)
///   2. CPU Reference (matching precision chain)
/// Compares C (f32, bit-exact) and E (f16, bit-exact).

#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>
#include <random>
#include <iomanip>

#include "common/types.h"
#include "common/f16.h"
#include "common/memory_model.h"
#include "common/sync_manager.h"
#include "common/functional_engine.h"
#include "workload/dual_gemm.h"
#include "workload/cpu_reference.h"

using namespace mobol;

static constexpr int DIM = 32;
static constexpr uint32_t SEED = 42;

/// Generate random f16 values in [-1.0, 1.0] using deterministic seed.
static std::vector<f16> generate_random_matrix(uint32_t seed, int rows, int cols) {
    std::mt19937 rng(seed);
    // Generate uniform floats in [-1, 1]
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<f16> mat(rows * cols);
    for (auto& v : mat) {
        v = f16(dist(rng));
    }
    return mat;
}

/// Compare two f32 arrays bit-exact.
static bool compare_f32_exact(const float* a, const float* b, int count,
                               const char* name, int max_print = 5) {
    int mismatches = 0;
    for (int i = 0; i < count; i++) {
        uint32_t bits_a, bits_b;
        std::memcpy(&bits_a, &a[i], sizeof(float));
        std::memcpy(&bits_b, &b[i], sizeof(float));
        if (bits_a != bits_b) {
            if (mismatches < max_print) {
                int row = i / DIM, col = i % DIM;
                std::cerr << "  MISMATCH " << name << "[" << row << "][" << col << "]: "
                          << "sim=0x" << std::hex << bits_a
                          << " ref=0x" << bits_b << std::dec
                          << " sim=" << a[i] << " ref=" << b[i]
                          << "\n";
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        std::cerr << "  Total " << name << " mismatches: " << mismatches << "/" << count << "\n";
        return false;
    }
    return true;
}

/// Compare two f16 arrays bit-exact.
static bool compare_f16_exact(const f16* a, const f16* b, int count,
                               const char* name, int max_print = 5) {
    int mismatches = 0;
    for (int i = 0; i < count; i++) {
        if (a[i].bits != b[i].bits) {
            if (mismatches < max_print) {
                int row = i / DIM, col = i % DIM;
                std::cerr << "  MISMATCH " << name << "[" << row << "][" << col << "]: "
                          << "sim=0x" << std::hex << a[i].bits
                          << " ref=0x" << std::hex << b[i].bits << std::dec
                          << " sim=" << a[i].to_f32() << " ref=" << b[i].to_f32()
                          << "\n";
            }
            mismatches++;
        }
    }
    if (mismatches > 0) {
        std::cerr << "  Total " << name << " mismatches: " << mismatches << "/" << count << "\n";
        return false;
    }
    return true;
}

int main() {
    std::cout << "=== §5.1 Functional Correctness Test ===\n";
    std::cout << "Seed: " << SEED << "\n";
    std::cout << "Matrices: " << DIM << "x" << DIM << " f16, MXU: 16x16x16\n\n";

    // ─── Step 1: Generate input matrices ─────────────────────
    std::cout << "Generating A, B, D with seed " << SEED << "...\n";
    auto A = generate_random_matrix(SEED + 0, DIM, DIM);
    auto B = generate_random_matrix(SEED + 1, DIM, DIM);
    auto D = generate_random_matrix(SEED + 2, DIM, DIM);

    // ─── Step 2: Run CPU reference ───────────────────────────
    std::cout << "Running CPU reference implementation...\n";
    std::vector<float> C_ref(DIM * DIM);
    std::vector<f16>   E_ref(DIM * DIM);
    cpu_dual_gemm_reference(A.data(), B.data(), D.data(), C_ref.data(), E_ref.data());

    // ─── Step 3: Run Functional Engine ───────────────────────
    std::cout << "Running Functional Engine (simulator path)...\n";
    MemoryModel mem;
    SyncManager sync;
    FunctionalEngine engine(mem);

    // Load inputs to DRAM
    load_inputs_to_dram(mem, A.data(), B.data(), D.data());

    // Execute the dual GEMM schedule
    execute_dual_gemm(mem, sync, engine);

    // Read results
    std::vector<float> C_sim(DIM * DIM);
    std::vector<f16>   E_sim(DIM * DIM);
    read_C_from_shared(mem, C_sim.data());
    read_output_from_dram(mem, E_sim.data());

    // ─── Step 4: Compare results ─────────────────────────────
    std::cout << "\n=== Comparison ===\n";
    bool c_pass = compare_f32_exact(C_sim.data(), C_ref.data(), DIM * DIM, "C (f32)");
    bool e_pass = compare_f16_exact(E_sim.data(), E_ref.data(), DIM * DIM, "E (f16)");

    std::cout << "\n  C (f32, intermediate): " << (c_pass ? "PASS ✓" : "FAIL ✗") << "\n";
    std::cout << "  E (f16, final):        " << (e_pass ? "PASS ✓" : "FAIL ✗") << "\n";

    // ─── Step 5: Verify DRAM traffic (C should not appear) ───
    // dram_write_bytes_ counts all writes. A+B+D = 3 × 2048 = 6144 reads.
    // E = 2048 bytes written. C should NOT be in DRAM traffic.
    // After loading inputs (write) and reading E, the DRAM write should include:
    //   - Input load: 3 × 2048 = 6144 bytes (A, B, D written to DRAM)
    //   - E output: 2048 bytes (E written from tile LOCAL to DRAM)
    //   - Total: 8192 bytes
    std::cout << "\n  DRAM writes: " << mem.dram_write_count() << " bytes\n";
    std::cout << "  DRAM reads:  " << mem.dram_read_count() << " bytes\n";

    // Verify C never went to DRAM:
    // Total writes = 6144 (inputs) + 2048 (E) = 8192
    // If C went to DRAM, writes would be >= 8192 + 4096 = 12288
    bool dram_pass = (mem.dram_write_count() == 8192);
    std::cout << "  C fusion (no DRAM write): " << (dram_pass ? "PASS ✓" : "FAIL ✗") << "\n";

    bool all_pass = c_pass && e_pass && dram_pass;
    std::cout << "\n=== §5.1 Functional Test: "
              << (all_pass ? "PASSED" : "FAILED") << " ===\n";

    return all_pass ? 0 : 1;
}
