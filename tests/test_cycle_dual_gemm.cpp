/// @file test_cycle_dual_gemm.cpp
/// @brief Cycle-accurate dual-GEMM: bit-exact numerics + timing sanity.
///
/// Checks:
///   1. C (f32 in SHARED) and E (f16 in DRAM) match the CPU reference
///      bit-for-bit — the execution-driven datapath computed the right
///      values through real DMA/NoC/DRAM movement.
///   2. C never appears in DRAM traffic (fusion stays on-chip).
///   3. Causality holds (acquires after releases, DMA spans positive).
///   4. Cycle count is positive and plausible.

#include "cycle/chip.h"
#include "workload/dual_gemm_cycle.h"
#include "workload/dual_gemm.h"
#include "workload/cpu_reference.h"

#include <cstring>
#include <iostream>
#include <random>
#include <vector>

using namespace mobol;
using namespace mobol::cycle;

static constexpr int DIM = 32;
static constexpr uint32_t SEED = 42;

static std::vector<f16> gen(uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<f16> m(DIM * DIM);
    for (auto& v : m) v = f16(dist(rng));
    return m;
}

int main(int argc, char** argv) {
    std::string ramulator_cfg = argc > 1 ? argv[1] : "";

    auto A = gen(SEED), B = gen(SEED + 1), D = gen(SEED + 2);

    // CPU reference (independent implementation, identical precision chain).
    std::vector<float> C_ref(DIM * DIM);
    std::vector<f16> E_ref(DIM * DIM);
    cpu_dual_gemm_reference(A.data(), B.data(), D.data(), C_ref.data(), E_ref.data());

    // Cycle-accurate run.
    CycleConfig cfg;
    cfg.ramulator_config = ramulator_cfg;
    Chip chip(cfg);
    chip.mem().write(make_dram_addr(DramLayout::A_BASE), A.data(), DIM * DIM * 2);
    chip.mem().write(make_dram_addr(DramLayout::B_BASE), B.data(), DIM * DIM * 2);
    chip.mem().write(make_dram_addr(DramLayout::D_BASE), D.data(), DIM * DIM * 2);

    Program prog = build_dual_gemm_program();
    ChipSummary s = chip.run(prog);
    chip.print_summary(s);

    std::vector<float> C_sim(DIM * DIM);
    std::vector<f16> E_sim(DIM * DIM);
    chip.mem().read(make_shared_addr(SharedLayout::C_BANK, SharedLayout::C_BASE_OFF),
                    C_sim.data(), DIM * DIM * 4);
    chip.mem().read(make_dram_addr(DramLayout::E_BASE), E_sim.data(), DIM * DIM * 2);

    bool pass = true;

    // 1. Bit-exact numerics.
    if (std::memcmp(C_sim.data(), C_ref.data(), DIM * DIM * 4) != 0) {
        std::cerr << "FAIL: C mismatch vs CPU reference\n";
        for (int i = 0; i < DIM * DIM; i++)
            if (C_sim[i] != C_ref[i]) {
                std::cerr << "  first diff at " << i << ": sim=" << C_sim[i]
                          << " ref=" << C_ref[i] << "\n";
                break;
            }
        pass = false;
    } else {
        std::cout << "[PASS] C (f32, SHARED) bit-exact vs CPU reference\n";
    }
    if (std::memcmp(E_sim.data(), E_ref.data(), DIM * DIM * 2) != 0) {
        std::cerr << "FAIL: E mismatch vs CPU reference\n";
        pass = false;
    } else {
        std::cout << "[PASS] E (f16, DRAM) bit-exact vs CPU reference\n";
    }

    // 2. Fusion: DRAM traffic is exactly A,B,D block reads + E writes.
    //    Reads: 3 matrices x 4 tiles-loads... measured in 64 B transactions;
    //    just assert E-sized writes and no C-sized (4 KB) write volume.
    uint64_t expect_write_bytes = DIM * DIM * 2; // E only
    if (s.dram_write_bytes != expect_write_bytes) {
        std::cerr << "FAIL: DRAM write bytes = " << s.dram_write_bytes
                  << ", expected " << expect_write_bytes << " (E only; C must stay on-chip)\n";
        pass = false;
    } else {
        std::cout << "[PASS] C fused in SHARED: DRAM writes = E only ("
                  << s.dram_write_bytes << " B)\n";
    }
    // Each of A,B,D is read as 8 16x16 blocks of f16 (16 rows x 32 B).
    uint64_t expect_read_bytes = 3 * 8 * 16 * 32;
    if (s.dram_read_bytes != expect_read_bytes) {
        std::cerr << "FAIL: DRAM read bytes = " << s.dram_read_bytes
                  << ", expected " << expect_read_bytes << "\n";
        pass = false;
    } else {
        std::cout << "[PASS] DRAM reads = A+B+D blocks only ("
                  << s.dram_read_bytes << " B)\n";
    }

    // 3. Causality.
    if (!s.causality_ok) {
        std::cerr << "FAIL: causality violations: " << s.causality_violations << "\n";
        pass = false;
    } else {
        std::cout << "[PASS] causality verified ("
                  << chip.events().releases.size() << " releases, "
                  << chip.events().acquire_passes.size() << " acquires)\n";
    }

    // 4. Timing sanity.
    if (s.total_cycles <= 0 || s.total_cycles > 1'000'000) {
        std::cerr << "FAIL: implausible cycle count " << s.total_cycles << "\n";
        pass = false;
    } else {
        std::cout << "[PASS] total cycles = " << s.total_cycles << "\n";
    }
    if (chip.events().mxu_ops != 16) { // 8 blocks/GEMM x 2 GEMMs
        std::cerr << "FAIL: MXU op count " << chip.events().mxu_ops << " != 16\n";
        pass = false;
    }

    std::cout << "\n=== cycle dual-GEMM: " << (pass ? "PASSED" : "FAILED") << " ===\n";
    return pass ? 0 : 1;
}
