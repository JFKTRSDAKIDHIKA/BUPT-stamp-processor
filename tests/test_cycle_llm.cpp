/// @file test_cycle_llm.cpp
/// @brief End-to-end LLM (2-layer transformer decoder prefill) on the
///        cycle-accurate simulator.
///
/// Checks:
///   1. Final activations bit-exact vs the schedule-mirror CPU reference.
///   2. Close to an independent float64 model (guards against a bug
///      mirrored into both the simulator and the reference).
///   3. Causality holds; simulation drains completely.

#include "cycle/chip.h"
#include "workload/llm_layer.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

using namespace mobol;
using namespace mobol::cycle;

int main(int argc, char** argv) {
    std::string ramulator_cfg = argc > 1 ? argv[1] : "";

    LlmParams p;
    p.layers = 2;
    p.seed = 42;

    LlmData data = llm_generate(p);
    std::vector<f16> ref = llm_reference(p, data);
    std::vector<double> ref64 = llm_reference_f64(p, data);
    const int N = LlmParams::SEQ * LlmParams::DM;

    // All three paradigms move the same values through the same
    // arithmetic in the same order — one reference validates them all.
    struct Variant { LlmArch arch; const char* name; bool nmc; bool stream; };
    const Variant variants[] = {
        {LlmArch::BASELINE,  "baseline (3-die)",           false, false},
        {LlmArch::NO_BUFFER, "no-buffer-die (2-die)",      false, false},
        {LlmArch::NMC,       "buffer-die NMC (3-die+)",    true,  false},
        {LlmArch::BASELINE,  "baseline + weight-stream",   false, true},
        {LlmArch::NMC,       "NMC + weight-stream",        true,  true},
    };

    bool pass = true;
    for (const auto& v : variants) {
        p.arch = v.arch;
        p.stream_weights = v.stream;
        CycleConfig cfg;
        cfg.ramulator_config = ramulator_cfg;
        cfg.nmc_enable = v.nmc;
        Chip chip(cfg);
        llm_load_dram(chip.mem(), p, data);

        Program prog = build_llm_program(p);
        std::cout << "\n──── " << v.name << ": " << prog.total_instrs()
                  << " instructions, " << p.layers << " layers ────\n";
        ChipSummary s = chip.run(prog);
        chip.print_summary(s);

        std::vector<f16> out(N);
        chip.mem().read(make_dram_addr(LlmDram::ACT_BASE + p.layers * LlmDram::ACT_STRIDE),
                        out.data(), N * 2);

        if (std::memcmp(out.data(), ref.data(), N * 2) != 0) {
            int diffs = 0;
            for (int i = 0; i < N; i++)
                if (out[i].bits != ref[i].bits) {
                    if (diffs < 5)
                        std::cerr << "  diff@" << i << ": sim=" << out[i].to_f32()
                                  << " ref=" << ref[i].to_f32() << "\n";
                    diffs++;
                }
            std::cerr << "FAIL [" << v.name << "]: output != CPU reference ("
                      << diffs << "/" << N << ")\n";
            pass = false;
        } else {
            std::cout << "[PASS] " << v.name
                      << ": final activations bit-exact vs CPU reference\n";
        }

        double max_err = 0.0, max_mag = 0.0;
        for (int i = 0; i < N; i++) {
            max_err = std::max(max_err, std::abs(out[i].to_f32() - ref64[i]));
            max_mag = std::max(max_mag, std::abs(ref64[i]));
        }
        double rel = max_err / (max_mag > 0 ? max_mag : 1.0);
        if (rel > 0.05) {
            std::cerr << "FAIL [" << v.name << "]: rel err vs float64 " << rel << "\n";
            pass = false;
        } else {
            std::cout << "[PASS] " << v.name << ": within f16 tolerance of "
                         "float64 model (rel " << rel << ")\n";
        }
        if (!s.causality_ok) {
            std::cerr << "FAIL [" << v.name << "]: causality violations\n";
            pass = false;
        } else {
            std::cout << "[PASS] " << v.name << ": causality verified ("
                      << chip.events().releases.size() << " releases, "
                      << chip.events().acquire_passes.size() << " acquires)\n";
        }
    }

    std::cout << "\n=== cycle LLM inference (5 variants): "
              << (pass ? "PASSED" : "FAILED") << " ===\n";
    return pass ? 0 : 1;
}
