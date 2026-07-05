/// @file main.cpp
/// @brief MOBOL cycle-accurate simulator CLI.
///
/// Usage:
///   mobol_sim --workload dual_gemm [--ramulator cfg.yaml]
///   mobol_sim --workload llm [--layers N] [--ramulator cfg.yaml]
///             [--shared-contention] [--compute-sram-ports]
///
/// Runs the workload on the cycle-accurate chip model, verifies numerics
/// against the CPU reference, and prints the timing summary.

#include "cycle/chip.h"
#include "cycle/arch_config.h"
#include "workload/dual_gemm_cycle.h"
#include "workload/dual_gemm.h"
#include "workload/cpu_reference.h"
#include "workload/llm_layer.h"

#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace mobol;
using namespace mobol::cycle;

namespace {

int run_dual_gemm(const CycleConfig& cfg, bool use_shared) {
    constexpr int DIM = 32;
    auto gen = [](uint32_t seed) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        std::vector<f16> m(DIM * DIM);
        for (auto& v : m) v = f16(dist(rng));
        return m;
    };
    auto A = gen(42), B = gen(43), D = gen(44);

    std::vector<float> C_ref(DIM * DIM);
    std::vector<f16> E_ref(DIM * DIM);
    cpu_dual_gemm_reference(A.data(), B.data(), D.data(), C_ref.data(), E_ref.data());

    Chip chip(cfg);
    chip.mem().write(make_dram_addr(DramLayout::A_BASE), A.data(), DIM * DIM * 2);
    chip.mem().write(make_dram_addr(DramLayout::B_BASE), B.data(), DIM * DIM * 2);
    chip.mem().write(make_dram_addr(DramLayout::D_BASE), D.data(), DIM * DIM * 2);

    ChipSummary s = chip.run(build_dual_gemm_program(use_shared));
    chip.print_summary(s);

    std::vector<f16> E_sim(DIM * DIM);
    chip.mem().read(make_dram_addr(DramLayout::E_BASE), E_sim.data(), DIM * DIM * 2);
    bool ok = std::memcmp(E_sim.data(), E_ref.data(), DIM * DIM * 2) == 0
              && s.causality_ok;
    std::cout << "dual_gemm: " << (ok ? "BIT-EXACT vs CPU reference, causality OK"
                                      : "FAILED") << "\n";
    return ok ? 0 : 1;
}

int run_llm(const CycleConfig& cfg, int layers, LlmArch arch, bool stream_w) {
    LlmParams p;
    p.layers = layers;
    p.arch = arch;
    p.stream_weights = stream_w;
    LlmData data = llm_generate(p);
    std::vector<f16> ref = llm_reference(p, data);

    Chip chip(cfg);
    llm_load_dram(chip.mem(), p, data);
    Program prog = build_llm_program(p);
    std::cout << "LLM decoder prefill: " << layers << " layers, "
              << prog.total_instrs() << " instructions "
              << "(seq=16, d_model=64, heads=4, ffn=256)\n";

    ChipSummary s = chip.run(prog);
    chip.print_summary(s);

    const int N = LlmParams::SEQ * LlmParams::DM;
    std::vector<f16> out(N);
    chip.mem().read(make_dram_addr(LlmDram::ACT_BASE + layers * LlmDram::ACT_STRIDE),
                    out.data(), N * 2);
    bool ok = std::memcmp(out.data(), ref.data(), N * 2) == 0 && s.causality_ok;
    std::cout << "llm: " << (ok ? "BIT-EXACT vs CPU reference, causality OK"
                                : "FAILED") << "\n";
    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    std::string workload = "dual_gemm";
    int layers = 2;
    LlmArch arch = LlmArch::BASELINE;
    bool stream_w = false;

    // Base config from the unified architecture YAML (single source of
    // truth). --arch-config picks a different file; CLI flags below then
    // override individual fields. Falls back to built-in defaults if the
    // default file is absent (e.g. run from an unusual cwd).
    std::string arch_yaml = "config/mobol_arch.yaml";
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "--arch-config" && i + 1 < argc)
            arch_yaml = argv[i + 1];
    CycleConfig cfg;
    try {
        cfg = load_arch_config(arch_yaml);
    } catch (const std::exception& e) {
        std::cerr << "[warn] " << e.what() << "; using built-in defaults\n";
    }

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << a << " needs a value\n"; exit(1); }
            return argv[++i];
        };
        if (a == "--arch-config") { next(); continue; }  // already handled
        if (a == "--workload") workload = next();
        else if (a == "--ramulator") cfg.ramulator_config = next();
        else if (a == "--layers") layers = std::stoi(next());
        else if (a == "--shared-contention") cfg.shared_port_contention = true;
        else if (a == "--compute-sram-ports") cfg.compute_uses_sram_ports = true;
        else if (a == "--dram-density") {
            // Hybrid-bond density per group column, in bits per logic
            // cycle (one 64 B flit = 512 bits): 512/1024/2048/4096/8192.
            int bits = std::stoi(next());
            if (bits % 512 != 0 || bits <= 0) {
                std::cerr << "--dram-density must be a positive multiple of 512\n";
                return 1;
            }
            cfg.vbond_dram_flits_per_cycle = bits / 512;
        }
        else if (a == "--dram-txn") cfg.dram_txn_bytes = std::stoi(next());
        else if (a == "--arch") {
            std::string v = next();
            if (v == "baseline") arch = LlmArch::BASELINE;
            else if (v == "nobuffer") arch = LlmArch::NO_BUFFER;
            else if (v == "nmc") { arch = LlmArch::NMC; cfg.nmc_enable = true; }
            else { std::cerr << "--arch: baseline|nobuffer|nmc\n"; return 1; }
        }
        else if (a == "--stream-weights") stream_w = true;
        else if (a == "--tile-link") cfg.vbond_flits_per_cycle = std::stoi(next());
        else if (a == "--wr-ports") cfg.local_wr_ports = std::stoi(next());
        else if (a == "--rd-ports") cfg.local_rd_ports = std::stoi(next());
        else if (a == "--dma-rate") cfg.dma_chunks_per_cycle = std::stoi(next());
        else if (a == "--help") {
            std::cout << "usage: mobol_sim --workload dual_gemm|llm "
                         "[--layers N] [--ramulator cfg.yaml] "
                         "[--arch baseline|nobuffer|nmc] "
                         "[--dram-density BITS] [--dram-txn BYTES] "
                         "[--stream-weights] [--tile-link N] [--wr-ports N] [--rd-ports N] "
                         "[--dma-rate N] "
                         "[--shared-contention] [--compute-sram-ports]\n";
            return 0;
        } else {
            std::cerr << "unknown flag: " << a << "\n";
            return 1;
        }
    }

    std::cout << "MOBOL cycle-accurate simulator — 16 tiles / 4 shared banks / "
                 "3D-stacked DRAM die, column density "
              << cfg.vbond_dram_flits_per_cycle * 512 << " bits/cycle\n";
    if (workload == "dual_gemm")
        return run_dual_gemm(cfg, arch != LlmArch::NO_BUFFER);
    if (workload == "llm") return run_llm(cfg, layers, arch, stream_w);
    std::cerr << "unknown workload: " << workload << "\n";
    return 1;
}
