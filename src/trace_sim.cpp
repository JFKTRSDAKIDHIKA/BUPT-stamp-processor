/// @file trace_sim.cpp
/// @brief Run a compiler-emitted trace on the cycle-accurate simulator.
///
/// Usage:
///   trace_sim <base>.trace <base>.mem --ramulator cfg.yaml [--out-dir DIR]
///             [--json]
///
/// Loads the DRAM image, parses the trace (config directives inside it may
/// set ramulator/ports/etc.), runs the chip, dumps every .dump region to
/// DIR/<name>.bin for the Python golden replayer, and prints a summary
/// (optionally as JSON for the compiler's DSE loop to parse).

#include "cycle/chip.h"
#include "cycle/trace_loader.h"
#include "cycle/arch_config.h"

#include <iostream>
#include <string>

using namespace mobol;
using namespace mobol::cycle;

int main(int argc, char** argv) {
    std::string trace_path, mem_path, ramulator, out_dir = ".";
    std::string arch_yaml = "config/mobol_arch.yaml";
    bool json = false;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << a << " needs a value\n"; exit(1); }
            return argv[++i];
        };
        if (a == "--ramulator") ramulator = next();
        else if (a == "--arch-config") arch_yaml = next();
        else if (a == "--out-dir") out_dir = next();
        else if (a == "--json") json = true;
        else if (trace_path.empty()) trace_path = a;
        else if (mem_path.empty()) mem_path = a;
        else { std::cerr << "unexpected arg: " << a << "\n"; return 1; }
    }
    if (trace_path.empty() || mem_path.empty()) {
        std::cerr << "usage: trace_sim <base>.trace <base>.mem "
                     "[--arch-config arch.yaml] [--ramulator cfg.yaml] "
                     "[--out-dir DIR] [--json]\n";
        return 1;
    }

    // Layer the config: architecture YAML (base) -> trace .config -> CLI.
    CycleConfig base;
    try {
        base = load_arch_config(arch_yaml);
    } catch (const std::exception& e) {
        std::cerr << "[warn] " << e.what() << "; using built-in defaults\n";
    }
    TraceProgram tp = load_trace(trace_path, base);
    if (!ramulator.empty()) tp.cfg.ramulator_config = ramulator;
    if (tp.cfg.ramulator_config.empty()) {
        std::cerr << "no ramulator config (in arch yaml, trace .config or --ramulator)\n";
        return 1;
    }

    Chip chip(tp.cfg);
    load_mem_image(chip.mem(), mem_path);

    ChipSummary s;
    try {
        s = chip.run(tp.prog);
    } catch (const std::exception& e) {
        std::cerr << "SIMULATION ERROR: " << e.what() << "\n";
        return 2;
    }

    for (const auto& d : tp.dumps)
        dump_region(chip.mem(), d.dram_off, d.bytes, out_dir + "/" + d.name + ".bin");

    if (json) {
        std::cout << "{"
                  << "\"total_cycles\":" << s.total_cycles
                  << ",\"instrs\":" << s.total_instrs
                  << ",\"mxu_ops\":" << chip.events().mxu_ops
                  << ",\"mxu_util\":" << s.mxu_utilization
                  << ",\"noc_flit_hops\":" << s.noc_flit_hops
                  << ",\"noc_max_link_util\":" << s.max_noc_link_utilization
                  << ",\"dram_reads\":" << s.dram_reads
                  << ",\"dram_read_bytes\":" << s.dram_read_bytes
                  << ",\"dram_write_bytes\":" << s.dram_write_bytes
                  << ",\"dram_avg_read_latency\":" << s.dram_avg_read_latency
                  << ",\"causality_ok\":" << (s.causality_ok ? "true" : "false")
                  << ",\"tiles\":[";
        for (int t = 0; t < NUM_TILES; t++) {
            const auto& ts = s.tiles[t];
            std::cout << (t ? "," : "") << "{"
                      << "\"instrs\":" << ts.instrs
                      << ",\"busy\":" << ts.busy_cycles
                      << ",\"stall_acquire\":" << ts.stall_acquire
                      << ",\"stall_dma\":" << ts.stall_dma
                      << ",\"stall_mxu\":" << ts.stall_mxu
                      << ",\"stall_vpu\":" << ts.stall_vpu
                      << ",\"stall_inject\":" << ts.stall_inject
                      << ",\"idle_after_halt\":" << ts.idle_after_halt
                      << ",\"mxu_ops\":" << ts.mxu_ops
                      << ",\"mxu_busy\":" << ts.mxu_busy_cycles
                      << ",\"vpu_busy\":" << ts.vpu_busy_cycles
                      << ",\"dma_busy\":" << ts.dma_busy_cycles
                      << ",\"mxu_dma_overlap\":" << ts.mxu_dma_overlap_cycles
                      << ",\"dma_bytes\":" << ts.dma_bytes
                      << "}";
        }
        std::cout << "]}\n";
    } else {
        chip.print_summary(s);
    }
    return s.causality_ok ? 0 : 3;
}
