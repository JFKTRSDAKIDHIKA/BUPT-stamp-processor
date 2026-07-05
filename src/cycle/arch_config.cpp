/// @file arch_config.cpp
/// @brief Unified architecture YAML loader.

#include "cycle/arch_config.h"

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace mobol::cycle {

namespace {

[[noreturn]] void fail(const std::string& msg) {
    throw std::runtime_error("arch config: " + msg);
}

void expect_eq(const YAML::Node& n, const char* key, int compiled,
               const char* name) {
    if (!n[key]) return;  // optional; only validated if present
    int v = n[key].as<int>();
    if (v != compiled) {
        std::ostringstream os;
        os << "structural." << key << " = " << v
           << " but this build was compiled with " << name << " = " << compiled
           << ". Topology constants are compile-time; rebuild to change them.";
        fail(os.str());
    }
}

} // namespace

CycleConfig load_arch_config(const std::string& path) {
    YAML::Node y;
    try {
        y = YAML::LoadFile(path);
    } catch (const std::exception& e) {
        fail(std::string("cannot load '") + path + "': " + e.what());
    }

    CycleConfig c;

    // ── structural validation (compile-time constants) ──
    if (y["structural"]) {
        const auto s = y["structural"];
        expect_eq(s, "num_tiles", NUM_TILES, "NUM_TILES");
        expect_eq(s, "num_banks", NUM_BANKS, "NUM_BANKS");
        expect_eq(s, "tiles_per_group", TILES_PER_GROUP, "TILES_PER_GROUP");
        expect_eq(s, "mxu_m", MXU_M, "MXU_M");
        expect_eq(s, "mxu_n", MXU_N, "MXU_N");
        expect_eq(s, "mxu_k", MXU_K, "MXU_K");
    }

    // ── tile compute ──
    if (auto t = y["tile"]) {
        if (t["mxu_latency"]) c.mxu_latency = t["mxu_latency"].as<int>();
        if (t["mxu_issue_interval"]) c.mxu_issue_interval = t["mxu_issue_interval"].as<int>();
        if (t["compute_uses_sram_ports"])
            c.compute_uses_sram_ports = t["compute_uses_sram_ports"].as<bool>();
    }
    if (auto v = y["vpu"]) {
        if (v["add_cycles"]) c.vpu_add_cycles = v["add_cycles"].as<int>();
        if (v["convert_cycles"]) c.vpu_convert_cycles = v["convert_cycles"].as<int>();
        if (v["transpose_cycles"]) c.vpu_transpose_cycles = v["transpose_cycles"].as<int>();
        if (v["scale_cycles"]) c.vpu_scale_cycles = v["scale_cycles"].as<int>();
        if (v["softmax_cycles"]) c.vpu_softmax_cycles = v["softmax_cycles"].as<int>();
        if (v["gelu_cycles"]) c.vpu_gelu_cycles = v["gelu_cycles"].as<int>();
        if (v["layernorm_cycles"]) c.vpu_layernorm_cycles = v["layernorm_cycles"].as<int>();
        if (v["mul_cycles"]) c.vpu_mul_cycles = v["mul_cycles"].as<int>();
        if (v["silu_cycles"]) c.vpu_silu_cycles = v["silu_cycles"].as<int>();
        if (v["rmsnorm_cycles"]) c.vpu_rmsnorm_cycles = v["rmsnorm_cycles"].as<int>();
        if (v["rope_cycles"]) c.vpu_rope_cycles = v["rope_cycles"].as<int>();
    }
    if (auto l = y["local_sram"]) {
        if (l["rd_ports"]) c.local_rd_ports = l["rd_ports"].as<int>();
        if (l["wr_ports"]) c.local_wr_ports = l["wr_ports"].as<int>();
        if (l["port_width"]) c.sram_port_width = l["port_width"].as<int>();
    }
    if (auto s = y["shared_sram"]) {
        if (s["port_contention"]) c.shared_port_contention = s["port_contention"].as<bool>();
        if (s["rw_ports"]) c.shared_rw_ports = s["rw_ports"].as<int>();
    }
    if (auto n = y["noc"]) {
        if (n["flit_bytes"]) c.noc_flit_bytes = n["flit_bytes"].as<int>();
        if (n["hop_latency"]) c.noc_hop_latency = n["hop_latency"].as<int>();
        if (n["buffer_depth"]) c.noc_buffer_depth = n["buffer_depth"].as<int>();
        if (n["inject_queue_depth"]) c.noc_inject_queue_depth = n["inject_queue_depth"].as<int>();
    }
    if (auto v = y["vbond"]) {
        if (v["latency"]) c.vbond_latency = v["latency"].as<int>();
        if (v["queue_depth"]) c.vbond_queue_depth = v["queue_depth"].as<int>();
        if (v["flits_per_cycle"]) c.vbond_flits_per_cycle = v["flits_per_cycle"].as<int>();
        if (v["dram_flits_per_cycle"]) c.vbond_dram_flits_per_cycle = v["dram_flits_per_cycle"].as<int>();
        if (v["dram_col_queue_depth"]) c.dram_col_queue_depth = v["dram_col_queue_depth"].as<int>();
    }
    if (auto d = y["dma"]) {
        if (d["setup_cycles"]) c.dma_setup_cycles = d["setup_cycles"].as<int>();
        if (d["queue_depth"]) c.dma_queue_depth = d["queue_depth"].as<int>();
        if (d["max_inflight_chunks"]) c.dma_max_inflight_chunks = d["max_inflight_chunks"].as<int>();
        if (d["chunks_per_cycle"]) c.dma_chunks_per_cycle = d["chunks_per_cycle"].as<int>();
    }
    if (auto d = y["dram"]) {
        if (d["txn_bytes"]) c.dram_txn_bytes = d["txn_bytes"].as<int>();
        if (d["ctrl_queue_depth"]) c.dram_ctrl_queue_depth = d["ctrl_queue_depth"].as<int>();
        if (d["issue_width"]) c.dram_issue_width = d["issue_width"].as<int>();
        if (d["ticks_per_logic"]) c.dram_ticks_per_logic = d["ticks_per_logic"].as<int>();
        if (d["ramulator_config"]) {
            std::string rc = d["ramulator_config"].as<std::string>();
            // Resolve relative to the arch YAML's directory.
            std::filesystem::path p(rc);
            if (p.is_relative()) {
                std::filesystem::path base(path);
                p = base.parent_path().parent_path() / rc;  // config/ -> repo root
                if (!std::filesystem::exists(p))
                    p = base.parent_path() / std::filesystem::path(rc).filename();
            }
            c.ramulator_config = std::filesystem::exists(p) ? p.string() : rc;
        }
    }
    if (auto n = y["nmc"]) {
        if (n["enable"]) c.nmc_enable = n["enable"].as<bool>();
        if (n["reduce_cycles_per_block"]) c.nmc_reduce_cycles_per_block = n["reduce_cycles_per_block"].as<int>();
        if (n["ln_cycles_per_block"]) c.nmc_ln_cycles_per_block = n["ln_cycles_per_block"].as<int>();
        if (n["cvt_cycles_per_block"]) c.nmc_cvt_cycles_per_block = n["cvt_cycles_per_block"].as<int>();
        if (n["queue_depth"]) c.nmc_queue_depth = n["queue_depth"].as<int>();
    }
    if (auto s = y["sync"]) {
        if (s["max_live_tags"]) c.max_live_tags = s["max_live_tags"].as<int>();
    }
    if (auto s = y["sim"]) {
        if (s["max_cycles"]) c.max_cycles = s["max_cycles"].as<Cycle>();
        if (s["deadlock_window"]) c.deadlock_window = s["deadlock_window"].as<Cycle>();
        if (s["collect_trace"]) c.collect_trace = s["collect_trace"].as<bool>();
    }
    return c;
}

} // namespace mobol::cycle
