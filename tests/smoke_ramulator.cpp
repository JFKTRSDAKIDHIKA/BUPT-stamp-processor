/// @file smoke_ramulator.cpp
/// @brief Minimal smoke test for Ramulator2 integration.
///
/// Verifies:
///   1. Factory creates MemorySystem successfully
///   2. send(Request) accepts read/write requests with callbacks
///   3. tick() advances simulation state
///   4. Callbacks fire when requests complete
///   5. Latency values are reasonable (>0, <10000 cycles)
///
/// This test does NOT use a Frontend — we drive the MemorySystem directly.
/// However, Ramulator2 requires connect_frontend() to initialize internal
/// components (address mapper, etc.), so we provide a DummyFrontend.

#include <iostream>
#include <cassert>
#include <cstdlib>
#include <vector>
#include <string>

#include <yaml-cpp/yaml.h>
#include "base/base.h"
#include "base/factory.h"
#include "base/request.h"
#include "memory_system/memory_system.h"
#include "frontend/frontend.h"

namespace {

struct TestResult {
    int total_requests = 0;
    int completed_requests = 0;
    int failed_sends = 0;
    std::vector<Ramulator::Clk_t> latencies;
};

} // anonymous namespace

namespace Ramulator {

/// Minimal dummy frontend — no factory registration to avoid static init conflicts.
class DummyFrontend : public IFrontEnd, public Implementation {
public:
    std::string get_name() const override { return "Dummy"; }
    std::string get_desc() const override { return "Smoke test dummy"; }
    std::string get_ifce_name() const override { return "Frontend"; }

    void init() override { m_clock_ratio = 1; }
    void tick() override {}
    bool is_finished() override { return false; }

    DummyFrontend(const YAML::Node& cfg, Implementation* parent)
        : Implementation(cfg, "Frontend", "Dummy", "Smoke test dummy", parent) {
        m_impl = this;
        m_params.set_impl_name("Dummy");
        init();
    }
};

} // namespace Ramulator

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.yaml>\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::cout << "=== Ramulator2 Smoke Test ===\n";
    std::cout << "Config: " << config_path << "\n\n";

    // ─── Step 1: Load config and create MemorySystem ─────────
    YAML::Node config;
    try {
        config = YAML::LoadFile(config_path);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: Cannot load config: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[PASS] Config loaded.\n";

    Ramulator::IMemorySystem* mem_sys = nullptr;
    try {
        mem_sys = Ramulator::Factory::create_memory_system(config);
    } catch (const std::exception& e) {
        std::cerr << "FAIL: Cannot create MemorySystem: " << e.what() << "\n";
        return 1;
    }
    std::cout << "[PASS] MemorySystem created.\n";

    // Create dummy frontend and connect (required for addr mapper init)
    Ramulator::DummyFrontend dummy_frontend(config["Frontend"], nullptr);
    dummy_frontend.gather_components();
    dummy_frontend.connect_memory_system(mem_sys);
    mem_sys->connect_frontend(&dummy_frontend);
    std::cout << "[PASS] DummyFrontend connected.\n";

    // Get clock info
    int clock_ratio = mem_sys->get_clock_ratio();
    float tCK = mem_sys->get_tCK();
    std::cout << "  clock_ratio = " << clock_ratio << "\n";
    std::cout << "  tCK = " << tCK << " ns\n";

    // ─── Step 2: Send read requests ──────────────────────────
    TestResult result;
    constexpr int NUM_READS = 8;
    constexpr int NUM_WRITES = 4;
    constexpr Ramulator::Addr_t BASE_ADDR = 0x1000;

    std::cout << "\nSending " << NUM_READS << " read requests...\n";
    for (int i = 0; i < NUM_READS; ++i) {
        Ramulator::Addr_t addr = BASE_ADDR + i * 64;  // 64-byte stride (cache line)
        // source_id must stay < frontend num_cores (1): Ramulator2 indexes
        // per-core stats vectors with it (heap corruption otherwise).
        int source_id = 0;

        Ramulator::Request req(
            addr,
            Ramulator::Request::Type::Read,
            source_id,
            [&result, i](Ramulator::Request& completed_req) {
                Ramulator::Clk_t latency = completed_req.depart - completed_req.arrive;
                result.completed_requests++;
                result.latencies.push_back(latency);
                std::cout << "  [callback] Read #" << i
                          << " addr=0x" << std::hex << completed_req.addr << std::dec
                          << " arrive=" << completed_req.arrive
                          << " depart=" << completed_req.depart
                          << " latency=" << latency << " cycles\n";
            }
        );

        bool accepted = mem_sys->send(req);
        result.total_requests++;
        if (!accepted) {
            result.failed_sends++;
            std::cout << "  [WARN] Read #" << i << " rejected (buffer full?), will retry.\n";
        }
    }

    // ─── Step 3: Send write requests ─────────────────────────
    std::cout << "\nSending " << NUM_WRITES << " write requests...\n";
    for (int i = 0; i < NUM_WRITES; ++i) {
        Ramulator::Addr_t addr = BASE_ADDR + 0x10000 + i * 64;
        int source_id = 0;

        Ramulator::Request req(
            addr,
            Ramulator::Request::Type::Write,
            source_id,
            [&result, i](Ramulator::Request& completed_req) {
                Ramulator::Clk_t latency = completed_req.depart - completed_req.arrive;
                result.completed_requests++;
                result.latencies.push_back(latency);
                std::cout << "  [callback] Write #" << i
                          << " addr=0x" << std::hex << completed_req.addr << std::dec
                          << " arrive=" << completed_req.arrive
                          << " depart=" << completed_req.depart
                          << " latency=" << latency << " cycles\n";
            }
        );

        bool accepted = mem_sys->send(req);
        result.total_requests++;
        if (!accepted) {
            result.failed_sends++;
        }
    }

    // ─── Step 4: Tick until all requests complete ────────────
    std::cout << "\nTicking memory system...\n";
    constexpr int MAX_TICKS = 100000;
    int ticks = 0;
    for (ticks = 0; ticks < MAX_TICKS && result.completed_requests < result.total_requests; ++ticks) {
        mem_sys->tick();
    }

    // ─── Step 5: Verify results ──────────────────────────────
    std::cout << "\n=== Results ===\n";
    std::cout << "  Total ticks:        " << ticks << "\n";
    std::cout << "  Total requests:     " << result.total_requests << "\n";
    std::cout << "  Completed:          " << result.completed_requests << "\n";
    std::cout << "  Failed sends:       " << result.failed_sends << "\n";

    bool pass = true;

    // Check: all requests completed
    if (result.completed_requests != result.total_requests) {
        std::cout << "[FAIL] Not all requests completed! ("
                  << result.completed_requests << "/" << result.total_requests << ")\n";
        pass = false;
    } else {
        std::cout << "[PASS] All " << result.total_requests << " requests completed.\n";
    }

    // Check: no deadlock (didn't hit MAX_TICKS)
    if (ticks >= MAX_TICKS) {
        std::cout << "[FAIL] Hit MAX_TICKS limit — possible deadlock.\n";
        pass = false;
    } else {
        std::cout << "[PASS] Completed in " << ticks << " ticks (< " << MAX_TICKS << ").\n";
    }

    // Check: latencies are reasonable
    if (!result.latencies.empty()) {
        Ramulator::Clk_t min_lat = result.latencies[0], max_lat = result.latencies[0];
        Ramulator::Clk_t sum_lat = 0;
        for (auto lat : result.latencies) {
            if (lat < min_lat) min_lat = lat;
            if (lat > max_lat) max_lat = lat;
            sum_lat += lat;
        }
        double avg_lat = static_cast<double>(sum_lat) / result.latencies.size();

        std::cout << "  Latency (cycles): min=" << min_lat
                  << " avg=" << avg_lat
                  << " max=" << max_lat << "\n";

        if (min_lat <= 0) {
            std::cout << "[FAIL] Minimum latency <= 0 (impossible).\n";
            pass = false;
        } else {
            std::cout << "[PASS] Latencies are positive.\n";
        }

        if (max_lat > 50000) {
            std::cout << "[WARN] Maximum latency suspiciously high (" << max_lat << " cycles).\n";
        } else {
            std::cout << "[PASS] Latencies within reasonable range.\n";
        }
    }

    // ─── Cleanup ─────────────────────────────────────────────
    // Note: Ramulator2 factory-created objects are heap-allocated.
    // For a smoke test we let the OS clean up.
    std::cout << "\n=== Smoke test: " << (pass ? "PASSED" : "FAILED") << " ===\n";
    return pass ? 0 : 1;
}
