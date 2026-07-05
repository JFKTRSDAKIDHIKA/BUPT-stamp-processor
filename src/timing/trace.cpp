/// @file trace.cpp

#include "timing/trace.h"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace mobol {

static const char* event_type_str(TraceEventType t) {
    switch (t) {
        case TraceEventType::DMA_START:      return "DMA_START";
        case TraceEventType::DMA_COMPLETE:    return "DMA_COMPLETE";
        case TraceEventType::MXU_LAUNCH:      return "MXU_LAUNCH";
        case TraceEventType::MXU_COMPLETE:    return "MXU_COMPLETE";
        case TraceEventType::RELEASE:         return "RELEASE";
        case TraceEventType::ACQUIRE_BLOCK:   return "ACQUIRE_BLOCK";
        case TraceEventType::ACQUIRE_PASS:    return "ACQUIRE_PASS";
        case TraceEventType::DRAM_REQ_SENT:   return "DRAM_REQ_SENT";
        case TraceEventType::DRAM_REQ_DONE:   return "DRAM_REQ_DONE";
        case TraceEventType::TILE_START:      return "TILE_START";
        case TraceEventType::TILE_DONE:       return "TILE_DONE";
    }
    return "UNKNOWN";
}

void write_trace(const std::vector<TraceEvent>& events, const std::string& path) {
    std::ofstream ofs(path);
    if (!ofs) {
        std::cerr << "ERROR: Cannot open trace file: " << path << "\n";
        return;
    }

    // Header
    ofs << "# MOBOL Cycle-Level Trace\n";
    ofs << "# cycle | type | tile | details\n";

    for (const auto& ev : events) {
        ofs << std::setw(8) << ev.timestamp
            << " | " << std::setw(14) << std::left << event_type_str(ev.type)
            << " | T" << std::setw(2) << std::right << static_cast<int>(ev.tile_id);

        switch (ev.type) {
            case TraceEventType::DMA_START:
            case TraceEventType::DMA_COMPLETE:
                ofs << " | src=0x" << std::hex << ev.src_addr
                    << " dst=0x" << ev.dst_addr << std::dec
                    << " bytes=" << ev.size_bytes;
                break;
            case TraceEventType::MXU_LAUNCH:
            case TraceEventType::MXU_COMPLETE:
                ofs << " | " << ev.note;
                break;
            case TraceEventType::RELEASE:
                ofs << " | tag=" << ev.tag_index << " cnt=" << ev.join_count;
                break;
            case TraceEventType::ACQUIRE_BLOCK:
            case TraceEventType::ACQUIRE_PASS:
                ofs << " | tag=" << ev.tag_index << " arity=" << ev.arity
                    << " cnt=" << ev.join_count;
                break;
            case TraceEventType::DRAM_REQ_SENT:
            case TraceEventType::DRAM_REQ_DONE:
                ofs << " | addr=0x" << std::hex << ev.src_addr << std::dec
                    << " bytes=" << ev.size_bytes
                    << " dram_lat=" << ev.dram_latency;
                break;
            case TraceEventType::TILE_START:
            case TraceEventType::TILE_DONE:
                break;
        }
        ofs << "\n";
    }

    ofs.close();
    std::cout << "Trace written to: " << path << " (" << events.size() << " events)\n";
}

void print_summary(const TimingSummary& s) {
    std::cout << "\n╔══════════════════════════════════════════════════╗\n";
    std::cout << "║            TIMING SUMMARY REPORT                ║\n";
    std::cout << "╠══════════════════════════════════════════════════╣\n";

    std::cout << "║ Total cycles:          " << std::setw(12) << s.total_cycles
              << "             ║\n";

    // DRAM
    std::cout << "╠──────────────────────────────────────────────────╣\n";
    std::cout << "║ DRAM read requests:    " << std::setw(12) << s.dram_read_requests
              << "             ║\n";
    std::cout << "║ DRAM write requests:   " << std::setw(12) << s.dram_write_requests
              << "             ║\n";
    std::cout << "║ DRAM read bytes:       " << std::setw(12) << s.dram_read_bytes
              << "             ║\n";
    std::cout << "║ DRAM write bytes:      " << std::setw(12) << s.dram_write_bytes
              << "             ║\n";

    // Resource utilization
    std::cout << "╠──────────────────────────────────────────────────╣\n";
    std::cout << "║ MXU utilization:       " << std::setw(11) << std::fixed << std::setprecision(1)
              << (s.mxu_utilization * 100.0) << "%            ║\n";
    std::cout << "║ DMA utilization:       " << std::setw(11) << std::fixed << std::setprecision(1)
              << (s.dma_utilization * 100.0) << "%            ║\n";
    std::cout << "║ NoC utilization:       " << std::setw(11) << std::fixed << std::setprecision(1)
              << (s.noc_utilization * 100.0) << "%            ║\n";

    // Per-tile breakdown
    std::cout << "╠──────────────────────────────────────────────────╣\n";
    std::cout << "║ Tile │  busy  │  idle  │  stall                 ║\n";
    for (const auto& [tid, ts] : s.tile_stats) {
        std::cout << "║  T" << std::setw(2) << static_cast<int>(tid)
                  << "  │" << std::setw(6) << ts.busy_cycles
                  << "  │" << std::setw(6) << ts.idle_cycles
                  << "  │" << std::setw(6) << ts.stall_cycles
                  << "                  ║\n";
    }

    // Deadlock / causality
    std::cout << "╠──────────────────────────────────────────────────╣\n";
    std::cout << "║ Deadlock check:  "
              << (s.all_acquires_passed && s.all_dmas_completed ? "PASS ✓" : "FAIL ✗")
              << "                                ║\n";
    if (s.pending_acquires > 0)
        std::cout << "║   pending acquires: " << s.pending_acquires
                  << "                              ║\n";
    if (s.pending_dmas > 0)
        std::cout << "║   pending DMAs:     " << s.pending_dmas
                  << "                              ║\n";

    std::cout << "║ Causality check:  "
              << (s.causality_ok ? "PASS ✓" : "FAIL ✗")
              << "                                ║\n";
    if (s.causality_violations > 0)
        std::cout << "║   violations: " << s.causality_violations
                  << "                                   ║\n";

    std::cout << "╚══════════════════════════════════════════════════╝\n";
}

} // namespace mobol
