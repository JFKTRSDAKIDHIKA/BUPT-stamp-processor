/// @file trace_loader.h
/// @brief Load a compiler-emitted instruction trace + weight image and run
///        it on the cycle-accurate chip.
///
/// The Python AI compiler (tools/edgc/) emits three artifacts:
///   <name>.trace  — text ISA, one instruction per line, grouped by tile
///   <name>.mem    — binary DRAM image (host-endian bytes at given offsets)
///   <name>.meta   — JSON with entry addresses and a golden output hash
///
/// This loader parses the trace into a Program, preloads the DRAM image,
/// runs the simulator, and (optionally) dumps a named DRAM region so the
/// Python golden replayer can do a bit-exact comparison. Keeping the trace
/// as text keeps the compiler and simulator decoupled and auditable — the
/// same trace can be diffed, hand-inspected, or replayed on the reference.
#pragma once

#include "cycle/isa.h"
#include "cycle/cycle_config.h"
#include "common/memory_model.h"

#include <string>
#include <vector>
#include <cstdint>

namespace mobol::cycle {

struct TraceProgram {
    Program prog;
    CycleConfig cfg;                 ///< config lines in the trace override defaults
    // DRAM regions to preload: (offset, bytes) filled from the .mem image.
    // Output regions to dump after the run, keyed by name.
    struct Region { std::string name; uint64_t dram_off; uint64_t bytes; };
    std::vector<Region> dumps;
};

/// Parse a .trace file. `base` seeds TraceProgram.cfg (e.g. from the unified
/// architecture YAML); the trace's inline .config directives override
/// individual fields on top. Throws std::runtime_error with line context on
/// any malformed line (a tape-out toolchain must never silently miscompile).
TraceProgram load_trace(const std::string& trace_path,
                        const CycleConfig& base = CycleConfig{});

/// Load a .mem binary image (records: u64 offset, u64 len, bytes) into DRAM.
void load_mem_image(MemoryModel& mem, const std::string& mem_path);

/// Dump a DRAM region to a flat binary file (for golden comparison).
void dump_region(const MemoryModel& mem, uint64_t dram_off, uint64_t bytes,
                 const std::string& out_path);

} // namespace mobol::cycle
