/// @file memory_model.h
/// @brief PGAS address-space aware memory model for the Functional Engine.
///
/// Provides read/write/zero over the full 40-bit MOBOL address space:
///   LOCAL  (per-tile 256 KB scratchpad)
///   SHARED (per-bank 8 MB scratchpad)
///   DRAM   (16 GB main memory)
///
/// No timing, no latency — pure functional semantics.
#pragma once

#include "common/types.h"
#include "common/address.h"
#include <vector>
#include <cstring>
#include <stdexcept>

namespace mobol {

class MemoryModel {
public:
    MemoryModel();

    /// Write `size` bytes from `data` to the given PGAS address.
    void write(Addr addr, const void* data, size_t size);

    /// Read `size` bytes from the given PGAS address into `data`.
    void read(Addr addr, void* data, size_t size);

    /// Zero `size` bytes starting at the given PGAS address.
    void zero(Addr addr, size_t size);

    /// Direct pointer access (for MXU / reducer operations within a tile).
    /// Returns a mutable pointer to `size` contiguous bytes at `addr`.
    /// Only valid for LOCAL and SHARED segments.
    uint8_t*       local_ptr(Addr addr);
    const uint8_t* local_ptr(Addr addr) const;

    uint8_t*       shared_ptr(Addr addr);
    const uint8_t* shared_ptr(Addr addr) const;

    /// Access DRAM storage by offset (not by full PGAS address).
    uint8_t*       dram_ptr(Addr offset);
    const uint8_t* dram_ptr(Addr offset) const;

    /// Get the total bytes written to DRAM (for verification: C must not appear).
    size_t dram_write_count() const { return dram_write_bytes_; }
    size_t dram_read_count() const { return dram_read_bytes_; }

    /// Reset all memory to zero (for fresh runs).
    void reset();

private:
    // Storage
    std::vector<std::vector<uint8_t>> local_;   // [NUM_TILES][LOCAL_TILE_SIZE]
    std::vector<std::vector<uint8_t>> shared_;  // [NUM_BANKS][SHARED_BANK_SIZE]
    std::vector<uint8_t>              dram_;    // [DRAM_POOL_SIZE] — allocated lazily

    // DRAM stats
    size_t dram_write_bytes_ = 0;
    size_t dram_read_bytes_  = 0;

    // Internal: resolve address to raw pointer + validate
    uint8_t*       resolve(Addr addr, size_t size, bool is_write);
    const uint8_t* resolve(Addr addr, size_t size) const;
};

} // namespace mobol
