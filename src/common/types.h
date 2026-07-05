/// @file types.h
/// @brief Core type definitions for the MOBOL cycle-level simulator.
#pragma once

#include <cstdint>
#include <cstddef>
#include <array>
#include <vector>
#include <string>
#include <cassert>

namespace mobol {

// ─── Address types ────────────────────────────────────────────
using Addr = uint64_t;       ///< 40-bit physical address (stored in 64-bit)
constexpr Addr ADDR_WIDTH = 40;
constexpr Addr ADDR_MASK  = (Addr{1} << ADDR_WIDTH) - 1;

// ─── Tile / group identifiers ────────────────────────────────
using TileId  = uint8_t;     ///< 0..15
using GroupId = uint8_t;     ///< 0..3
using BankId  = uint8_t;     ///< 0..3

constexpr TileId NUM_TILES  = 16;
constexpr GroupId NUM_GROUPS = 4;
constexpr BankId NUM_BANKS  = 4;
constexpr uint8_t TILES_PER_GROUP = 4;

// ─── MXU parameters ──────────────────────────────────────────
constexpr int MXU_M = 16;    ///< MXU native M dimension
constexpr int MXU_N = 16;    ///< MXU native N dimension
constexpr int MXU_K = 16;    ///< MXU native K dimension

// ─── Timing parameters (configurable, approved defaults) ─────
constexpr int MXU_LATENCY    = 4;   ///< Cycles from launch to result
constexpr int MXU_THROUGHPUT = 1;   ///< Cycles between consecutive launches
constexpr int DMA_SETUP_CYCLES = 2; ///< DMA descriptor setup latency
constexpr int NOC_HOP_LATENCY  = 1; ///< Cycles per NoC hop
constexpr int NOC_LINK_BW      = 64;///< Bytes per cycle per NoC link

// ─── Data types ──────────────────────────────────────────────
enum class DType : uint8_t {
    F16 = 0,
    F32 = 1,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::F16: return 2;
        case DType::F32: return 4;
    }
    return 0;
}

// ─── Cycle type ──────────────────────────────────────────────
using Cycle = int64_t;

} // namespace mobol
