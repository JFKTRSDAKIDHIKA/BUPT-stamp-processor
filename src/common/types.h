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
using TileId  = uint8_t;     ///< 0..NUM_TILES-1
using GroupId = uint8_t;     ///< 0..NUM_GROUPS-1
using BankId  = uint8_t;     ///< 0..NUM_BANKS-1

// Structural constants. Compile-time (they size fixed arrays and the PGAS
// address encoding) but injectable per build via -D so every DSE structural
// variant is an independent build directory:
//   cmake -DMOBOL_NUM_TILES=32 -DMOBOL_TILES_PER_GROUP=8 -DMOBOL_SHARED_MB=16
#ifndef MOBOL_NUM_TILES
#define MOBOL_NUM_TILES 16
#endif
#ifndef MOBOL_TILES_PER_GROUP
#define MOBOL_TILES_PER_GROUP 4
#endif

constexpr TileId NUM_TILES  = MOBOL_NUM_TILES;
constexpr uint8_t TILES_PER_GROUP = MOBOL_TILES_PER_GROUP;
constexpr GroupId NUM_GROUPS = NUM_TILES / TILES_PER_GROUP;
constexpr BankId NUM_BANKS  = NUM_TILES / TILES_PER_GROUP;

/// Compile-time ceil(log2(v)) for the address-field widths.
constexpr int clog2(uint64_t v) {
    int b = 0;
    while ((uint64_t{1} << b) < v) b++;
    return b;
}

static_assert(NUM_TILES >= 2 && (NUM_TILES & (NUM_TILES - 1)) == 0,
              "NUM_TILES must be a power of two");
static_assert(TILES_PER_GROUP >= 1
              && (TILES_PER_GROUP & (TILES_PER_GROUP - 1)) == 0,
              "TILES_PER_GROUP must be a power of two");
static_assert(NUM_TILES % TILES_PER_GROUP == 0 && NUM_BANKS >= 1,
              "NUM_TILES must be divisible by TILES_PER_GROUP");
constexpr int TILES_PER_GROUP_LOG2 = clog2(TILES_PER_GROUP);

// ─── Interconnect topology (base-die NoC) ────────────────────
// 0 = bidirectional ring (baseline), 1 = 2D mesh (XY routing),
// 2 = 2D torus (XY routing with wraparound). Compile-time: it changes the
// fabric's link structure.
#ifndef MOBOL_TOPOLOGY
#define MOBOL_TOPOLOGY 0
#endif
constexpr int TOPO_RING  = 0;
constexpr int TOPO_MESH  = 1;
constexpr int TOPO_TORUS = 2;
constexpr int TOPOLOGY = MOBOL_TOPOLOGY;
static_assert(TOPOLOGY >= TOPO_RING && TOPOLOGY <= TOPO_TORUS,
              "MOBOL_TOPOLOGY must be 0 (ring), 1 (mesh) or 2 (torus)");
constexpr const char* MOBOL_TOPOLOGY_NAME =
    TOPOLOGY == TOPO_RING ? "ring" : TOPOLOGY == TOPO_MESH ? "mesh" : "torus";

// Grid shape for mesh/torus: as square as possible with W >= H.
// 8 -> 4x2, 16 -> 4x4, 32 -> 8x4.
constexpr int NOC_GRID_W = Addr{1} << ((clog2(NUM_TILES) + 1) / 2);
constexpr int NOC_GRID_H = NUM_TILES / NOC_GRID_W;

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
