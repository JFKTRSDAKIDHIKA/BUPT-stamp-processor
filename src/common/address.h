/// @file address.h
/// @brief PGAS address space encoding/decoding per MOBOL spec §2.
#pragma once

#include "common/types.h"
#include <string>

namespace mobol {

/// Segment identifiers (one-hot encoding, bits [39:37])
enum class Segment : uint8_t {
    NULL_SEG = 0,  ///< 000 — unmapped, access traps
    LOCAL    = 1,  ///< 001 — base die, tile-private scratchpad
    SHARED   = 2,  ///< 010 — buffer die, shared scratchpad banks
    DRAM     = 4,  ///< 100 — top die, main memory
};

/// Decoded address fields
struct AddrFields {
    Segment seg;
    TileId  tile;     ///< Valid only for LOCAL
    BankId  bank;     ///< Valid only for SHARED
    Addr    offset;   ///< Byte offset within the segment unit
};

// ─── Segment base addresses ──────────────────────────────────
constexpr Addr LOCAL_BASE  = Addr{0x20} << 32;  ///< 0x20_0000_0000
constexpr Addr SHARED_BASE = Addr{0x40} << 32;  ///< 0x40_0000_0000
constexpr Addr DRAM_BASE   = Addr{0x80} << 32;  ///< 0x80_0000_0000

// ─── Parameterized 40-bit layout ─────────────────────────────
// Bits [39:37] hold the one-hot segment; the selector fields are anchored
// directly below at bit 36 so every structural variant (8..32 tiles,
// 2..16 banks, 4..16 MB banks) fits without moving the segment bits:
//   LOCAL : TILE[36 : 37-TILE_SEL_BITS] | hole | offset[17:0]
//   SHARED: BANK[36 : 37-BANK_SEL_BITS] | hole | offset[SHARED_OFF_BITS-1:0]
//   DRAM  :                       hole | offset[33:0]
// Widths never shrink below the historical 4/2 bits, so the baseline
// (16 tiles, 4 banks, 8 MB) layout is bit-identical to the original spec.
constexpr int TILE_SEL_BITS = clog2(NUM_TILES) > 4 ? clog2(NUM_TILES) : 4;
constexpr int BANK_SEL_BITS = clog2(NUM_BANKS) > 2 ? clog2(NUM_BANKS) : 2;
constexpr int TILE_SEL_LO   = 37 - TILE_SEL_BITS;
constexpr int BANK_SEL_LO   = 37 - BANK_SEL_BITS;

#ifndef MOBOL_SHARED_MB
#define MOBOL_SHARED_MB 8
#endif
constexpr int SHARED_MB = MOBOL_SHARED_MB;
static_assert(SHARED_MB >= 1 && (SHARED_MB & (SHARED_MB - 1)) == 0,
              "SHARED_MB must be a power of two");
constexpr int LOCAL_OFF_BITS  = 18;
constexpr int SHARED_OFF_BITS = 20 + clog2(SHARED_MB);

// ─── Capacity constants ──────────────────────────────────────
constexpr Addr LOCAL_TILE_SIZE  = Addr{1} << LOCAL_OFF_BITS;   ///< 256 KB per tile
constexpr Addr SHARED_BANK_SIZE = Addr{1} << SHARED_OFF_BITS;  ///< SHARED_MB per bank
constexpr Addr DRAM_POOL_SIZE   = Addr{1} << 34;  ///< 16 GB total (34-bit offset)

// The selector field must sit strictly above the offset field, or the
// 40-bit layout cannot host this (num_banks, shared_mb) combination.
static_assert(TILE_SEL_LO >= LOCAL_OFF_BITS,
              "LOCAL: tile selector overlaps the offset bits");
static_assert(BANK_SEL_LO >= SHARED_OFF_BITS,
              "SHARED: bank selector overlaps the offset bits "
              "(too many banks for this SHARED_MB)");

// ─── Selector strides (bit position of each field's LSB) ─────
constexpr Addr LOCAL_TILE_STRIDE  = Addr{1} << TILE_SEL_LO;  ///< baseline 0x2_0000_0000
constexpr Addr SHARED_BANK_STRIDE = Addr{1} << BANK_SEL_LO;  ///< baseline 0x8_0000_0000

// ─── Encoding functions ──────────────────────────────────────

/// Build a LOCAL segment address for a given tile and byte offset.
Addr make_local_addr(TileId tile, Addr offset);

/// Build a SHARED segment address for a given bank and byte offset.
Addr make_shared_addr(BankId bank, Addr offset);

/// Build a DRAM segment address for a given byte offset.
Addr make_dram_addr(Addr offset);

// ─── Decoding functions ──────────────────────────────────────

/// Decode a 40-bit global address into its segment fields.
AddrFields decode_addr(Addr addr);

/// Extract the segment from an address (top 3 bits of 40-bit address).
Segment get_segment(Addr addr);

/// Extract the tile selector from a LOCAL address.
TileId get_local_tile(Addr addr);

/// Extract the bank selector from a SHARED address.
BankId get_shared_bank(Addr addr);

/// Extract the byte offset from any segment address.
Addr get_offset(Addr addr);

// ─── Affinity functions ──────────────────────────────────────

/// Get the group number for a tile. group = tile_id / 4.
GroupId tile_group(TileId tile_id);

/// Check if a SHARED address is "near" for a given tile.
/// Near iff addr.bank == tile_group(tile_id).
bool is_shared_near(Addr addr, TileId tile_id);

// ─── Validation ──────────────────────────────────────────────

/// Check if an address is canonical (hole bits are zero).
/// Returns true if valid, false if trap-worthy.
bool is_canonical_addr(Addr addr);

// ─── Pretty printing ────────────────────────────────────────
std::string addr_to_string(Addr addr);

} // namespace mobol
