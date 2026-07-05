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

// ─── Capacity constants ──────────────────────────────────────
constexpr Addr LOCAL_TILE_SIZE  = Addr{1} << 18;  ///< 256 KB per tile (18-bit offset)
constexpr Addr SHARED_BANK_SIZE = Addr{1} << 23;  ///< 8 MB per bank (23-bit offset)
constexpr Addr DRAM_POOL_SIZE   = Addr{1} << 34;  ///< 16 GB total (34-bit offset)

// ─── LOCAL tile stride (for base address calculation) ────────
constexpr Addr LOCAL_TILE_STRIDE = Addr{0x2} << 32; ///< 0x2_0000_0000 per tile
constexpr Addr SHARED_BANK_STRIDE = Addr{0x8} << 31; ///< 0x8_0000_0000 per bank

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
