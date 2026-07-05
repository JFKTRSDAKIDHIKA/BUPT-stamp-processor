/// @file address.cpp
/// @brief PGAS address space encoding/decoding implementation.

#include "common/address.h"
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace mobol {

// ─── Bit field extraction helpers ────────────────────────────

/// Extract bits [hi:lo] from a 40-bit address (inclusive).
static constexpr uint64_t extract_bits(Addr addr, int hi, int lo) {
    return (addr >> lo) & ((uint64_t{1} << (hi - lo + 1)) - 1);
}

// ─── Encoding ────────────────────────────────────────────────

Addr make_local_addr(TileId tile, Addr offset) {
    assert(tile < NUM_TILES);
    assert(offset < LOCAL_TILE_SIZE);
    return LOCAL_BASE | (Addr{tile} << 33) | offset;
}

Addr make_shared_addr(BankId bank, Addr offset) {
    assert(bank < NUM_BANKS);
    assert(offset < SHARED_BANK_SIZE);
    return SHARED_BASE | (Addr{bank} << 35) | offset;
}

Addr make_dram_addr(Addr offset) {
    assert(offset < DRAM_POOL_SIZE);
    return DRAM_BASE | offset;
}

// ─── Decoding ────────────────────────────────────────────────

Segment get_segment(Addr addr) {
    // Bits [39:37], one-hot
    uint8_t seg_bits = static_cast<uint8_t>(extract_bits(addr, 39, 37));
    switch (seg_bits) {
        case 0b001: return Segment::LOCAL;
        case 0b010: return Segment::SHARED;
        case 0b100: return Segment::DRAM;
        case 0b000: return Segment::NULL_SEG;
        default:
            throw std::runtime_error("Invalid segment bits: " + std::to_string(seg_bits));
    }
}

TileId get_local_tile(Addr addr) {
    return static_cast<TileId>(extract_bits(addr, 36, 33));
}

BankId get_shared_bank(Addr addr) {
    return static_cast<BankId>(extract_bits(addr, 36, 35));
}

Addr get_offset(Addr addr) {
    Segment seg = get_segment(addr);
    switch (seg) {
        case Segment::LOCAL:
            return addr & (LOCAL_TILE_SIZE - 1);  // low 18 bits
        case Segment::SHARED:
            return addr & (SHARED_BANK_SIZE - 1); // low 23 bits
        case Segment::DRAM:
            return addr & (DRAM_POOL_SIZE - 1);   // low 34 bits
        default:
            return 0;
    }
}

AddrFields decode_addr(Addr addr) {
    AddrFields f;
    f.seg = get_segment(addr);
    f.tile = 0;
    f.bank = 0;
    f.offset = 0;

    switch (f.seg) {
        case Segment::LOCAL:
            f.tile = get_local_tile(addr);
            f.offset = get_offset(addr);
            break;
        case Segment::SHARED:
            f.bank = get_shared_bank(addr);
            f.offset = get_offset(addr);
            break;
        case Segment::DRAM:
            f.offset = get_offset(addr);
            break;
        case Segment::NULL_SEG:
            break;
    }
    return f;
}

// ─── Affinity ────────────────────────────────────────────────

GroupId tile_group(TileId tile_id) {
    assert(tile_id < NUM_TILES);
    return static_cast<GroupId>(tile_id >> 2);
}

bool is_shared_near(Addr addr, TileId tile_id) {
    return get_shared_bank(addr) == tile_group(tile_id);
}

// ─── Validation ──────────────────────────────────────────────

bool is_canonical_addr(Addr addr) {
    Segment seg = get_segment(addr);
    switch (seg) {
        case Segment::LOCAL: {
            // Hole bits [32:18] must be zero
            uint64_t hole = extract_bits(addr, 32, 18);
            return hole == 0 && get_offset(addr) < LOCAL_TILE_SIZE;
        }
        case Segment::SHARED: {
            // Hole bits [34:23] must be zero
            uint64_t hole = extract_bits(addr, 34, 23);
            return hole == 0 && get_offset(addr) < SHARED_BANK_SIZE;
        }
        case Segment::DRAM: {
            // Hole bits [36:34] must be zero
            uint64_t hole = extract_bits(addr, 36, 34);
            return hole == 0 && get_offset(addr) < DRAM_POOL_SIZE;
        }
        case Segment::NULL_SEG:
            return false;  // NULL access always traps
    }
    return false;
}

// ─── Pretty printing ────────────────────────────────────────

std::string addr_to_string(Addr addr) {
    std::ostringstream oss;
    oss << "0x" << std::hex << std::setfill('0') << std::setw(10) << addr;

    Segment seg = get_segment(addr);
    switch (seg) {
        case Segment::LOCAL:
            oss << " [LOCAL tile=" << static_cast<int>(get_local_tile(addr))
                << " off=0x" << std::hex << get_offset(addr) << "]";
            break;
        case Segment::SHARED:
            oss << " [SHARED bank=" << static_cast<int>(get_shared_bank(addr))
                << " off=0x" << std::hex << get_offset(addr) << "]";
            break;
        case Segment::DRAM:
            oss << " [DRAM off=0x" << std::hex << get_offset(addr) << "]";
            break;
        case Segment::NULL_SEG:
            oss << " [NULL]";
            break;
    }
    return oss.str();
}

} // namespace mobol
