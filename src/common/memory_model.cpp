/// @file memory_model.cpp

#include "common/memory_model.h"
#include "common/address.h"
#include <algorithm>
#include <cassert>
#include <sstream>

namespace {

/// Always-on trap (spec §3: hole bits / out-of-range => hardware trap).
/// Unlike assert(), these fire in release builds too — a tape-out-grade
/// simulator must never silently corrupt memory.
[[noreturn]] void trap(const char* what, mobol::Addr addr, size_t size) {
    std::ostringstream os;
    os << "MEM TRAP: " << what << " addr=" << mobol::addr_to_string(addr)
       << " (0x" << std::hex << addr << std::dec << ") size=" << size;
    throw std::runtime_error(os.str());
}

} // namespace

namespace mobol {

MemoryModel::MemoryModel() {
    reset();
}

void MemoryModel::reset() {
    local_.assign(NUM_TILES, std::vector<uint8_t>(LOCAL_TILE_SIZE, 0));
    shared_.assign(NUM_BANKS, std::vector<uint8_t>(SHARED_BANK_SIZE, 0));
    // DRAM: allocate 16 MB for milestone 1 (actual matrices are tiny)
    // Full 16 GB would be wasteful; we only need enough for the workload.
    dram_.assign(16 * 1024 * 1024, 0); // 16 MB
    dram_write_bytes_ = 0;
    dram_read_bytes_ = 0;
}

void MemoryModel::write(Addr addr, const void* data, size_t size) {
    uint8_t* dst = resolve(addr, size, /*is_write=*/true);
    std::memcpy(dst, data, size);

    if (get_segment(addr) == Segment::DRAM)
        dram_write_bytes_ += size;
}

void MemoryModel::read(Addr addr, void* data, size_t size) {
    const uint8_t* src = resolve(addr, size);
    std::memcpy(data, src, size);

    if (get_segment(addr) == Segment::DRAM)
        dram_read_bytes_ += size;
}

void MemoryModel::zero(Addr addr, size_t size) {
    uint8_t* dst = resolve(addr, size, /*is_write=*/true);
    std::memset(dst, 0, size);
}

uint8_t* MemoryModel::local_ptr(Addr addr) {
    assert(get_segment(addr) == Segment::LOCAL);
    TileId t = get_local_tile(addr);
    Addr off = get_offset(addr);
    return local_[t].data() + off;
}

const uint8_t* MemoryModel::local_ptr(Addr addr) const {
    assert(get_segment(addr) == Segment::LOCAL);
    TileId t = get_local_tile(addr);
    Addr off = get_offset(addr);
    return local_[t].data() + off;
}

uint8_t* MemoryModel::shared_ptr(Addr addr) {
    assert(get_segment(addr) == Segment::SHARED);
    BankId b = get_shared_bank(addr);
    Addr off = get_offset(addr);
    return shared_[b].data() + off;
}

const uint8_t* MemoryModel::shared_ptr(Addr addr) const {
    assert(get_segment(addr) == Segment::SHARED);
    BankId b = get_shared_bank(addr);
    Addr off = get_offset(addr);
    return shared_[b].data() + off;
}

uint8_t* MemoryModel::dram_ptr(Addr offset) {
    assert(offset < dram_.size());
    return dram_.data() + offset;
}

const uint8_t* MemoryModel::dram_ptr(Addr offset) const {
    assert(offset < dram_.size());
    return dram_.data() + offset;
}

// ─── Internal ────────────────────────────────────────────────

uint8_t* MemoryModel::resolve(Addr addr, size_t size, bool is_write) {
    (void)is_write;
    if (!is_canonical_addr(addr)) trap("non-canonical address (hole bits)", addr, size);
    Segment seg = get_segment(addr);
    Addr off = get_offset(addr);

    switch (seg) {
        case Segment::LOCAL: {
            TileId t = get_local_tile(addr);
            if (off + size > LOCAL_TILE_SIZE) trap("LOCAL out of range", addr, size);
            return local_[t].data() + off;
        }
        case Segment::SHARED: {
            BankId b = get_shared_bank(addr);
            if (off + size > SHARED_BANK_SIZE) trap("SHARED out of range", addr, size);
            return shared_[b].data() + off;
        }
        case Segment::DRAM: {
            if (off + size > dram_.size()) trap("DRAM out of backing range", addr, size);
            return dram_.data() + off;
        }
        default:
            trap("NULL/invalid segment", addr, size);
    }
}

const uint8_t* MemoryModel::resolve(Addr addr, size_t size) const {
    if (!is_canonical_addr(addr)) trap("non-canonical address (hole bits)", addr, size);
    Segment seg = get_segment(addr);
    Addr off = get_offset(addr);

    switch (seg) {
        case Segment::LOCAL: {
            TileId t = get_local_tile(addr);
            if (off + size > LOCAL_TILE_SIZE) trap("LOCAL out of range", addr, size);
            return local_[t].data() + off;
        }
        case Segment::SHARED: {
            BankId b = get_shared_bank(addr);
            if (off + size > SHARED_BANK_SIZE) trap("SHARED out of range", addr, size);
            return shared_[b].data() + off;
        }
        case Segment::DRAM: {
            if (off + size > dram_.size()) trap("DRAM out of backing range", addr, size);
            return dram_.data() + off;
        }
        default:
            trap("NULL/invalid segment", addr, size);
    }
}

} // namespace mobol
