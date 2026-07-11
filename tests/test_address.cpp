/// @file test_address.cpp
/// @brief Unit tests for the parameterized PGAS address encoding.
///
/// Compiled once per structural variant (-DMOBOL_NUM_TILES / _TILES_PER_GROUP
/// / _SHARED_MB), so every DSE build validates decode_addr / make_*_addr /
/// is_canonical_addr / is_shared_near at ITS OWN bit widths. This is the
/// gate against silently-wrong address encodings producing plausible cycle
/// counts.

#include "common/address.h"

#include <cstdio>
#include <cstdlib>

using namespace mobol;

static int failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);    \
            failures++;                                                    \
        }                                                                  \
    } while (0)

int main() {
    std::printf("address test: NUM_TILES=%d NUM_BANKS=%d TILES_PER_GROUP=%d "
                "SHARED_MB=%d  tile@[36:%d] bank@[36:%d]\n",
                int(NUM_TILES), int(NUM_BANKS), int(TILES_PER_GROUP),
                SHARED_MB, TILE_SEL_LO, BANK_SEL_LO);

    // ── LOCAL: full round-trip over every tile at boundary offsets ──
    for (int t = 0; t < NUM_TILES; t++) {
        for (Addr off : {Addr{0}, Addr{63}, Addr{4096},
                         LOCAL_TILE_SIZE - 1}) {
            Addr a = make_local_addr(static_cast<TileId>(t), off);
            CHECK(a <= ADDR_MASK);
            AddrFields f = decode_addr(a);
            CHECK(f.seg == Segment::LOCAL);
            CHECK(f.tile == t);
            CHECK(f.offset == off);
            CHECK(is_canonical_addr(a));
        }
    }

    // ── SHARED: full round-trip over every bank at boundary offsets ──
    for (int b = 0; b < NUM_BANKS; b++) {
        for (Addr off : {Addr{0}, Addr{64}, SHARED_BANK_SIZE / 2,
                         SHARED_BANK_SIZE - 1}) {
            Addr a = make_shared_addr(static_cast<BankId>(b), off);
            CHECK(a <= ADDR_MASK);
            AddrFields f = decode_addr(a);
            CHECK(f.seg == Segment::SHARED);
            CHECK(f.bank == b);
            CHECK(f.offset == off);
            CHECK(is_canonical_addr(a));
        }
    }

    // ── DRAM round-trip ──
    for (Addr off : {Addr{0}, Addr{0x1234560}, DRAM_POOL_SIZE - 1}) {
        Addr a = make_dram_addr(off);
        AddrFields f = decode_addr(a);
        CHECK(f.seg == Segment::DRAM);
        CHECK(f.offset == off);
        CHECK(is_canonical_addr(a));
    }

    // ── Field disjointness: distinct (tile,offset) never collide ──
    CHECK(make_local_addr(0, LOCAL_TILE_SIZE - 1)
          < make_local_addr(1, 0));
    if (NUM_BANKS > 1)
        CHECK(make_shared_addr(0, SHARED_BANK_SIZE - 1)
              < make_shared_addr(1, 0));

    // ── Canonicality: hole bits reject ──
    if (TILE_SEL_LO > LOCAL_OFF_BITS) {
        Addr bad = make_local_addr(0, 0) | (Addr{1} << LOCAL_OFF_BITS);
        CHECK(!is_canonical_addr(bad));
    }
    if (BANK_SEL_LO > SHARED_OFF_BITS) {
        Addr bad = make_shared_addr(0, 0) | (Addr{1} << SHARED_OFF_BITS);
        CHECK(!is_canonical_addr(bad));
    }
    // Selector beyond the last tile/bank rejects (field may be wider than
    // clog2(count) for small counts).
    if ((Addr{NUM_TILES} << TILE_SEL_LO) < (Addr{1} << 37)) {
        Addr bad = LOCAL_BASE | (Addr{NUM_TILES} << TILE_SEL_LO);
        CHECK(!is_canonical_addr(bad));
    }
    if ((Addr{NUM_BANKS} << BANK_SEL_LO) < (Addr{1} << 37)) {
        Addr bad = SHARED_BASE | (Addr{NUM_BANKS} << BANK_SEL_LO);
        CHECK(!is_canonical_addr(bad));
    }
    CHECK(!is_canonical_addr(0));  // NULL segment traps

    // ── Affinity: is_shared_near matches tile_group for every pair ──
    for (int t = 0; t < NUM_TILES; t++) {
        CHECK(tile_group(static_cast<TileId>(t)) == t / TILES_PER_GROUP);
        for (int b = 0; b < NUM_BANKS; b++) {
            Addr a = make_shared_addr(static_cast<BankId>(b), 128);
            bool near = is_shared_near(a, static_cast<TileId>(t));
            CHECK(near == (b == t / TILES_PER_GROUP));
        }
    }

    // ── Baseline bit-compatibility anchor (only meaningful at 16/4/8) ──
    if (NUM_TILES == 16 && NUM_BANKS == 4 && SHARED_MB == 8) {
        CHECK(make_local_addr(3, 0x200) == 0x2600000200ull);
        CHECK(make_shared_addr(2, 0x40) == 0x5000000040ull);
        CHECK(make_dram_addr(0x1000) == 0x8000001000ull);
    }

    if (failures == 0) {
        std::printf("address test: ALL PASS\n");
        return 0;
    }
    std::printf("address test: %d FAILURES\n", failures);
    return 1;
}
