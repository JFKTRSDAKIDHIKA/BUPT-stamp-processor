/// @file f16.h
/// @brief IEEE 754 half-precision (binary16) storage type with f32 conversion.
///
/// Storage: uint16_t (IEEE 754 binary16 bit pattern).
/// All arithmetic is performed in f32; this type handles packing/unpacking only.
/// Conversion uses round-to-nearest-even (RN) per IEEE 754.
#pragma once

#include <cstdint>
#include <cstring>
#include <bit>

namespace mobol {

struct f16 {
    uint16_t bits = 0;

    f16() = default;
    explicit f16(uint16_t raw_bits) : bits(raw_bits) {}
    explicit f16(float f) : bits(f32_to_f16_bits(f)) {}

    /// Convert to f32 for arithmetic.
    float to_f32() const { return f16_to_f32_bits(bits); }
    explicit operator float() const { return to_f32(); }

    bool operator==(f16 o) const { return bits == o.bits; }
    bool operator!=(f16 o) const { return bits != o.bits; }

private:
    // ─── f16 → f32 ────────────────────────────────────────────
    static float f16_to_f32_bits(uint16_t h) {
        uint32_t sign = (h >> 15) & 1;
        uint32_t exp5 = (h >> 10) & 0x1F;
        uint32_t man10 = h & 0x3FF;

        uint32_t f32;
        if (exp5 == 0) {
            if (man10 == 0) {
                // ±Zero
                f32 = sign << 31;
            } else {
                // Subnormal: normalize
                uint32_t m = man10;
                uint32_t e = 0;
                while ((m & 0x400) == 0) {
                    m <<= 1;
                    e++;
                }
                m &= 0x3FF; // strip implicit bit
                f32 = (sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13);
            }
        } else if (exp5 == 0x1F) {
            // Inf or NaN
            f32 = (sign << 31) | (0xFF << 23) | (man10 << 13);
        } else {
            // Normal
            f32 = (sign << 31) | ((exp5 - 15 + 127) << 23) | (man10 << 13);
        }

        return std::bit_cast<float>(f32);
    }

    // ─── f32 → f16 ────────────────────────────────────────────
    static uint16_t f32_to_f16_bits(float f) {
        uint32_t f32 = std::bit_cast<uint32_t>(f);
        uint32_t sign  = (f32 >> 31) & 1;
        int32_t  exp32 = static_cast<int32_t>((f32 >> 23) & 0xFF);
        uint32_t man23 = f32 & 0x7FFFFF;

        // NaN / Inf passthrough
        if (exp32 == 0xFF) {
            if (man23 != 0)
                return static_cast<uint16_t>((sign << 15) | 0x7E00); // qNaN
            return static_cast<uint16_t>((sign << 15) | 0x7C00);     // Inf
        }

        int32_t unbiased = exp32 - 127;
        int32_t new_exp  = unbiased + 15;

        // ── Overflow → ±Inf ──
        if (new_exp >= 31) {
            return static_cast<uint16_t>((sign << 15) | 0x7C00);
        }

        // ── Normal f16 (1 ≤ new_exp ≤ 30) ──
        if (new_exp >= 1) {
            uint32_t result_man = man23 >> 13;           // truncate to 10 bits
            uint32_t round_bit  = (man23 >> 12) & 1;
            uint32_t sticky     = (man23 & 0xFFF) != 0 ? 1u : 0u;

            // Round-to-nearest-even
            if (round_bit && (sticky || (result_man & 1)))
                result_man++;

            // Handle mantissa overflow → bump exponent
            if (result_man >= 0x400) {
                result_man = 0;
                new_exp++;
                if (new_exp >= 31)
                    return static_cast<uint16_t>((sign << 15) | 0x7C00);
            }

            return static_cast<uint16_t>(
                (sign << 15) | (static_cast<uint32_t>(new_exp) << 10) | result_man);
        }

        // ── Subnormal or underflow (new_exp ≤ 0) ──
        int shift = 1 - new_exp; // extra right-shift beyond the normal 13

        // For shift >= 13 the 24-bit significand shifts out entirely (result
        // and round bit both 0), so the value rounds to ±0. Guarding here
        // also avoids C's shift-by->=32 UB in `full_man >> total_shift` below
        // (total_shift = 13 + shift - 1); the old guard (>= 25) let shifts of
        // 32..36 hit UB and return a spurious nonzero. This region is only
        // reachable via deep cancellation and never fired in the validated
        // workloads, but the fix makes f16.h correct-IEEE and consistent with
        // the RTL and the Python golden (both flush to 0 here).
        if (shift >= 13)
            return static_cast<uint16_t>(sign << 15); // too small → ±0

        // Include implicit leading 1
        uint32_t full_man = man23 | 0x800000;
        uint32_t total_shift = 13 + static_cast<uint32_t>(shift - 1);
        uint32_t shifted = full_man >> total_shift;

        // Rounding bits (relative to a 10-bit result)
        uint32_t round_bit = (full_man >> (total_shift - 1)) & 1;
        uint32_t sticky    = 0;
        if (total_shift >= 2) {
            uint32_t mask = (1u << (total_shift - 1)) - 1;
            sticky = (full_man & mask) != 0 ? 1u : 0u;
        }

        uint32_t result_man = shifted;
        if (round_bit && (sticky || (result_man & 1)))
            result_man++;

        // Rounding may push subnormal → smallest normal
        if (result_man >= 0x400)
            return static_cast<uint16_t>((sign << 15) | 0x0400); // exp=1, man=0

        return static_cast<uint16_t>((sign << 15) | result_man);
    }
};

} // namespace mobol
