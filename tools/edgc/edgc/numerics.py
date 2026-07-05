"""Bit-exact numerics mirroring the C++ simulator (src/common/f16.h,
src/cycle/blockops.h).

The golden replayer runs the SAME arithmetic in the SAME order as the
tiles do, so a byte-for-byte match against the simulator's DRAM dump is a
test of the compiler's *dataflow*, not of any math library. Everything is
float32 internally; f16 uses IEEE-754 round-to-nearest-even.

Uses numpy when available (fast path) but every kernel is exercised through
scalar-equivalent operations so results do not depend on numpy's reductions.
"""
from __future__ import annotations
import struct
import math

# ── f16 <-> f32 (RNE), matching src/common/f16.h exactly ──────

def f32_to_f16_bits(f: float) -> int:
    b = struct.unpack("<I", struct.pack("<f", f))[0]
    sign = (b >> 31) & 1
    exp = (b >> 23) & 0xFF
    man = b & 0x7FFFFF
    if exp == 0xFF:
        if man:
            return (sign << 15) | 0x7E00
        return (sign << 15) | 0x7C00
    new_exp = exp - 127 + 15
    if new_exp >= 31:
        return (sign << 15) | 0x7C00
    if new_exp >= 1:
        rman = man >> 13
        round_bit = (man >> 12) & 1
        sticky = 1 if (man & 0xFFF) else 0
        if round_bit and (sticky or (rman & 1)):
            rman += 1
        if rman >= 0x400:
            rman = 0
            new_exp += 1
            if new_exp >= 31:
                return (sign << 15) | 0x7C00
        return (sign << 15) | (new_exp << 10) | rman
    shift = 1 - new_exp
    if shift >= 25:
        return sign << 15
    full = man | 0x800000
    total = 13 + (shift - 1)
    shifted = full >> total
    round_bit = (full >> (total - 1)) & 1
    sticky = 0
    if total >= 2:
        sticky = 1 if (full & ((1 << (total - 1)) - 1)) else 0
    rman = shifted
    if round_bit and (sticky or (rman & 1)):
        rman += 1
    if rman >= 0x400:
        return (sign << 15) | 0x0400
    return (sign << 15) | rman


def f16_bits_to_f32(h: int) -> float:
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    man = h & 0x3FF
    if exp == 0:
        if man == 0:
            u = sign << 31
        else:
            e = 0
            m = man
            while (m & 0x400) == 0:
                m <<= 1
                e += 1
            m &= 0x3FF
            u = (sign << 31) | ((127 - 15 + 1 - e) << 23) | (m << 13)
    elif exp == 0x1F:
        u = (sign << 31) | (0xFF << 23) | (man << 13)
    else:
        u = (sign << 31) | ((exp - 15 + 127) << 23) | (man << 13)
    return struct.unpack("<f", struct.pack("<I", u))[0]


def _round_f32(f: float) -> float:
    """Force a Python float through 32-bit rounding (tiles are f32)."""
    return struct.unpack("<f", struct.pack("<f", f))[0]


# The simulator's blockops.h evaluates transcendentals with the C library's
# single-precision std::exp/tanh/sqrt (float overloads). libm's expf and
# Python's math.exp round-to-f32 disagree on ~3% of inputs by 1 ULP, which
# amplifies across layers. To keep the golden/reference bit-identical to the
# simulator, call the SAME libm float functions via ctypes. (A tape-out
# datapath would use a defined LUT/polynomial, not libm — swapping both
# sides to that shared approximation is the eventual step; using the shared
# libm here already removes all cross-implementation divergence.)
import ctypes as _ct

def _load_libm():
    for name in ("libm.so.6", "libm.dylib", "libSystem.dylib", None):
        try:
            lib = _ct.CDLL(name)
            lib.expf.restype = _ct.c_float; lib.expf.argtypes = [_ct.c_float]
            lib.tanhf.restype = _ct.c_float; lib.tanhf.argtypes = [_ct.c_float]
            lib.sqrtf.restype = _ct.c_float; lib.sqrtf.argtypes = [_ct.c_float]
            _ = lib.expf(_ct.c_float(0.5))
            return lib
        except Exception:
            continue
    return None

_LIBM = _load_libm()

if _LIBM is not None:
    def expf(x):  return float(_LIBM.expf(_ct.c_float(x)))
    def tanhf(x): return float(_LIBM.tanhf(_ct.c_float(x)))
    def sqrtf(x): return float(_LIBM.sqrtf(_ct.c_float(x)))
else:  # pragma: no cover — fallback (may differ from sim by 1 ULP)
    def expf(x):  return _round_f32(math.exp(x))
    def tanhf(x): return _round_f32(math.tanh(x))
    def sqrtf(x): return _round_f32(math.sqrt(x))


# ── Vectorized helpers (numpy optional) ───────────────────────
try:
    import numpy as _np
    HAVE_NP = True
except Exception:  # pragma: no cover
    _np = None
    HAVE_NP = False


def to_f16_array(vals):
    """List[float] -> list of uint16 f16 bit patterns."""
    return [f32_to_f16_bits(_round_f32(float(v))) for v in vals]


def f16_bytes(bits_list) -> bytes:
    return b"".join(struct.pack("<H", b & 0xFFFF) for b in bits_list)


def f32_bytes(vals) -> bytes:
    return b"".join(struct.pack("<f", _round_f32(float(v))) for v in vals)


def bytes_to_f16(buf: bytes):
    return [struct.unpack("<H", buf[i:i + 2])[0] for i in range(0, len(buf), 2)]


def bytes_to_f32(buf: bytes):
    return [struct.unpack("<f", buf[i:i + 4])[0] for i in range(0, len(buf), 4)]


# ── f32 matmul with fixed k-order accumulation (matches MXU) ──

def matmul_f32(A, B, M, K, N, a_is_f16=True, b_is_f16=True, c_init=None):
    """Row-major A(MxK), B(KxN) -> C(MxN) f32. Inputs are lists of f16 bits
    or f32 floats. k ascending, f32 accumulate — bit-identical to blockops
    ::mxu. `c_init` (f32 list) seeds the accumulator, so tiling K into
    16-blocks with continued accumulation (blockops mxu acc=true seeds s
    with the old C) is associativity-identical to a single K-loop."""
    def av(i, k):
        v = A[i * K + k]
        return f16_bits_to_f32(v) if a_is_f16 else v
    def bv(k, j):
        v = B[k * N + j]
        return f16_bits_to_f32(v) if b_is_f16 else v
    C = [0.0] * (M * N)
    for i in range(M):
        for j in range(N):
            s = c_init[i * N + j] if c_init is not None else 0.0
            for k in range(K):
                s = _round_f32(s + _round_f32(av(i, k) * bv(k, j)))
            C[i * N + j] = s
    return C


def add_f32(a, b):
    return [_round_f32(x + y) for x, y in zip(a, b)]


def mul_f32(a, b):
    return [_round_f32(x * y) for x, y in zip(a, b)]


def silu_f32(a):
    # blockops: d = x / (1.0f + expf(-x))   (all float)
    return [_round_f32(x / _round_f32(1.0 + expf(_round_f32(-x)))) for x in a]


def gelu_f32(a):
    # blockops: 0.5f*x*(1.0f + tanhf(k*(x + 0.044715f*x*x*x)))
    # C left-assoc: 0.044715f*x*x*x == (((c*x)*x)*x)
    k = _round_f32(0.7978845608028654)
    c = _round_f32(0.044715)
    out = []
    for x in a:
        cterm = _round_f32(_round_f32(_round_f32(c * x) * x) * x)
        inner = _round_f32(k * _round_f32(x + cterm))
        t = tanhf(inner)
        out.append(_round_f32(_round_f32(0.5 * x) * _round_f32(1.0 + t)))
    return out


def rmsnorm(row, eps):
    # blockops rmsnorm_blk: inv = 1/sqrtf(ss/W + eps); d = v*inv
    eps = _round_f32(eps)  # C++ receives eps as float
    W = len(row)
    ss = 0.0
    for v in row:
        ss = _round_f32(ss + _round_f32(v * v))
    inv = _round_f32(1.0 / sqrtf(_round_f32(_round_f32(ss / W) + eps)))
    return [_round_f32(v * inv) for v in row]


def layernorm(row, eps):
    # blockops layernorm_f32: mean, var accumulated in f32, inv=1/sqrtf(var+eps)
    eps = _round_f32(eps)  # C++ receives eps as float
    W = len(row)
    mean = 0.0
    for v in row:
        mean = _round_f32(mean + v)
    mean = _round_f32(mean / W)
    var = 0.0
    for v in row:
        t = _round_f32(v - mean)
        var = _round_f32(var + _round_f32(t * t))
    var = _round_f32(var / W)
    inv = _round_f32(1.0 / sqrtf(_round_f32(var + eps)))
    return [_round_f32(_round_f32(v - mean) * inv) for v in row]


def softmax_row(scores, scale, valid_hi=None, valid_lo=0):
    # blockops softmax_blk: m over scaled, e = expf(scaled - m), sum, div
    n = len(scores)
    hi = n if valid_hi is None else valid_hi
    m = -math.inf
    for j in range(valid_lo, hi):
        m = max(m, _round_f32(scores[j] * scale))
    out = [0.0] * n
    ssum = 0.0
    for j in range(valid_lo, hi):
        e = expf(_round_f32(_round_f32(scores[j] * scale) - m))
        out[j] = e
        ssum = _round_f32(ssum + e)
    inv = _round_f32(1.0 / ssum) if ssum > 0 else 0.0
    return [_round_f32(v * inv) for v in out]
