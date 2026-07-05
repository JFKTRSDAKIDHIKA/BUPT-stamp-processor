"""Golden ISA interpreter: executes a compiler .trace against a functional
memory model, using bit-exact numerics (edgc.numerics).

This is the reference semantics of the tile ISA. Two independent checks
pin correctness:
  (1) golden DRAM dump == simulator DRAM dump   -> the C++ simulator
      executes the ISA exactly as specified (validates the simulator).
  (2) golden DRAM dump == high-level model forward (f16 tolerance) ->
      the compiler lowered the model correctly (validates the compiler).

The interpreter is untimed: it runs each tile's program to completion in
dependency order driven by the same release/acquire tokens, so it exercises
the exact reductions the schedule encodes. Because the schedule is a static
DAG with no cycles, a simple fixpoint over blocked tiles converges.
"""
from __future__ import annotations
import struct
import math
from typing import Dict, List
from . import numerics as N
from .isa import (NUM_TILES, NUM_BANKS, LOCAL_BASE, LOCAL_STRIDE, SHARED_BASE,
                  SHARED_STRIDE, DRAM_BASE, LOCAL_SIZE, SHARED_SIZE, tile_group,
                  NMC_REDUCE_F32, NMC_LN_F16, NMC_PREFETCH)

BN = 16


class Mem:
    """Byte-addressable LOCAL/SHARED/DRAM, matching the simulator layout."""
    def __init__(self, dram_bytes=64 << 20):
        self.local = [bytearray(LOCAL_SIZE) for _ in range(NUM_TILES)]
        self.shared = [bytearray(SHARED_SIZE) for _ in range(NUM_BANKS)]
        self.dram = bytearray(dram_bytes)

    def _resolve(self, addr):
        seg = (addr >> 37) & 0x7
        if seg == 0b001:
            tile = (addr >> 33) & 0xF
            off = addr & ((1 << 18) - 1)
            return self.local[tile], off
        if seg == 0b010:
            bank = (addr >> 35) & 0x3
            off = addr & ((1 << 23) - 1)
            return self.shared[bank], off
        if seg == 0b100:
            off = addr & ((1 << 34) - 1)
            return self.dram, off
        raise ValueError(f"bad segment in addr 0x{addr:x}")

    def read(self, addr, n):
        buf, off = self._resolve(addr)
        return bytes(buf[off:off + n])

    def write(self, addr, data):
        buf, off = self._resolve(addr)
        buf[off:off + len(data)] = data


class Golden:
    def __init__(self, trace_path, mem_path):
        self.mem = Mem()
        self.dumps = []
        self._load_mem(mem_path)
        self.progs = self._parse(trace_path)
        self.pc = [0] * NUM_TILES
        self.cnt: Dict[tuple, int] = {}   # (tile, tag) -> count

    # ── loading ──
    def _load_mem(self, path):
        with open(path, "rb") as f:
            blob = f.read()
        i = 0
        while i < len(blob):
            off, n = struct.unpack_from("<QQ", blob, i)
            i += 16
            self.mem.write(DRAM_BASE + off, blob[i:i + n])
            i += n

    def _parse(self, path):
        progs = [[] for _ in range(NUM_TILES)]
        cur = -1
        with open(path) as f:
            for raw in f:
                line = raw.split("#", 1)[0].strip()
                if not line:
                    continue
                toks = line.split()
                head = toks[0]
                if head == ".config":
                    continue
                if head == ".dump":
                    kv = _kv(toks[1:])
                    self.dumps.append((kv["name"], int(kv["off"], 0), int(kv["bytes"])))
                    continue
                if head == ".tile":
                    cur = int(toks[1])
                    continue
                kv = _kv(toks[1:])
                progs[cur].append((head, kv))
        return progs

    # ── execution: fixpoint over blocked tiles ──
    def run(self):
        done = [False] * NUM_TILES
        guard = 0
        limit = sum(len(p) for p in self.progs) * 4 + 1000
        while not all(done):
            progressed = False
            for t in range(NUM_TILES):
                if done[t]:
                    continue
                while self.pc[t] < len(self.progs[t]):
                    op, kv = self.progs[t][self.pc[t]]
                    if op == "ACQUIRE":
                        tag = _i(kv, "tag")
                        ar = _i(kv, "arity")
                        if self.cnt.get((t, tag), 0) < ar:
                            break  # blocked; try again next sweep
                        self.cnt[(t, tag)] -= ar
                        self.pc[t] += 1
                        progressed = True
                        continue
                    self._exec(t, op, kv)
                    self.pc[t] += 1
                    progressed = True
                    if op == "HALT":
                        done[t] = True
                        break
                else:
                    done[t] = True
            guard += 1
            if not progressed and not all(done):
                blocked = [t for t in range(NUM_TILES) if not done[t]]
                raise RuntimeError(f"golden deadlock; blocked tiles {blocked} "
                                   f"at pc {[self.pc[t] for t in blocked]}")
            if guard > limit:
                raise RuntimeError("golden run exceeded step limit")

    def dump(self, name):
        for nm, off, nb in self.dumps:
            if nm == name:
                return self.mem.read(DRAM_BASE + off, nb)
        raise KeyError(name)

    # ── per-instruction semantics ──
    def _exec(self, t, op, kv):
        m = self.mem
        if op in ("NOP", "HALT", "WAIT_MXU", "DMA_FENCE"):
            return
        if op == "DMA":
            self._dma(t, kv)
            return
        if op in ("MXU_F16F16", "MXU_F32F16"):
            self._mxu(t, op, kv)
            return
        if op == "LOAD_SHARED":
            self._dma(t, kv)  # same 2D copy semantics
            return
        if op == "RELEASE":
            cons = _i(kv, "consumer")
            tag = _i(kv, "tag")
            self.cnt[(cons, tag)] = self.cnt.get((cons, tag), 0) + 1
            return
        if op == "NMC":
            self._nmc(t, kv)
            return
        if op.startswith("VPU_"):
            self._vpu(t, op, kv)
            return
        raise ValueError(f"golden: unhandled op {op}")

    def _dma(self, t, kv):
        src = _i(kv, "src")
        dst = _i(kv, "dst")
        rows = _i(kv, "rows", 1)
        rb = _i(kv, "row_bytes")
        ss = _i(kv, "src_stride", 0)
        ds = _i(kv, "dst_stride", 0)
        for r in range(rows):
            data = self.mem.read(src + r * ss, rb)
            self.mem.write(dst + r * ds, data)

    def _mxu(self, t, op, kv):
        a = _abs_local(t, _i(kv, "a"))
        b = _abs_local(t, _i(kv, "b"))
        d = _abs_local(t, _i(kv, "d"))
        acc = _i(kv, "acc", 0)
        a_f16 = (op == "MXU_F16F16")
        A = self.mem.read(a, 512 if a_f16 else 1024)
        B = self.mem.read(b, 512)
        Ain = N.bytes_to_f16(A) if a_f16 else N.bytes_to_f32(A)
        Bin = N.bytes_to_f16(B)
        # Continued accumulation: seed with old C (matches blockops mxu,
        # where acc=true starts s = C[i][j] then adds the 16 products).
        old = N.bytes_to_f32(self.mem.read(d, 1024)) if acc else None
        C = N.matmul_f32(Ain, Bin, 16, 16, 16, a_is_f16=a_f16, b_is_f16=True,
                         c_init=old)
        self.mem.write(d, N.f32_bytes(C))

    def _vpu(self, t, op, kv):
        a = _abs_local(t, _i(kv, "a"))
        b = _abs_local(t, _i(kv, "b"))
        d = _abs_local(t, _i(kv, "d"))
        cnt = _i(kv, "count", 1)
        scalar = _f(kv, "scalar", 0.0)
        m = self.mem
        if op == "VPU_ADD_F32":
            x = N.bytes_to_f32(m.read(a, 1024)); y = N.bytes_to_f32(m.read(b, 1024))
            m.write(d, N.f32_bytes(N.add_f32(x, y)))
        elif op == "VPU_ADD_F16":
            x = [N.f16_bits_to_f32(v) for v in N.bytes_to_f16(m.read(a, 512))]
            y = [N.f16_bits_to_f32(v) for v in N.bytes_to_f16(m.read(b, 512))]
            m.write(d, N.f16_bytes(N.to_f16_array(N.add_f32(x, y))))
        elif op == "VPU_MUL_F32":
            x = N.bytes_to_f32(m.read(a, 1024)); y = N.bytes_to_f32(m.read(b, 1024))
            m.write(d, N.f32_bytes(N.mul_f32(x, y)))
        elif op == "VPU_SILU_F32":
            x = N.bytes_to_f32(m.read(a, 1024))
            m.write(d, N.f32_bytes(N.silu_f32(x)))
        elif op == "VPU_GELU_F32":
            x = N.bytes_to_f32(m.read(a, 1024))
            m.write(d, N.f32_bytes(N.gelu_f32(x)))
        elif op == "VPU_CVT_F32_F16":
            x = N.bytes_to_f32(m.read(a, 1024))
            m.write(d, N.f16_bytes(N.to_f16_array(x)))
        elif op == "VPU_CVT_F16_F32":
            x = [N.f16_bits_to_f32(v) for v in N.bytes_to_f16(m.read(a, 512))]
            m.write(d, N.f32_bytes(x))
        elif op == "VPU_TRANS_F16":
            x = N.bytes_to_f16(m.read(a, 512))
            out = [0] * 256
            for i in range(16):
                for j in range(16):
                    out[j * 16 + i] = x[i * 16 + j]
            m.write(d, N.f16_bytes(out))
        elif op == "VPU_SCALE_F32":
            x = N.bytes_to_f32(m.read(a, 1024))
            m.write(d, N.f32_bytes([N._round_f32(v * scalar) for v in x]))
        elif op in ("VPU_LAYERNORM_F32", "VPU_RMSNORM_BLK"):
            blocked = (op == "VPU_RMSNORM_BLK")
            x = N.bytes_to_f32(m.read(a, 1024 * cnt))
            W = 16 * cnt
            out = [0.0] * (16 * W)
            for i in range(16):
                row = [_blk_get(x, i, j) if blocked else x[i * W + j] for j in range(W)]
                nr = N.rmsnorm(row, scalar) if blocked else N.layernorm(row, scalar)
                for j in range(W):
                    if blocked:
                        _blk_set(out, i, j, nr[j])
                    else:
                        out[i * W + j] = nr[j]
            m.write(d, N.f32_bytes(out))
        elif op in ("VPU_SOFTMAX_F32", "VPU_SOFTMAX_BLK"):
            blocked = (op == "VPU_SOFTMAX_BLK")
            x = N.bytes_to_f32(m.read(a, 1024 * cnt))
            W = 16 * cnt
            valid = _i(kv, "row_bytes", 0)
            row0 = _i(kv, "aux", -1)
            window = _i(kv, "rows", 0)
            out = [0.0] * (16 * W)
            for i in range(16):
                row = [_blk_get(x, i, j) if blocked else x[i * W + j] for j in range(W)]
                cols = valid if valid > 0 else W
                hi = cols
                lo = 0
                if row0 >= 0:
                    hi = min(cols, row0 + i + 1)
                    if window > 0:
                        lo = max(0, row0 + i - window + 1)
                sm = N.softmax_row(row, scalar, hi, lo)
                for j in range(W):
                    if blocked:
                        _blk_set(out, i, j, sm[j])
                    else:
                        out[i * W + j] = sm[j]
            m.write(d, N.f32_bytes(out))
        elif op == "VPU_ROPE_F16":
            aux = _i(kv, "aux")
            half = cnt
            x = N.bytes_to_f16(m.read(a, 512))
            cosb = N.bytes_to_f32(m.read(_abs_local(t, aux), 1024))
            sinb = N.bytes_to_f32(m.read(_abs_local(t, aux + 1024), 1024))
            o = [0] * 256
            for i in range(BN):
                for k in range(half):
                    av = N.f16_bits_to_f32(x[i * BN + k])
                    bv = N.f16_bits_to_f32(x[i * BN + half + k])
                    c = cosb[i * BN + k]; s = sinb[i * BN + k]
                    o[i * BN + k] = N.f32_to_f16_bits(N._round_f32(av * c - bv * s))
                    o[i * BN + half + k] = N.f32_to_f16_bits(N._round_f32(bv * c + av * s))
            m.write(d, N.f16_bytes(o))
        else:
            raise ValueError(f"golden vpu: {op}")

    def _nmc(self, t, kv):
        nmc_op = _i(kv, "arity")
        m = self.mem
        if nmc_op == NMC_PREFETCH:
            src = _i(kv, "src"); dst = _i(kv, "dst")
            rows = _i(kv, "rows"); rb = _i(kv, "row_bytes")
            ss = _i(kv, "src_stride"); ds = _i(kv, "dst_stride")
            for r in range(rows):
                m.write(dst + r * ds, m.read(src + r * ss, rb))
        elif nmc_op == NMC_REDUCE_F32:
            src = _i(kv, "src"); dst = _i(kv, "dst"); cnt = _i(kv, "count")
            acc = N.bytes_to_f32(m.read(src, 1024))
            for j in range(1, cnt):
                acc = N.add_f32(acc, N.bytes_to_f32(m.read(src + j * 1024, 1024)))
            m.write(dst, N.f32_bytes(acc))
        elif nmc_op == NMC_LN_F16:
            src = _i(kv, "src"); dst = _i(kv, "dst"); cnt = _i(kv, "count")
            scalar = _f(kv, "scalar", 0.0)
            x = N.bytes_to_f32(m.read(src, 1024 * cnt))
            W = 16 * cnt
            out = []
            for i in range(16):
                out.extend(N.layernorm(x[i * W:(i + 1) * W], scalar))
            m.write(dst, N.f16_bytes(N.to_f16_array(out)))
        else:
            raise ValueError(f"golden nmc op {nmc_op}")


def _blk_get(a, i, j):
    b = j // BN; jj = j % BN
    return a[b * BN * BN + i * BN + jj]


def _blk_set(d, i, j, v):
    b = j // BN; jj = j % BN
    d[b * BN * BN + i * BN + jj] = v


def _abs_local(tile, off):
    return LOCAL_BASE + tile * LOCAL_STRIDE + off


def _kv(toks):
    out = {}
    for tk in toks:
        if "=" in tk:
            k, v = tk.split("=", 1)
            out[k] = v
    return out


def _i(kv, k, default=None):
    if k not in kv:
        if default is None:
            raise KeyError(k)
        return default
    return int(kv[k], 0)


def _f(kv, k, default=0.0):
    return float(kv[k]) if k in kv else default
