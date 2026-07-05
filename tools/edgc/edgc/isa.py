"""Tile-ISA emitter and PGAS address helpers for the MOBOL target.

Mirrors src/cycle/isa.h (opcode names, field names) and src/common/address.h
(segment encoding). The compiler builds per-tile instruction lists and
serializes them to the text .trace format that src/cycle/trace_loader.cpp
parses. Field names here MUST match the loader's key= names.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# ── Address space (src/common/address.h) ──────────────────────
# Topology comes from the unified architecture YAML (config/mobol_arch.yaml)
# via edgc.arch, so it is not hardcoded here. The address-encoding bases are
# fixed by the PGAS spec (address.h) and are constants.
from .arch import load_arch as _load_arch

try:
    _ARCH = _load_arch()
    NUM_TILES = _ARCH.num_tiles
    NUM_BANKS = _ARCH.num_banks
    TILES_PER_GROUP = _ARCH.tiles_per_group
    LOCAL_SIZE = _ARCH.local_bytes
    SHARED_SIZE = _ARCH.shared_bytes
except Exception:
    NUM_TILES, NUM_BANKS, TILES_PER_GROUP = 16, 4, 4
    LOCAL_SIZE, SHARED_SIZE = 1 << 18, 1 << 23

LOCAL_BASE = 0x20 << 32
LOCAL_STRIDE = 0x2 << 32
SHARED_BASE = 0x40 << 32
SHARED_STRIDE = 0x8 << 31
DRAM_BASE = 0x80 << 32


def local(tile: int, off: int) -> int:
    return LOCAL_BASE + tile * LOCAL_STRIDE + off


def shared(bank: int, off: int) -> int:
    return SHARED_BASE + bank * SHARED_STRIDE + off


def dram(off: int) -> int:
    return DRAM_BASE + off


def tile_group(tile: int) -> int:
    return tile >> 2


# NmcOp enum values (src/cycle/flit.h)
NMC_REDUCE_F32 = 1
NMC_LN_F16 = 2
NMC_PREFETCH = 3


@dataclass
class Instr:
    op: str
    fields: Dict[str, object] = field(default_factory=dict)

    def line(self) -> str:
        parts = [self.op]
        for k, v in self.fields.items():
            if isinstance(v, int):
                parts.append(f"{k}=0x{v:x}" if v >= 16 or k in (
                    "src", "dst", "a", "b", "d") else f"{k}={v}")
            elif isinstance(v, float):
                parts.append(f"{k}={v!r}")
            else:
                parts.append(f"{k}={v}")
        return " ".join(parts)


class TileProgram:
    """Instruction list for one tile."""
    def __init__(self, tid: int):
        self.tid = tid
        self.code: List[Instr] = []

    def emit(self, op: str, **fields) -> None:
        self.code.append(Instr(op, fields))

    # ── DMA ──
    def dma(self, src, dst, rows, row_bytes, src_stride, dst_stride, note=""):
        self.emit("DMA", src=src, dst=dst, rows=rows, row_bytes=row_bytes,
                  src_stride=src_stride, dst_stride=dst_stride, **_note(note))

    def dma_linear(self, src, dst, nbytes, note=""):
        self.dma(src, dst, 1, nbytes, 0, 0, note)

    def dma_fence(self):
        self.emit("DMA_FENCE")

    # ── MXU ──
    def mxu(self, op, a, b, d, acc, note=""):
        self.emit(op, a=a, b=b, d=d, acc=1 if acc else 0, **_note(note))

    def wait_mxu(self):
        self.emit("WAIT_MXU")

    # ── VPU ──
    def vpu(self, op, a=0, b=0, d=0, scalar=None, count=1, aux=None,
            rows=None, row_bytes=None, note=""):
        f = dict(a=a, b=b, d=d, count=count)
        if scalar is not None:
            f["scalar"] = float(scalar)
        if aux is not None:
            f["aux"] = aux
        if rows is not None:
            f["rows"] = rows
        if row_bytes is not None:
            f["row_bytes"] = row_bytes
        f.update(_note(note))
        self.emit(op, **f)

    # ── SHARED direct ──
    def load_shared(self, src, dst, rows, row_bytes, src_stride, dst_stride):
        self.emit("LOAD_SHARED", src=src, dst=dst, rows=rows,
                  row_bytes=row_bytes, src_stride=src_stride, dst_stride=dst_stride)

    # ── NMC ──
    def nmc(self, nmc_op, src, dst, count, tag, scalar=0.0, note=""):
        self.emit("NMC", src=src, dst=dst, count=count, tag=tag,
                  arity=nmc_op, scalar=float(scalar), **_note(note))

    def nmc_prefetch(self, src_dram, dst_shared, rows, row_bytes,
                     src_stride, dst_stride, tag, note=""):
        self.emit("NMC", src=src_dram, dst=dst_shared, rows=rows,
                  row_bytes=row_bytes, src_stride=src_stride,
                  dst_stride=dst_stride, tag=tag, arity=NMC_PREFETCH, **_note(note))

    # ── Sync ──
    def release(self, consumer, tag, note=""):
        self.emit("RELEASE", consumer=consumer, tag=tag, **_note(note))

    def acquire(self, tag, arity, note=""):
        self.emit("ACQUIRE", tag=tag, arity=arity, **_note(note))

    def halt(self):
        self.emit("HALT")


def _note(s):
    return {"note": s.replace(" ", "_")} if s else {}


class TraceBuilder:
    """Whole-chip trace: 16 tile programs, config, DRAM image, dump regions."""
    def __init__(self):
        self.tiles = [TileProgram(t) for t in range(NUM_TILES)]
        self.config: Dict[str, object] = {}
        self.dumps: List[tuple] = []       # (name, dram_off, bytes)
        self.mem_records: List[tuple] = []  # (dram_off, bytes)

    def t(self, tid: int) -> TileProgram:
        return self.tiles[tid]

    def set_config(self, **kw):
        self.config.update(kw)

    def add_dump(self, name, dram_off, nbytes):
        self.dumps.append((name, dram_off, nbytes))

    def load_dram(self, dram_off: int, data: bytes):
        self.mem_records.append((dram_off, data))

    def write_trace(self, path: str):
        with open(path, "w") as f:
            if self.config:
                cfg = " ".join(f"{k}={v}" for k, v in self.config.items())
                f.write(f".config {cfg}\n")
            for name, off, nb in self.dumps:
                f.write(f".dump name={name} off=0x{off:x} bytes={nb}\n")
            for tp in self.tiles:
                if not tp.code:
                    tp.halt()
                f.write(f".tile {tp.tid}\n")
                for ins in tp.code:
                    f.write("  " + ins.line() + "\n")

    def write_mem(self, path: str):
        import struct
        with open(path, "wb") as f:
            for off, data in self.mem_records:
                f.write(struct.pack("<QQ", off, len(data)))
                f.write(data)
