"""Lowering compiler: ModelConfig + Weights -> tile-ISA trace + DRAM image.

Design (edge-scale, seq and d_model multiples of 16):
  * The whole prefill runs on the tile array with an OUTPUT-BLOCK-STATIONARY
    GEMM mapping. A logical matmul (SxK)*(KxN) tiles into 16x16 blocks;
    output block (m,n) is computed on one tile by accumulating over K
    blocks (fixed k-order => bit-reproducible). Blocks are round-robin
    assigned across the 16 tiles for parallelism.
  * Activations/weights are staged in DRAM; the tile pulls the operand
    blocks it needs, runs MXU/VPU, writes results back to DRAM. Between
    dependent ops a global DMA_FENCE + a barrier tag serialize the phases
    (correct and simple; the DSE explores overlap-friendlier schedules).
  * Attention, RoPE, norms, SwiGLU/GELU, and MoE routing all lower to the
    same block primitives; data-dependent MoE routing is resolved at
    compile time from the reference forward (static per given input).

The emitted trace is validated two ways (see edgc.golden): golden==sim and
golden==reference_forward.

DRAM scratch layout (offsets in bytes):
  0x00000  X       : seq x d_model  f16   (in/out activation, ping-pong)
  weights are placed by gen; we re-emit them into a compact region.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import os
from . import isa
from . import numerics as N
from .model import ModelConfig, Weights, _rope_tables, routing_decisions

BN = 16
F16 = 2
F32 = 4


class DramAlloc:
    def __init__(self, base=0x1000):
        self.p = base
    def alloc(self, nbytes, align=64):
        self.p = (self.p + align - 1) & ~(align - 1)
        a = self.p
        self.p += nbytes
        return a


@dataclass
class Sched:
    """DSE knobs the compiler chooses among."""
    stream_weights: bool = False   # bank-prefetch weights (needs buffer die)
    nmc: bool = False              # buffer-die near-memory reduce/LN
    tile_link: int = 1
    wr_ports: int = 1
    rd_ports: int = 2
    dma_rate: int = 1
    dram_density: int = 2048
    label: str = "baseline"


class Compiler:
    def __init__(self, cfg: ModelConfig, w: Weights, sched: Sched,
                 ramulator: str):
        self.cfg = cfg
        self.w = w
        self.s = sched
        self.b = isa.TraceBuilder()
        self.dalloc = DramAlloc()
        self.ram = ramulator
        self.tag = 0           # global barrier tag counter
        # per-tile local scratch bump allocator resets per op via fixed map
        # DRAM regions
        S, dm = cfg.seq_len, cfg.d_model
        self.act_a = self.dalloc.alloc(S * dm * F16)
        self.act_b = self.dalloc.alloc(S * dm * F16)
        self.scratch = {}      # named DRAM scratch
        # ── Priority-1 barrier elimination ───────────────────────────
        # Each primitive declares the DRAM buffers it reads/writes; _sync
        # inserts a global barrier ONLY when the new primitive conflicts
        # (RAW/WAR/WAW) with a buffer touched since the last barrier. This
        # drops barriers between provably-independent primitives (the three
        # QKV projections, independent MoE experts, ...) while preserving
        # every real cross-tile data dependency. Mode "global" (env
        # EDGC_BARRIER=global) forces a barrier before every primitive and
        # reproduces the original phase-serialized schedule bit-for-bit —
        # the A/B baseline. Default "dep" is the optimized schedule.
        self.dep_barriers = os.environ.get("EDGC_BARRIER", "dep") != "global"
        self._tr = set()       # DRAM buffers read   since last barrier
        self._tw = set()       # DRAM buffers written since last barrier
        self._emitted_any = False
        self.barrier_count = 0
        # P2: multi-buffered GEMM inner loop (prefetch operands D-1 steps
        # ahead while the MXU computes). EDGC_DBUF=0 disables (single-buffer
        # A/B path). EDGC_DBUF_DEPTH sets the number of slots (2 = classic
        # double buffer); deeper hides DRAM read latency to saturate the bond.
        self.dbuf = os.environ.get("EDGC_DBUF", "1") != "0"
        self.dbuf_depth = max(2, min(8, int(os.environ.get("EDGC_DBUF_DEPTH", "3"))))

    # ── public entry ──
    def compile(self):
        cfg = self.cfg
        self.b.set_config(
            ramulator=self.ram,
            tile_link=self.s.tile_link, wr_ports=self.s.wr_ports,
            rd_ports=self.s.rd_ports, dma_rate=self.s.dma_rate,
            dram_density=self.s.dram_density,
            nmc_enable=1 if self.s.nmc else 0)
        self._emit_weights_and_x()
        cur = self.act_a
        nxt = self.act_b
        routes = routing_decisions(cfg, self.w) if cfg.n_experts > 1 else None
        for l in range(cfg.n_layers):
            cur = self._layer(l, cur, nxt, routes)
            nxt = self.act_a if cur == self.act_b else self.act_b
        # final activation is at `cur`. One closing barrier fences every
        # tile's outstanding store DMAs before halt so the `out` dump reads
        # a fully-committed image (both modes; negligible vs. removed count).
        self._barrier()
        S, dm = cfg.seq_len, cfg.d_model
        self.b.add_dump("out", cur - isa.DRAM_BASE if cur >= isa.DRAM_BASE else cur,
                        S * dm * F16)
        self.final_off = (cur - isa.DRAM_BASE) if cur >= isa.DRAM_BASE else cur
        for t in range(isa.NUM_TILES):
            self.b.t(t).halt()
        return self.b

    # ── weight / input staging ──
    def _emit_weights_and_x(self):
        cfg, w = self.cfg, self.w
        dm, dkv, ff = cfg.d_model, cfg.d_kv, cfg.ffn_hidden
        self.w_addr = {}
        def put(name, bits):
            off = self.dalloc.alloc(len(bits) * F16)
            self.b.load_dram(off, N.f16_bytes(bits))
            self.w_addr[name] = off
        put("x0", w.x0)
        # x0 lives at act_a; copy image directly there instead
        self.b.load_dram(self.act_a, N.f16_bytes(w.x0))
        for l in range(cfg.n_layers):
            put(f"wq{l}", w.wq[l]); put(f"wk{l}", w.wk[l])
            put(f"wv{l}", w.wv[l]); put(f"wo{l}", w.wo[l])
            for e in range(cfg.n_experts):
                put(f"wg{l}_{e}", w.w_gate[l][e])
                put(f"wu{l}_{e}", w.w_up[l][e])
                put(f"wd{l}_{e}", w.w_down[l][e])
            if cfg.n_experts > 1:
                put(f"wr{l}", w.w_router[l])
        # rope tables (f32, per position block) staged in DRAM
        if cfg.pos == "rope":
            self._emit_rope_tables()

    def _emit_rope_tables(self):
        cfg = self.cfg
        cos, sin = _rope_tables(cfg)
        half = cfg.d_head // 2
        # Build, per head, 16x16 f32 cos/sin blocks broadcasting the half
        # angle across the 16 columns of the first half-block. For d_head=16
        # (half=8) we lay the 8 angles into columns 0..7 and replicate to
        # 8..15 (rotate-half pairs (k, k+8)). Stored as [pos-block][cos|sin].
        # seq is a multiple of 16 -> one position block covers 16 rows.
        nblk = cfg.seq_len // BN
        self.rope_addr = self.dalloc.alloc(nblk * 2 * 1024)  # cos+sin per blk
        buf = bytearray()
        import struct
        for pb in range(nblk):
            for which in (cos, sin):
                blk = [0.0] * 256
                for i in range(BN):
                    p = pb * BN + i
                    for k in range(half):
                        blk[i * BN + k] = which[p][k]
                        blk[i * BN + half + k] = which[p][k]
                buf += b"".join(struct.pack("<f", v) for v in blk)
        self.b.load_dram(self.rope_addr, bytes(buf))

    # ── a full transformer layer ──
    def _layer(self, l, x_in, x_out, routes):
        cfg = self.cfg
        dm, dkv, ff = cfg.d_model, cfg.d_kv, cfg.ffn_hidden
        S = cfg.seq_len
        # scratch DRAM regions for this layer's intermediates
        xn = self.dalloc.alloc(S * dm * F16)
        q = self.dalloc.alloc(S * dm * F16)
        k = self.dalloc.alloc(S * dkv * F16)
        v = self.dalloc.alloc(S * dkv * F16)
        ctx = self.dalloc.alloc(S * dm * F16)
        attn = self.dalloc.alloc(S * dm * F16)
        res1 = self.dalloc.alloc(S * dm * F16)
        hn = self.dalloc.alloc(S * dm * F16)

        # 1. pre-attention norm  (x_in -> xn)
        self._norm(x_in, xn, S, dm)
        # 2. QKV projections
        self._gemm(xn, self.w_addr[f"wq{l}"], q, S, dm, dm, f"q{l}")
        self._gemm(xn, self.w_addr[f"wk{l}"], k, S, dm, dkv, f"k{l}")
        self._gemm(xn, self.w_addr[f"wv{l}"], v, S, dm, dkv, f"v{l}")
        # 3. RoPE on q,k
        if cfg.pos == "rope":
            self._rope(q, S, cfg.n_heads, dm)
            self._rope(k, S, cfg.n_kv_heads, dkv)
        # 4. attention -> ctx
        self._attention(q, k, v, ctx, l)
        # 5. output projection + residual
        self._gemm(ctx, self.w_addr[f"wo{l}"], attn, S, dm, dm, f"o{l}")
        self._residual(x_in, attn, res1, S, dm)
        # 6. ffn norm
        self._norm(res1, hn, S, dm)
        # 7. FFN (dense or MoE) + residual -> x_out
        self._ffn(l, hn, res1, x_out, routes, S, dm, ff)
        return x_out

    # ── primitives (all block-parallel over tiles) ──
    def _sync(self, reads=(), writes=()):
        """Priority-1 dependency-aware barrier insertion. Emit a global
        barrier before this primitive ONLY if it conflicts with a buffer
        touched since the last barrier (RAW: reads a pending write; WAW:
        overwrites a pending write; WAR: overwrites a pending read).
        Read-read is not a conflict, so independent primitives that share
        an input (QKV all read `xn`; MoE experts all read `hn`) run without
        an intervening barrier. `global` mode forces a barrier before every
        primitive, reproducing the original schedule."""
        rd = {b for b in reads if b is not None}
        wr = {b for b in writes if b is not None}
        if self.dep_barriers:
            conflict = bool(rd & self._tw) or bool(wr & self._tw) or bool(wr & self._tr)
        else:
            conflict = self._emitted_any  # barrier before every primitive
        if conflict:
            self._barrier()
            self._tr.clear(); self._tw.clear()
        self._tr |= rd
        self._tw |= wr
        self._emitted_any = True

    def _barrier(self):
        """Global barrier: every tile releases to tile0, tile0 waits, then
        releases back. Simple phase serialization."""
        self.barrier_count += 1
        bt = self.tag; self.tag += 1
        # producers: all tiles release to tile 0
        for t in range(isa.NUM_TILES):
            self.b.t(t).dma_fence()
            self.b.t(t).release(0, bt, "barrier_up")
        self.b.t(0).acquire(bt, isa.NUM_TILES, "barrier_join")
        bt2 = self.tag; self.tag += 1
        for t in range(isa.NUM_TILES):
            self.b.t(0).release(t, bt2, "barrier_down")
        for t in range(isa.NUM_TILES):
            self.b.t(t).acquire(bt2, 1, "barrier_go")

    def _tile_for(self, idx):
        return idx % isa.NUM_TILES

    # LOCAL scratch map. Each logical buffer gets a disjoint 16 KB slot so
    # a strip of up to 16 blocks (f32) never overlaps a temp. LOCAL is
    # 256 KB/tile, so this is comfortable. Sub-offsets used in code
    # (LB+nj*512, LC+nj*1024, LA+0x1000, LT+0x2000, LB+0x1000) all stay
    # within their owner's slot.
    LA = 0x00000   # operand A strip
    LB = 0x04000   # operand B strip
    LC = 0x08000   # f32 accumulator / result strip
    LT = 0x0C000   # temp f32 strip
    LT2 = 0x10000  # small f16 temp (1 block)
    LROPE = 0x14000  # rope cos/sin (2 KB)

    def _dram_blk(self, base, ld, r, c, esz=F16):
        return isa.dram(base + (r * BN * ld + c * BN) * esz)

    def _load_block(self, tp, base, ld, r, c, off, esz=F16):
        tp.dma(self._dram_blk(base, ld, r, c, esz), isa.local(tp.tid, off),
               BN, BN * esz, ld * esz, BN * esz)

    def _load_block_shared(self, tp, base, ld, r, c, off, esz=F16):
        """Pull a 16x16 block of a SHARED-resident operand from this tile's
        NEAR bank (steady-state residency: e.g. the decode KV cache lives in
        the buffer die, so reads ride the vertical bond instead of DRAM).
        Timing-only path: the bank SRAM is zero-initialized, which is fine
        for shape-driven Tier-1 runs."""
        bank = isa.tile_group(tp.tid)
        src = isa.shared(bank, base + (r * BN * ld + c * BN) * esz)
        tp.dma(src, isa.local(tp.tid, off), BN, BN * esz, ld * esz, BN * esz)

    def _store_block(self, tp, off, base, ld, r, c, esz=F16):
        tp.dma(isa.local(tp.tid, off), self._dram_blk(base, ld, r, c, esz),
               BN, BN * esz, BN * esz, ld * esz)

    # Low-precision MMA timing model. Per MXU issue the operand footprint is
    # a fixed 512 B slab, but a lower precision packs more of the K axis into
    # it (INT8: 2 K-blocks, INT4: 4) => 2x/4x MAC throughput. eb = element
    # bytes; kpo = K-blocks packed per issue; op = MXU opcode.
    PREC = {
        "f16": dict(eb=2.0, kpo=1, op="MXU_F16F16"),
        "i8":  dict(eb=1.0, kpo=2, op="MXU_I8I8"),
        "i4":  dict(eb=0.5, kpo=4, op="MXU_I4I4"),
    }

    def _gemm(self, a_base, b_base, c_base, M, K, N_, name, prec="f16",
              b_shared=False):
        """C(MxN) = A(MxK)*B(KxN). Output block (m,n) accumulated over the K
        axis on one tile. `prec` selects the MMA precision (timing model):
        f16 processes one 16-K-block per MXU issue, i8 two, i4 four, with
        DRAM operands sized at the precision's element width. `b_shared`
        reads B from the tile's near buffer-die bank instead of DRAM
        (steady-state SHARED residency, e.g. the decode KV cache);
        `b_base` is then a byte offset inside the bank."""
        self._sync(reads=[a_base, ("sh", b_base) if b_shared else b_base],
                   writes=[c_base])
        if b_shared:
            eb = self.PREC[prec]["eb"]
            assert int(K * N_ * eb) <= isa.SHARED_SIZE, \
                f"SHARED-resident operand {K}x{N_} exceeds bank capacity"
        if prec != "f16":
            return self._gemm_prec(a_base, b_base, c_base, M, K, N_, name,
                                   prec, b_shared)
        mb, kb, nb = M // BN, K // BN, N_ // BN
        ldb = self._load_block_shared if b_shared else self._load_block
        idx = 0
        for m in range(mb):
            for n in range(nb):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                if self.dbuf and kb >= 2:
                    self._gemm_block_dbuf(tp, a_base, b_base, K, N_, m, n, kb,
                                          name, ldb)
                else:
                    for kk in range(kb):
                        self._load_block(tp, a_base, K, m, kk, self.LA)
                        ldb(tp, b_base, N_, kk, n, self.LB)
                        tp.dma_fence()
                        tp.mxu("MXU_F16F16", self.LA, self.LB, self.LC, kk > 0, name)
                        tp.wait_mxu()
                # convert f32 acc -> f16 and store
                tp.vpu("VPU_CVT_F32_F16", a=self.LC, d=self.LT2)
                self._store_block(tp, self.LT2, c_base, N_, m, n)

    def _gemm_block_dbuf(self, tp, a_base, b_base, K, N_, m, n, kb, name,
                         ldb=None):
        """One output block via a depth-D multi-buffered K-loop. D operand
        slots; the loop keeps D-1 K-steps prefetched ahead while the MXU
        computes the current step. `DMA_FENCE keep=2*ahead` drains only the
        current step's operands (the oldest outstanding) and leaves the
        prefetched steps in flight, so the DMA engine overlaps the MXU and
        stays fed across DRAM read latency. No WAIT_MXU between steps: the
        MXU's in-flight accumulator forwarding chains the k-accumulation."""
        if ldb is None:
            ldb = self._load_block
        D = min(self.dbuf_depth, kb)
        A = [self.LA + i * 0x800 for i in range(D)]  # D A slots (512 B blocks)
        B = [self.LB + i * 0x800 for i in range(D)]  # D B slots
        # prologue: prefetch the first D-1 steps
        for j in range(D - 1):
            self._load_block(tp, a_base, K, m, j, A[j % D])
            ldb(tp, b_base, N_, j, n, B[j % D])
        for kk in range(kb):
            s = kk % D
            nj = kk + (D - 1)             # step to prefetch this iteration
            if nj < kb:
                self._load_block(tp, a_base, K, m, nj, A[nj % D])
                ldb(tp, b_base, N_, nj, n, B[nj % D])
            # steps still allowed outstanding ahead of the current step
            ahead = min(kb - 1, kk + D - 1) - kk
            tp.dma_fence(keep=2 * ahead)  # current step's operands drained
            tp.mxu("MXU_F16F16", A[s], B[s], self.LC, kk > 0, name)
        tp.wait_mxu()                     # accumulator committed before convert

    def _gemm_prec(self, a_base, b_base, c_base, M, K, N_, name, prec,
                   b_shared=False):
        """Low-precision (INT8/INT4) GEMM, timing model. Each MXU issue packs
        `kpo` K-blocks into the fixed 512 B operand slab, so a lower precision
        needs proportionally fewer issues and moves fewer DRAM bytes. Reuses
        the depth-D prefetch pipeline (P2)."""
        p = self.PREC[prec]; eb = p["eb"]; kpo = p["kpo"]; op = p["op"]
        kw = BN * kpo                     # K columns packed per issue
        assert K % kw == 0, f"{prec} GEMM needs K ({K}) divisible by {kw}"
        mb, nb, kops = M // BN, N_ // BN, K // kw
        ib = lambda x: int(round(x))      # byte count (eb may be 0.5 for i4)

        def load_A(tp, m, j, off):        # A slab: 16 rows x kw cols (M x K)
            src = isa.dram(a_base + ib((m * BN * K + j * kw) * eb))
            tp.dma(src, isa.local(tp.tid, off), BN, ib(kw * eb),
                   ib(K * eb), ib(kw * eb))

        def load_B(tp, j, n, off):        # B slab: kw rows x 16 cols (K x N)
            if b_shared:                  # near-bank residency (see _gemm)
                src = isa.shared(isa.tile_group(tp.tid),
                                 b_base + ib((j * kw * N_ + n * BN) * eb))
            else:
                src = isa.dram(b_base + ib((j * kw * N_ + n * BN) * eb))
            tp.dma(src, isa.local(tp.tid, off), kw, ib(BN * eb),
                   ib(N_ * eb), ib(BN * eb))

        idx = 0
        D = min(self.dbuf_depth, kops)
        for m in range(mb):
            for n in range(nb):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                if self.dbuf and kops >= 2:
                    A = [self.LA + i * 0x800 for i in range(D)]
                    B = [self.LB + i * 0x800 for i in range(D)]
                    for jp in range(D - 1):
                        load_A(tp, m, jp, A[jp % D]); load_B(tp, jp, n, B[jp % D])
                    for j in range(kops):
                        s = j % D; nj = j + (D - 1)
                        if nj < kops:
                            load_A(tp, m, nj, A[nj % D]); load_B(tp, nj, n, B[nj % D])
                        ahead = min(kops - 1, j + D - 1) - j
                        tp.dma_fence(keep=2 * ahead)
                        tp.mxu(op, A[s], B[s], self.LC, j > 0, name)
                    tp.wait_mxu()
                else:
                    for j in range(kops):
                        load_A(tp, m, j, self.LA); load_B(tp, j, n, self.LB)
                        tp.dma_fence()
                        tp.mxu(op, self.LA, self.LB, self.LC, j > 0, name)
                        tp.wait_mxu()
                tp.vpu("VPU_CVT_F32_F16", a=self.LC, d=self.LT2)
                self._store_block(tp, self.LT2, c_base, N_, m, n)

    def _norm(self, x_base, out_base, S, dm):
        """RMSNorm/LayerNorm each row (width dm = dm/16 blocks). One row-
        block (16 rows) per tile; norm needs the whole row so we load the
        dm-wide strip into BLOCKED layout and run VPU_*NORM_BLK."""
        self._sync(reads=[x_base], writes=[out_base])
        op = "VPU_RMSNORM_BLK" if self.cfg.norm == "rmsnorm" else "VPU_LAYERNORM_F32"
        cnt = dm // BN
        rb = S // BN
        idx = 0
        for r in range(rb):
            tp = self.b.t(self._tile_for(idx)); idx += 1
            # load dm-wide strip: block c -> LA + c*1024 (as f32 via convert)
            for c in range(cnt):
                self._load_block(tp, x_base, dm, r, c, self.LB)  # f16
                tp.dma_fence()
                tp.vpu("VPU_CVT_F16_F32", a=self.LB, d=self.LA + c * 1024)
            if op == "VPU_LAYERNORM_F32":
                # layernorm expects row-major 16 x (16*cnt); repack blocked
                # -> row-major into LT region is complex. Use blocked RMS/LN
                # via the BLK path for both by staging as blocked and using
                # RMSNORM only; for layernorm fall back to per-block gather.
                self._layernorm_rowmajor(tp, cnt)
                src = self.LT
            else:
                tp.vpu(op, a=self.LA, d=self.LC, scalar=self.cfg.norm_eps, count=cnt)
                src = self.LC
            # Convert all blocks to f16 in a distinct per-block slot (LT2 is a
            # 16-block region), then store — so the async store DMAs never
            # race a later convert reusing the same buffer (WAR hazard).
            for c in range(cnt):
                tp.vpu("VPU_CVT_F32_F16", a=src + c * 1024, d=self.LT2 + c * 512)
                self._store_block(tp, self.LT2 + c * 512, out_base, dm, r, c)
            tp.dma_fence()

    def _layernorm_rowmajor(self, tp, cnt):
        # gather blocked LA -> row-major LT (16 x 16*cnt f32), run LN, keep
        # result in LT (blocked) for uniform store. For simplicity reuse
        # RMSNorm's blocked layout by computing LN in blocked form via the
        # blocked op is not available; emit VPU_LAYERNORM over a row-major
        # pack. We pack with strided DMA (local->local).
        W = cnt * BN
        # pack: for block c, its 16x16 -> columns [c*16, c*16+16) of row-major
        for c in range(cnt):
            tp.dma(isa.local(tp.tid, self.LA + c * 1024),
                   isa.local(tp.tid, self.LT + c * BN * F32),
                   BN, BN * F32, BN * F32, W * F32)
        tp.dma_fence()
        tp.vpu("VPU_LAYERNORM_F32", a=self.LT, d=self.LT + 0x2000,
               scalar=self.cfg.norm_eps, count=cnt)
        # unpack row-major -> blocked LT
        for c in range(cnt):
            tp.dma(isa.local(tp.tid, self.LT + 0x2000 + c * BN * F32),
                   isa.local(tp.tid, self.LT + c * 1024),
                   BN, BN * F32, W * F32, BN * F32)
        tp.dma_fence()

    def _residual(self, a_base, b_base, out_base, S, dm):
        self._sync(reads=[a_base, b_base], writes=[out_base])
        cnt = dm // BN
        rb = S // BN
        idx = 0
        for r in range(rb):
            for c in range(cnt):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                self._load_block(tp, a_base, dm, r, c, self.LA)
                self._load_block(tp, b_base, dm, r, c, self.LB)
                tp.dma_fence()
                tp.vpu("VPU_CVT_F16_F32", a=self.LA, d=self.LC)
                tp.vpu("VPU_CVT_F16_F32", a=self.LB, d=self.LT)
                tp.vpu("VPU_ADD_F32", a=self.LC, b=self.LT, d=self.LC)
                tp.vpu("VPU_CVT_F32_F16", a=self.LC, d=self.LT2)
                self._store_block(tp, self.LT2, out_base, dm, r, c)

    def _rope(self, base, S, nheads, dmodel):
        # d_head == 16 => each head is exactly one 16-wide block; RoPE
        # rotates dim k with dim k+half inside that block.
        cfg = self.cfg
        assert cfg.d_head == BN, "compiler _rope assumes d_head == 16"
        self._sync(reads=[base], writes=[base])
        half = cfg.d_head // 2
        rb = S // BN
        idx = 0
        for r in range(rb):
            for h in range(nheads):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                col = h  # head h occupies block column h
                self._load_block(tp, base, dmodel, r, col, self.LA)
                cos_off = self.rope_addr + (r * 2) * 1024
                tp.dma_linear(isa.dram(cos_off), isa.local(tp.tid, self.LROPE), 1024)
                tp.dma_linear(isa.dram(cos_off + 1024),
                              isa.local(tp.tid, self.LROPE + 1024), 1024)
                tp.dma_fence()
                tp.vpu("VPU_ROPE_F16", a=self.LA, d=self.LT2, count=half,
                       aux=self.LROPE)
                self._store_block(tp, self.LT2, base, dmodel, r, col)

    def _attention(self, q, k, v, ctx, l):
        cfg = self.cfg
        self._sync(reads=[q, k, v], writes=[ctx])
        S, dm, dkv, dh = cfg.seq_len, cfg.d_model, cfg.d_kv, cfg.d_head
        gsz = cfg.group_size()
        # For edge sizes S<=16 (one block) and d_head=16 (one block): each
        # head is a single 16x16 attention. Generalize over seq-blocks.
        sb = S // BN
        idx = 0
        for h in range(cfg.n_heads):
            kvh = h // gsz
            qcol = (h * dh) // BN
            kcol = (kvh * dh) // BN
            for mi in range(sb):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                # scores S_blk = Q_blk (16 x dh) * K^T. For dh=16, one block.
                self._load_block(tp, q, dm, mi, qcol, self.LA)   # Q rows
                # load K rows for all key blocks, transpose, matmul-accumulate
                # scores into LC (16 x S). For S<=16 single block.
                for nj in range(sb):
                    self._load_block(tp, k, dkv, nj, kcol, self.LB)
                    tp.dma_fence()
                    tp.vpu("VPU_TRANS_F16", a=self.LB, d=self.LT2)  # K^T block
                    tp.mxu("MXU_F16F16", self.LA, self.LT2,
                           self.LC + nj * 1024, False, "scores")
                    tp.wait_mxu()  # scores committed before softmax reads LC
                # softmax over the S columns (blocked), causal + window
                row0 = mi * BN
                mask = row0 if cfg.attn_mask == "causal" else -1
                tp.vpu("VPU_SOFTMAX_BLK", a=self.LC, d=self.LT,
                       scalar=cfg.attn_scale(), count=sb, aux=mask,
                       rows=cfg.sliding_window, row_bytes=S)
                # convert probs to f16 per block
                for nj in range(sb):
                    tp.vpu("VPU_CVT_F32_F16", a=self.LT + nj * 1024, d=self.LB + nj * 512)
                # ctx = P * V, accumulate over key blocks
                for nj in range(sb):
                    self._load_block(tp, v, dkv, nj, kcol, self.LA + 0x1000)
                    tp.dma_fence()
                    tp.mxu("MXU_F16F16", self.LB + nj * 512, self.LA + 0x1000,
                           self.LC, nj > 0, "ctx")
                    tp.wait_mxu()  # ctx committed before the convert reads LC
                tp.vpu("VPU_CVT_F32_F16", a=self.LC, d=self.LT2)
                self._store_block(tp, self.LT2, ctx, dm, mi, qcol)
                tp.dma_fence()  # store done before this tile reuses LT2

    def _ffn(self, l, hn, res_base, out_base, routes, S, dm, ff):
        cfg = self.cfg
        if cfg.n_experts == 1:
            self._expert_dense(l, 0, hn, res_base, out_base, S, dm, ff, None)
        else:
            # MoE: compile-time routing. For each token-block we know the
            # experts. For edge scale (S<=16) all tokens share one block, so
            # we compute each needed expert on the full block then combine
            # per-token with the compiled gates (masking non-routed tokens).
            self._moe(l, hn, res_base, out_base, routes[l], S, dm, ff)

    def _expert_dense(self, l, e, hn, res_base, out_base, S, dm, ff, gate_mask):
        cfg = self.cfg
        g = self.dalloc.alloc(S * ff * F16)
        act = self.dalloc.alloc(S * ff * F16)
        self._gemm(hn, self.w_addr[f"wg{l}_{e}"], g, S, dm, ff, f"g{l}")
        if cfg.ffn == "swiglu":
            u = self.dalloc.alloc(S * ff * F16)
            self._gemm(hn, self.w_addr[f"wu{l}_{e}"], u, S, dm, ff, f"u{l}")
            self._silu_gate(g, u, act, S, ff)
        else:
            self._gelu(g, act, S, ff)
        down = self.dalloc.alloc(S * dm * F16)
        self._gemm(act, self.w_addr[f"wd{l}_{e}"], down, S, ff, dm, f"d{l}")
        self._residual(res_base, down, out_base, S, dm)

    def _silu_gate(self, g_base, u_base, out_base, S, ff):
        self._sync(reads=[g_base, u_base], writes=[out_base])
        cnt = ff // BN
        rb = S // BN
        idx = 0
        for r in range(rb):
            for c in range(cnt):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                self._load_block(tp, g_base, ff, r, c, self.LA)
                self._load_block(tp, u_base, ff, r, c, self.LB)
                tp.dma_fence()
                tp.vpu("VPU_CVT_F16_F32", a=self.LA, d=self.LC)
                tp.vpu("VPU_CVT_F16_F32", a=self.LB, d=self.LT)
                tp.vpu("VPU_SILU_F32", a=self.LC, d=self.LC)
                tp.vpu("VPU_MUL_F32", a=self.LC, b=self.LT, d=self.LC)
                tp.vpu("VPU_CVT_F32_F16", a=self.LC, d=self.LT2)
                self._store_block(tp, self.LT2, out_base, ff, r, c)

    def _gelu(self, g_base, out_base, S, ff):
        self._sync(reads=[g_base], writes=[out_base])
        cnt = ff // BN
        rb = S // BN
        idx = 0
        for r in range(rb):
            for c in range(cnt):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                self._load_block(tp, g_base, ff, r, c, self.LA)
                tp.dma_fence()
                tp.vpu("VPU_CVT_F16_F32", a=self.LA, d=self.LC)
                tp.vpu("VPU_GELU_F32", a=self.LC, d=self.LC)
                tp.vpu("VPU_CVT_F32_F16", a=self.LC, d=self.LT2)
                self._store_block(tp, self.LT2, out_base, ff, r, c)

    def _moe(self, l, hn, res_base, out_base, routes, S, dm, ff):
        # Compute a zero-initialized accumulator in DRAM, add each expert's
        # gated contribution for the tokens routed to it. Static routing =>
        # per-token gate weights are compile-time constants baked as a
        # diagonal scale applied after the expert's down-projection.
        cfg = self.cfg
        acc = self.dalloc.alloc(S * dm * F16)
        # init acc = 0 by storing zeros (emit a zero block image in DRAM)
        self.b.load_dram(acc, b"\x00" * (S * dm * F16))
        experts_used = sorted({e for tok in routes for (e, _g) in tok})
        for e in experts_used:
            # per-token gate for this expert (0 if not routed)
            gate = [0.0] * S
            for ti, tok in enumerate(routes):
                for (ee, gg) in tok:
                    if ee == e:
                        gate[ti] = gg
            eo = self.dalloc.alloc(S * dm * F16)
            self._expert_only(l, e, hn, eo, S, dm, ff)
            self._scale_add(eo, gate, acc, S, dm)
        self._residual(res_base, acc, out_base, S, dm)

    def _expert_only(self, l, e, hn, out_base, S, dm, ff):
        cfg = self.cfg
        g = self.dalloc.alloc(S * ff * F16)
        act = self.dalloc.alloc(S * ff * F16)
        self._gemm(hn, self.w_addr[f"wg{l}_{e}"], g, S, dm, ff, f"g{l}_{e}")
        if cfg.ffn == "swiglu":
            u = self.dalloc.alloc(S * ff * F16)
            self._gemm(hn, self.w_addr[f"wu{l}_{e}"], u, S, dm, ff, f"u{l}_{e}")
            self._silu_gate(g, u, act, S, ff)
        else:
            self._gelu(g, act, S, ff)
        self._gemm(act, self.w_addr[f"wd{l}_{e}"], out_base, S, ff, dm, f"d{l}_{e}")

    def _scale_add(self, e_base, gate, acc_base, S, dm):
        # acc[row r] += gate[token] * e_base[row r]; gate is per token (row).
        # Each 16-row block has 16 tokens => per-row scalar. We apply the
        # scale by building a per-token diagonal via VPU_SCALE on 16 single
        # rows is wasteful; instead bake gate into a 16x16 f32 "gate block"
        # broadcast and multiply. For simplicity here: token == row, and a
        # block spans BN tokens, so emit a gate image and VPU_MUL.
        self._sync(reads=[e_base, acc_base], writes=[acc_base])
        cnt = dm // BN
        rb = S // BN
        import struct
        # gate block per row-block: 16x16, row i filled with gate[r*16+i]
        gate_dram = self.dalloc.alloc(rb * 1024)
        buf = bytearray()
        for r in range(rb):
            blk = [0.0] * 256
            for i in range(BN):
                tok = r * BN + i
                for j in range(BN):
                    blk[i * BN + j] = gate[tok] if tok < S else 0.0
            buf += b"".join(struct.pack("<f", v) for v in blk)
        self.b.load_dram(gate_dram, bytes(buf))
        idx = 0
        for r in range(rb):
            for c in range(cnt):
                tp = self.b.t(self._tile_for(idx)); idx += 1
                self._load_block(tp, e_base, dm, r, c, self.LA)
                self._load_block(tp, acc_base, dm, r, c, self.LB)
                tp.dma_linear(isa.dram(gate_dram + r * 1024),
                              isa.local(tp.tid, self.LT), 1024)
                tp.dma_fence()
                tp.vpu("VPU_CVT_F16_F32", a=self.LA, d=self.LC)
                tp.vpu("VPU_MUL_F32", a=self.LC, b=self.LT, d=self.LC)
                tp.vpu("VPU_CVT_F16_F32", a=self.LB, d=self.LT2 - 0)  # acc->f32 in LT? need space
                # use LA region (free now) as acc-f32
                tp.vpu("VPU_CVT_F16_F32", a=self.LB, d=self.LB + 0x1000)
                tp.vpu("VPU_ADD_F32", a=self.LC, b=self.LB + 0x1000, d=self.LC)
                tp.vpu("VPU_CVT_F32_F16", a=self.LC, d=self.LT2)
                self._store_block(tp, self.LT2, acc_base, dm, r, c)
