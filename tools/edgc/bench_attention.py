#!/usr/bin/env python3
"""P4-A — head-parallel attention tiling (decode phase).

Decode attention per head is tiny (QK^T: 1 x d_head x L, PV: 1 x L x d_head).
Run serially on one tile the 16-tile array is 1/16 utilized. Head-parallel
assigns head h -> tile (h % 16), so up to min(n_heads, 16) tiles work at
once with no inter-tile communication (each head is independent). Timing
only; measures active tiles + cycles.

  python3 bench_attention.py --sim ../../build/src/trace_sim
"""
import argparse
import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgc import ModelConfig, Compiler, Sched
from edgc import isa
from edgc.arch import load_arch, resolve_ramulator

BN, F16 = 16, 2


def emit_gemm_on_tile(comp, tp, a, b, c, M, K, N):
    """A whole (M x K)*(K x N) GEMM emitted on a single tile `tp` (f16,
    double-buffered K-loop). Used to pin a head's attention to one tile."""
    LA, LB, LC, LT2 = 0x0, 0x4000, 0x8000, 0x10000
    mb, kb, nb = M // BN, K // BN, N // BN

    def ld(base, ld_, r, cc, off):
        tp.dma(isa.dram(base + (r * BN * ld_ + cc * BN) * F16),
               isa.local(tp.tid, off), BN, BN * F16, ld_ * F16, BN * F16)
    for m in range(mb):
        for n in range(nb):
            D = min(3, kb) if kb >= 2 else 1
            if kb >= 2:
                As = [LA + i * 0x800 for i in range(D)]
                Bs = [LB + i * 0x800 for i in range(D)]
                for j in range(D - 1):
                    ld(a, K, m, j, As[j % D]); ld(b, N, j, n, Bs[j % D])
                for kk in range(kb):
                    s = kk % D; nj = kk + (D - 1)
                    if nj < kb:
                        ld(a, K, m, nj, As[nj % D]); ld(b, N, nj, n, Bs[nj % D])
                    ahead = min(kb - 1, kk + D - 1) - kk
                    tp.dma_fence(keep=2 * ahead)
                    tp.mxu("MXU_F16F16", As[s], Bs[s], LC, kk > 0, "qk")
                tp.wait_mxu()
            else:
                for kk in range(kb):
                    ld(a, K, m, kk, LA); ld(b, N, kk, n, LB)
                    tp.dma_fence()
                    tp.mxu("MXU_F16F16", LA, LB, LC, kk > 0, "qk")
                    tp.wait_mxu()
            tp.vpu("VPU_CVT_F32_F16", a=LC, d=LT2)
            tp.dma(isa.local(tp.tid, LT2), isa.dram(c + (m * BN * N + n * BN) * F16),
                   BN, BN * F16, BN * F16, N * F16)


def build(n_heads, d_head, L, mode, out_dir, sched, ram):
    cfg = ModelConfig(name=f"attn_{mode}_h{n_heads}", d_model=64, n_layers=1,
                      n_heads=4, n_kv_heads=4, d_head=16, ffn_hidden=64, seq_len=16)
    comp = Compiler(cfg, None, sched, ram)
    comp.b.set_config(ramulator=ram, tile_link=sched.tile_link, wr_ports=sched.wr_ports,
                      rd_ports=sched.rd_ports, dma_rate=sched.dma_rate,
                      dram_density=sched.dram_density, nmc_enable=0)
    M = BN                                   # one token block (decode)
    for h in range(n_heads):
        t = 0 if mode == "single" else (h % isa.NUM_TILES)
        tp = comp.b.t(t)
        q = comp.dalloc.alloc(M * d_head * F16)
        k = comp.dalloc.alloc(L * d_head * F16)
        s = comp.dalloc.alloc(M * L * F16)
        v = comp.dalloc.alloc(L * d_head * F16)
        ctx = comp.dalloc.alloc(M * d_head * F16)
        emit_gemm_on_tile(comp, tp, q, k, s, M, d_head, L)      # QK^T
        tp.vpu("VPU_SOFTMAX_BLK", a=0x8000, d=0x8000, scalar=1.0, count=L // BN)
        emit_gemm_on_tile(comp, tp, s, v, ctx, M, L, d_head)    # P@V
    for t in range(isa.NUM_TILES):
        comp.b.t(t).halt()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, cfg.name)
    comp.b.write_trace(base + ".trace"); comp.b.write_mem(base + ".mem")
    return base


def run(sim, base, ram, out_dir):
    r = subprocess.run([sim, base + ".trace", base + ".mem", "--ramulator", ram,
                        "--out-dir", out_dir, "--json"], capture_output=True, text=True)
    if r.returncode not in (0, 3):
        sys.stderr.write(r.stdout + r.stderr); raise RuntimeError(f"sim {r.returncode}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../build/src/trace_sim")
    ap.add_argument("--out", default="/tmp/edgc_attn")
    ap.add_argument("--arch", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    arch = load_arch(args.arch); ram = resolve_ramulator(arch)
    d = next(x for x in arch.dse if x["label"] == arch.default_sched)
    sched = Sched(label=d["label"], tile_link=d["tile_link"], wr_ports=d["wr_ports"],
                  rd_ports=d["rd_ports"], dma_rate=d["dma_rate"],
                  dram_density=d["dram_density"], nmc=d["nmc"])
    # (name, n_heads, d_head, L) from real decode attention configs
    configs = [("Llama-1B", 32, 64, 512), ("Llama-3B", 24, 128, 512),
               ("Mistral-7B", 32, 128, 512), ("edge(4h)", 4, 16, 512)]
    rows = []
    print(f"{'config':12}{'heads':>6}{'dh':>5}{'L':>5} | "
          f"{'tiles_1':>8}{'cyc_1':>8} | {'tiles_hp':>9}{'cyc_hp':>8} {'speedup':>8}")
    for nm, H, dh, L in configs:
        r1 = run(args.sim, build(H, dh, L, "single", args.out, sched, ram), ram, args.out)
        rh = run(args.sim, build(H, dh, L, "hp", args.out, sched, ram), ram, args.out)
        t1 = sum(1 for t in r1["tiles"] if t["mxu_busy"] > 0)
        th = sum(1 for t in rh["tiles"] if t["mxu_busy"] > 0)
        sp = r1["total_cycles"] / rh["total_cycles"]
        print(f"{nm:12}{H:>6}{dh:>5}{L:>5} | {t1:>8}{r1['total_cycles']:>8} | "
              f"{th:>9}{rh['total_cycles']:>8} {sp:>7.2f}x")
        rows.append(dict(config=nm, heads=H, d_head=dh, ctx=L,
                         tiles_single=t1, cyc_single=r1["total_cycles"],
                         tiles_hp=th, cyc_hp=rh["total_cycles"],
                         tile_util_single_pct=round(100 * t1 / 16, 1),
                         tile_util_hp_pct=round(100 * th / 16, 1),
                         speedup=round(sp, 3)))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
