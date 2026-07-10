#!/usr/bin/env python3
"""Tier-3 pure-GEMM microbenchmark for P2 (double-buffer) validation.

Emits a single output-block-stationary GEMM (M x K)*(K x N) as a tile-ISA
trace and runs it on the cycle-accurate simulator, sweeping K to expose the
double-buffered inner loop's steady state. Timing only (random/zero data).

  python3 bench_gemm.py --sim ../../build/src/trace_sim
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgc import ModelConfig, Compiler, Sched
from edgc.arch import load_arch, resolve_ramulator
from edgc.compile import F16


EB = {"f16": 2.0, "i8": 1.0, "i4": 0.5}


def build_gemm(M, K, N, out_dir, sched, ramulator, dbuf, prec="f16"):
    os.environ["EDGC_DBUF"] = "1" if dbuf else "0"
    cfg = ModelConfig(name=f"gemm_M{M}_K{K}_N{N}", d_model=64, n_layers=1,
                      n_heads=4, n_kv_heads=4, d_head=16, ffn_hidden=64,
                      seq_len=16)
    comp = Compiler(cfg, None, sched, ramulator)  # weights unused (timing only)
    comp.b.set_config(ramulator=ramulator, tile_link=sched.tile_link,
                      wr_ports=sched.wr_ports, rd_ports=sched.rd_ports,
                      dma_rate=sched.dma_rate, dram_density=sched.dram_density,
                      nmc_enable=0)
    eb = EB[prec]
    a = comp.dalloc.alloc(int(M * K * eb))
    b = comp.dalloc.alloc(int(K * N * eb))
    c = comp.dalloc.alloc(M * N * F16)
    # zero-fill operands so DRAM reads are valid (values irrelevant to timing)
    comp.b.load_dram(a, b"\x00" * int(M * K * eb))
    comp.b.load_dram(b, b"\x00" * int(K * N * eb))
    comp._gemm(a, b, c, M, K, N, "bench", prec=prec)
    comp._barrier()
    for t in range(16):
        comp.b.t(t).halt()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{cfg.name}_{prec}_{'db' if dbuf else 'sb'}")
    comp.b.write_trace(base + ".trace")
    comp.b.write_mem(base + ".mem")
    return base


def run(sim, base, ramulator, out_dir):
    r = subprocess.run([sim, base + ".trace", base + ".mem", "--ramulator",
                        ramulator, "--out-dir", out_dir, "--json"],
                       capture_output=True, text=True)
    if r.returncode not in (0, 3):
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"sim failed ({r.returncode})")
    return json.loads(r.stdout.strip().splitlines()[-1])


def metrics(s):
    tiles = s["tiles"]
    cyc = s["total_cycles"]
    smxu = sum(t["mxu_busy"] for t in tiles)
    sovl = sum(t["mxu_dma_overlap"] for t in tiles)
    dma_tiles = [t for t in tiles if t["dma_busy"] > 0]
    dbw = (100.0 * sum(t["dma_busy"] for t in dma_tiles) /
           (len(dma_tiles) * cyc)) if dma_tiles else 0.0
    return cyc, (100.0 * sovl / smxu if smxu else 0.0), dbw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../build/src/trace_sim")
    ap.add_argument("--out", default="/tmp/edgc_gemm")
    ap.add_argument("--arch", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--int", action="store_true",
                    help="precision sweep (f16/i8/i4) instead of the K sweep")
    args = ap.parse_args()
    arch = load_arch(args.arch)
    ram = resolve_ramulator(arch)
    d = next(x for x in arch.dse if x["label"] == arch.default_sched)
    sched = Sched(label=d["label"], tile_link=d.get("tile_link", 1),
                  wr_ports=d.get("wr_ports", 1), rd_ports=d.get("rd_ports", 2),
                  dma_rate=d.get("dma_rate", 1),
                  dram_density=d.get("dram_density", 2048), nmc=d.get("nmc", False))

    M = N = 128
    rows = []
    if args.int:
        print(f"{'prec':>5}{'K':>6} | {'cycles':>7} {'mxu_ops':>8} {'dram_rd_KB':>11} "
              f"{'vs_f16':>7} {'overlap%':>8} {'dmaBW%':>7}")
        for K in (256, 512, 1024):
            f16c = None
            for prec in ("f16", "i8", "i4"):
                base = build_gemm(M, K, N, args.out, sched, ram, dbuf=True, prec=prec)
                s = run(args.sim, base, ram, args.out)
                cyc, ovl, dbw = metrics(s)
                if prec == "f16":
                    f16c = cyc
                print(f"{prec:>5}{K:>6} | {cyc:>7} {s['mxu_ops']:>8} "
                      f"{s['dram_read_bytes']/1024:>10.1f} {f16c/cyc:>6.2f}x "
                      f"{ovl:>7.1f} {dbw:>7.1f}")
                rows.append(dict(prec=prec, M=M, K=K, N=N, cycles=cyc,
                                 mxu_ops=s["mxu_ops"],
                                 dram_read_kb=round(s["dram_read_bytes"]/1024, 1),
                                 speedup_vs_f16=round(f16c/cyc, 4),
                                 overlap_pct=round(ovl, 2), dma_bw_pct=round(dbw, 2)))
        if args.csv:
            import csv
            with open(args.csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader(); w.writerows(rows)
        return
    print(f"{'M':>4}{'K':>6}{'N':>5} | {'cyc_sb':>7} {'cyc_db':>7} {'speedup':>7} | "
          f"{'ovl_sb%':>7} {'ovl_db%':>7} | {'dmaBW_sb%':>9} {'dmaBW_db%':>9}")
    for K in (128, 256, 512, 1024):
        base_sb = build_gemm(M, K, N, args.out, sched, ram, dbuf=False)
        base_db = build_gemm(M, K, N, args.out, sched, ram, dbuf=True)
        c_sb, o_sb, b_sb = metrics(run(args.sim, base_sb, ram, args.out))
        c_db, o_db, b_db = metrics(run(args.sim, base_db, ram, args.out))
        print(f"{M:>4}{K:>6}{N:>5} | {c_sb:>7} {c_db:>7} {c_sb/c_db:>6.3f}x | "
              f"{o_sb:>6.1f} {o_db:>6.1f} | {b_sb:>8.1f} {b_db:>8.1f}")
        rows.append(dict(M=M, K=K, N=N, cyc_sb=c_sb, cyc_db=c_db,
                         speedup=round(c_sb / c_db, 4),
                         overlap_sb_pct=round(o_sb, 2), overlap_db_pct=round(o_db, 2),
                         dma_bw_sb_pct=round(b_sb, 2), dma_bw_db_pct=round(b_db, 2)))
    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
