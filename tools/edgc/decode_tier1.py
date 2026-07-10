#!/usr/bin/env python3
"""Tier-1 real-model-structure DECODE-phase timing.

Drives the cycle-accurate simulator with the *real* tensor shapes of edge
LLMs (Llama-3.2-1B/3B, Phi-3.5-mini, Mistral-7B-INT4, Mixtral) in the decode
regime (one generated token against a KV cache of length L). Timing only:
random/zero data, shape-driven.

Tractability (as specified): every transformer layer of a given model is
shape-identical, so each *unique* GEMM shape is simulated once and the
per-layer cost is composed and multiplied by n_layers (layer-sampling
extrapolation). Attention is emitted as real per-head QK^T / PV GEMMs.

  python3 decode_tier1.py --sim ../../build/src/trace_sim
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_gemm as bg
from edgc.arch import load_arch, resolve_ramulator
from edgc import Sched

CLOCK_HZ = 500e6  # base-die logic clock (ARCH_SPEC reference)


# Real network structures (decoder-only). d_head, d_model, d_ffn, heads,
# kv_heads, layers from each model's published config. prec = native edge
# deployment precision. moe: (n_experts, top_k) or None.
MODELS = {
    "Llama-3.2-1B":   dict(layers=16, d_model=2048, heads=32, kv_heads=8,
                           d_head=64, d_ffn=8192, ffn="swiglu", prec="f16",
                           moe=None),
    "Llama-3.2-3B":   dict(layers=28, d_model=3072, heads=24, kv_heads=8,
                           d_head=128, d_ffn=8192, ffn="swiglu", prec="f16",
                           moe=None),
    "Phi-3.5-mini":   dict(layers=32, d_model=3072, heads=32, kv_heads=32,
                           d_head=96, d_ffn=8192, ffn="swiglu", prec="i8",
                           moe=None),
    "Mistral-7B-INT4":dict(layers=32, d_model=4096, heads=32, kv_heads=8,
                           d_head=128, d_ffn=14336, ffn="swiglu", prec="i4",
                           moe=None),
    "Mixtral-8x7B":   dict(layers=32, d_model=4096, heads=32, kv_heads=8,
                           d_head=128, d_ffn=14336, ffn="swiglu", prec="f16",
                           moe=(8, 2)),
}


def rup(x, m):
    return ((x + m - 1) // m) * m


def layer_gemms(cfg, M, L):
    """(name, M, K, N, mult, prec) GEMMs for one decode layer."""
    dm, dh = cfg["d_model"], cfg["d_head"]
    qn, kn = cfg["heads"] * dh, cfg["kv_heads"] * dh
    p = cfg["prec"]
    g = [
        ("q_proj",  M, dm, qn, 1, p),
        ("kv_proj", M, dm, kn, 2, p),                # K and V, same shape
        ("qk^T",    M, dh, L, cfg["heads"], p),      # per-head scores
        ("p@v",     M, L, dh, cfg["heads"], p),      # per-head context
        ("o_proj",  M, qn, dm, 1, p),
    ]
    df = cfg["d_ffn"]
    fmult = cfg["moe"][1] if cfg["moe"] else 1        # active experts (top_k)
    g += [("ffn_gate", M, dm, df, fmult, p),
          ("ffn_up",   M, dm, df, fmult, p),
          ("ffn_down", M, df, dm, fmult, p)]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../build/src/trace_sim")
    ap.add_argument("--out", default="/tmp/edgc_tier1")
    ap.add_argument("--arch", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--models", default="")
    args = ap.parse_args()
    arch = load_arch(args.arch)
    ram = resolve_ramulator(arch)
    d = next(x for x in arch.dse if x["label"] == arch.default_sched)
    sched = Sched(label=d["label"], tile_link=d["tile_link"], wr_ports=d["wr_ports"],
                  rd_ports=d["rd_ports"], dma_rate=d["dma_rate"],
                  dram_density=d["dram_density"], nmc=d["nmc"])

    M = max(16, rup(args.batch, 16))       # token block (>=1 16-row block)
    L = rup(args.ctx, 64)                  # KV length, 64-aligned (INT4 needs K%64)
    cache = {}

    def sim_shape(Mx, K, N, prec):
        key = (Mx, K, N, prec)
        if key not in cache:
            base = bg.build_gemm(Mx, K, N, args.out, sched, ram, dbuf=True, prec=prec)
            s = bg.run(args.sim, base, ram, args.out)
            cache[key] = dict(cyc=s["total_cycles"],
                              mxu_busy=sum(t["mxu_busy"] for t in s["tiles"]),
                              mxu_tiles=sum(1 for t in s["tiles"] if t["mxu_busy"] > 0),
                              dram_rd=s["dram_read_bytes"],
                              causal=s["causality_ok"])
        return cache[key]

    names = args.models.split(",") if args.models else list(MODELS)
    rows = []
    print(f"decode  batch={args.batch} (M={M})  ctx={args.ctx} (L={L})")
    print(f"{'model':17}{'prec':>5}{'lyr':>4} | {'cyc/layer':>10}{'cyc_total':>11} "
          f"{'tok/s':>8} {'MXU_util%':>10} {'mxu_tiles':>10} {'DRAM_GB':>8}")
    for nm in names:
        cfg = MODELS[nm]
        gs = layer_gemms(cfg, M, L)
        cyc_layer = mxu_busy_layer = dram_layer = 0
        maxtiles = 0
        for (_, Mx, K, N, mult, prec) in gs:
            r = sim_shape(Mx, K, N, prec)
            cyc_layer += mult * r["cyc"]
            mxu_busy_layer += mult * r["mxu_busy"]
            dram_layer += mult * r["dram_rd"]
            maxtiles = max(maxtiles, r["mxu_tiles"])
        nl = cfg["layers"]
        cyc_total = cyc_layer * nl
        util = 100.0 * mxu_busy_layer / (16 * cyc_layer)
        # One M-block (16 rows) carries min(batch,16) real tokens; the rest
        # are padding. Throughput scales with the real tokens in flight, so
        # batching up to 16 fills the decode M-padding for free.
        toks = args.batch * CLOCK_HZ / cyc_total
        dram_gb = dram_layer * nl / 1e9
        print(f"{nm:17}{cfg['prec']:>5}{nl:>4} | {cyc_layer:>10}{cyc_total:>11} "
              f"{toks:>8.2f} {util:>10.2f} {maxtiles:>10} {dram_gb:>8.2f}")
        rows.append(dict(model=nm, prec=cfg["prec"], layers=nl, batch=args.batch,
                         ctx=args.ctx, cyc_per_layer=cyc_layer, cyc_total=cyc_total,
                         tokens_per_s=round(toks, 3), mxu_util_pct=round(util, 3),
                         max_mxu_tiles=maxtiles, dram_read_gb=round(dram_gb, 3)))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main()
