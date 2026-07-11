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


def layer_gemms(cfg, M, L, moe_frag=False, batch=None):
    """(name, M, K, N, mult, prec, kv) GEMMs for one decode layer.
    kv=True marks attention GEMMs whose B operand is the KV cache — the
    candidate for steady-state SHARED (buffer die) residency.

    moe_frag models MoE token fragmentation instead of the default
    whole-block top-k view: with `batch` real tokens routed top_k ways over
    n_experts, each active expert sees only ~batch*top_k/n_experts tokens,
    padded up to a 16-row M block — the M dimension fragments exactly when
    batching was supposed to fill it."""
    dm, dh = cfg["d_model"], cfg["d_head"]
    qn, kn = cfg["heads"] * dh, cfg["kv_heads"] * dh
    p = cfg["prec"]
    g = [
        ("q_proj",  M, dm, qn, 1, p, False),
        ("kv_proj", M, dm, kn, 2, p, False),            # K and V, same shape
        ("qk^T",    M, dh, L, cfg["heads"], p, True),   # per-head scores
        ("p@v",     M, L, dh, cfg["heads"], p, True),   # per-head context
        ("o_proj",  M, qn, dm, 1, p, False),
    ]
    df = cfg["d_ffn"]
    if cfg["moe"] and moe_frag:
        n_exp, top_k = cfg["moe"]
        toks = (batch if batch else M) * top_k
        active = min(n_exp, toks)              # experts actually hit
        m_e = max(16, rup((toks + active - 1) // active, 16))
        g += [("ffn_gate", m_e, dm, df, active, p, False),
              ("ffn_up",   m_e, dm, df, active, p, False),
              ("ffn_down", m_e, df, dm, active, p, False)]
        return g
    fmult = cfg["moe"][1] if cfg["moe"] else 1        # active experts (top_k)
    g += [("ffn_gate", M, dm, df, fmult, p, False),
          ("ffn_up",   M, dm, df, fmult, p, False),
          ("ffn_down", M, df, dm, fmult, p, False)]
    return g


def kv_cache_bytes(cfg, L):
    """Whole-model KV cache footprint at the GEMM element width."""
    eb = bg.EB[cfg["prec"]]
    return int(cfg["layers"] * 2 * L * cfg["kv_heads"] * cfg["d_head"] * eb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../build/src/trace_sim")
    ap.add_argument("--out", default="/tmp/edgc_tier1")
    ap.add_argument("--arch", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--models", default="")
    ap.add_argument("--W", type=int, default=0,
                    help="consumption-port width knob: overrides tile_link=W,"
                         " wr_ports=W, rd_ports=2W, dma_rate=W")
    ap.add_argument("--dram-density", type=int, default=0,
                    help="vertical bond density (bits/column/cycle) override")
    ap.add_argument("--kv", choices=("auto", "dram", "shared"), default="dram",
                    help="KV-cache placement: shared = steady-state resident "
                         "in the buffer-die banks; auto = shared iff the whole"
                         " KV cache fits num_banks*shared_mb")
    ap.add_argument("--moe-frag", action="store_true",
                    help="model MoE token fragmentation: each active expert "
                         "sees batch*top_k/n_experts tokens (padded M block) "
                         "instead of the whole batch")
    ap.add_argument("--json", default="", help="write results as JSON here")
    args = ap.parse_args()
    arch = load_arch(args.arch)
    ram = resolve_ramulator(arch)
    d = next(x for x in arch.dse if x["label"] == arch.default_sched)
    sched = Sched(label=d["label"], tile_link=d["tile_link"], wr_ports=d["wr_ports"],
                  rd_ports=d["rd_ports"], dma_rate=d["dma_rate"],
                  dram_density=d["dram_density"], nmc=d["nmc"])
    if args.W:
        sched = Sched(label=f"W{args.W}", tile_link=args.W, wr_ports=args.W,
                      rd_ports=2 * args.W, dma_rate=args.W,
                      dram_density=sched.dram_density, nmc=sched.nmc)
    if args.dram_density:
        sched.dram_density = args.dram_density

    M = max(16, rup(args.batch, 16))       # token block (>=1 16-row block)
    L = rup(args.ctx, 64)                  # KV length, 64-aligned (INT4 needs K%64)
    shared_cap = arch.num_banks * arch.shared_bytes
    cache = {}

    def sim_shape(Mx, K, N, prec, b_shared=False):
        key = (Mx, K, N, prec, b_shared)
        if key not in cache:
            base = bg.build_gemm(Mx, K, N, args.out, sched, ram, dbuf=True,
                                 prec=prec, b_shared=b_shared)
            s = bg.run(args.sim, base, ram, args.out)
            cache[key] = dict(cyc=s["total_cycles"],
                              mxu_busy=sum(t["mxu_busy"] for t in s["tiles"]),
                              mxu_tiles=sum(1 for t in s["tiles"] if t["mxu_busy"] > 0),
                              dram_rd=s["dram_read_bytes"],
                              causal=s["causality_ok"])
        return cache[key]

    names = args.models.split(",") if args.models else list(MODELS)
    rows = []
    print(f"decode  batch={args.batch} (M={M})  ctx={args.ctx} (L={L})  "
          f"W={sched.tile_link} density={sched.dram_density} kv={args.kv}")
    print(f"{'model':17}{'prec':>5}{'lyr':>4} | {'cyc/layer':>10}{'cyc_total':>11} "
          f"{'tok/s':>8} {'MXU_util%':>10} {'AI':>7} {'DRAM_GB':>8} {'kv':>7}")
    for nm in names:
        cfg = MODELS[nm]
        kv_bytes = kv_cache_bytes(cfg, L)
        kv_res = (args.kv == "shared"
                  or (args.kv == "auto" and kv_bytes <= shared_cap))
        gs = layer_gemms(cfg, M, L, moe_frag=args.moe_frag, batch=args.batch)
        cyc_layer = mxu_busy_layer = dram_layer = 0
        flops_layer = 0
        causal = True
        maxtiles = 0
        for (_, Mx, K, N, mult, prec, kv) in gs:
            r = sim_shape(Mx, K, N, prec, b_shared=kv and kv_res)
            cyc_layer += mult * r["cyc"]
            mxu_busy_layer += mult * r["mxu_busy"]
            dram_layer += mult * r["dram_rd"]
            flops_layer += mult * 2 * Mx * K * N
            causal = causal and r["causal"]
            maxtiles = max(maxtiles, r["mxu_tiles"])
        nl = cfg["layers"]
        cyc_total = cyc_layer * nl
        util = 100.0 * mxu_busy_layer / (arch.num_tiles * cyc_layer)
        # One M-block (16 rows) carries min(batch,16) real tokens; the rest
        # are padding. Throughput scales with the real tokens in flight, so
        # batching up to 16 fills the decode M-padding for free.
        toks = args.batch * CLOCK_HZ / cyc_total
        dram_gb = dram_layer * nl / 1e9
        ai = flops_layer / max(1, dram_layer)   # effective FLOP / DRAM byte
        print(f"{nm:17}{cfg['prec']:>5}{nl:>4} | {cyc_layer:>10}{cyc_total:>11} "
              f"{toks:>8.2f} {util:>10.2f} {ai:>7.2f} {dram_gb:>8.2f} "
              f"{'shared' if kv_res else 'dram':>7}")
        rows.append(dict(model=nm, prec=cfg["prec"], layers=nl, batch=args.batch,
                         ctx=args.ctx, cyc_per_layer=cyc_layer, cyc_total=cyc_total,
                         tokens_per_s=round(toks, 3), mxu_util_pct=round(util, 3),
                         max_mxu_tiles=maxtiles, dram_read_gb=round(dram_gb, 4),
                         flops_per_layer=flops_layer,
                         arith_intensity=round(ai, 4),
                         kv_resident=kv_res, kv_bytes=kv_bytes,
                         causality_ok=causal))
    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
