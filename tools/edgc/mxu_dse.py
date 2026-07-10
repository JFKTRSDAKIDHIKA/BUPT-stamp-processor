#!/usr/bin/env python3
"""P3 — MXU size analytical DSE.

Pen-and-paper (no simulator hardware change): for every GEMM shape in the
Tier-1 models, count how many S x S x S MXU tile-passes it takes and what
fraction of the issued MAC slots carry real (non-padding) work, for
MXU sizes S in {8, 16, 32}. The point: decode runs at M = batch tokens, so
the M dimension is tiny and boundary padding dominates -> utilization is
governed by min(M, S)/S. Larger MXUs need proportionally larger batch to
stay full.

  python3 mxu_dse.py [--csv out.csv] [--plot fig.png]

util(shape,S) = (M*N*K useful MACs) / (passes * S^3 peak MACs)
passes(shape,S) = ceil(M/S)*ceil(N/S)*ceil(K/S)
"""
import argparse
import csv
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decode_tier1 import MODELS, layer_gemms, rup

SIZES = (8, 16, 32)


def passes(M, K, N, S):
    return math.ceil(M / S) * math.ceil(N / S) * math.ceil(K / S)


def analyze_layer(cfg, M, L):
    """Aggregate useful MACs, tile-passes and utilization over one decode
    layer's GEMMs for each MXU size."""
    gs = layer_gemms(cfg, M, L)
    useful = 0
    per_S = {S: dict(passes=0, peak=0) for S in SIZES}
    for (_, Mx, K, N, mult, _prec) in gs:
        useful += mult * Mx * K * N
        for S in SIZES:
            p = mult * passes(Mx, K, N, S)
            per_S[S]["passes"] += p
            per_S[S]["peak"] += p * S ** 3
    out = {}
    for S in SIZES:
        peak = per_S[S]["peak"]
        out[S] = dict(passes=per_S[S]["passes"],
                      util=100.0 * useful / peak if peak else 0.0)
    return useful, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="")
    ap.add_argument("--plot", default="")
    ap.add_argument("--ctx", type=int, default=512)
    args = ap.parse_args()
    L = rup(args.ctx, 64)

    rows = []
    # Decode batch sensitivity: M = 1 (single token) .. 16 .. 32.
    print("MXU-size DSE — decode-phase MXU utilization (%) by array size")
    print(f"(ctx L={L}; util = useful MACs / issued MAC slots, weighted over "
          f"a layer's GEMMs)\n")
    for M in (1, 16, 32):
        print(f"── batch M={M} ─────────────────────────────────────────")
        print(f"{'model':17} | {'util@8x8x8':>11} {'util@16^3':>10} {'util@32^3':>10}"
              f" | {'passes@16^3':>12}")
        for nm, cfg in MODELS.items():
            _, per_S = analyze_layer(cfg, M, L)
            print(f"{nm:17} | {per_S[8]['util']:>10.1f}% {per_S[16]['util']:>9.1f}%"
                  f" {per_S[32]['util']:>9.1f}% | {per_S[16]['passes']:>12,}")
            rows.append(dict(model=nm, batch_M=M, ctx=L,
                             util_8=round(per_S[8]["util"], 2),
                             util_16=round(per_S[16]["util"], 2),
                             util_32=round(per_S[32]["util"], 2),
                             passes_8=per_S[8]["passes"],
                             passes_16=per_S[16]["passes"],
                             passes_32=per_S[32]["passes"]))
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"wrote {args.csv}")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f"[plot skipped: {e}]"); return
        Ms = [1, 2, 4, 8, 16, 32, 64]
        fig, axes = plt.subplots(1, len(MODELS), figsize=(4 * len(MODELS), 3.4),
                                 sharey=True)
        for ax, (nm, cfg) in zip(axes, MODELS.items()):
            for S in SIZES:
                ys = [analyze_layer(cfg, M, L)[1][S]["util"] for M in Ms]
                ax.plot(Ms, ys, marker="o", label=f"{S}x{S}x{S}")
            ax.set_xscale("log", base=2); ax.set_xticks(Ms)
            ax.set_xticklabels(Ms); ax.set_title(nm, fontsize=9)
            ax.set_xlabel("batch M (tokens)"); ax.grid(alpha=0.3)
        axes[0].set_ylabel("MXU utilization (%)")
        axes[0].legend(fontsize=8, title="MXU size")
        fig.suptitle("Decode MXU utilization vs. batch, by MXU array size", y=1.02)
        fig.tight_layout(); fig.savefig(args.plot, bbox_inches="tight", dpi=130)
        print(f"wrote {args.plot}")


if __name__ == "__main__":
    main()
