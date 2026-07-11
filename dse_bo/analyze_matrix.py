#!/usr/bin/env python3
"""Robustness analysis of the workload-matrix DSE.

Answers: does the single-corner (decode-only ctx512) recommendation survive
the full operating spectrum?

Outputs (results/figs/):
  mx_convergence.png   best-so-far weighted objective, BO vs random
  mx_pareto.png        decode_agg vs prefill_agg (Pareto set of the union
                       pool), area encoded as marker size
  mx_anchor_heat.png   per-anchor speedup heatmap of the top configs
  mx_rank_scatter.png  per-config decode rank vs prefill rank + Spearman
plus results/matrix_analysis.json with the tables cited in ARCH_SPEC.
"""
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(RES, "figs")
sys.path.insert(0, HERE)
from matrix import ANCHORS, BASELINE  # noqa: E402
from evaluate import KNOBS  # noqa: E402

BLUE, AQUA, GRAY, YELLOW, VIOLET = ("#2a78d6", "#1baf7a", "#9a9992",
                                    "#eda100", "#4a3aa7")
TXT, TXT2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9.5, "axes.edgecolor": "#d8d7d2",
    "axes.labelcolor": TXT, "text.color": TXT, "xtick.color": TXT2,
    "ytick.color": TXT2, "axes.grid": True, "grid.color": "#eceae5",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})

AIDS = [a["id"] for a in ANCHORS]


def pool():
    out = []
    for f in glob.glob(os.path.join(HERE, "cache_matrix", "agg_*.json")):
        d = json.load(open(f))
        if d.get("valid"):
            out.append(d)
    return out


def cfg_str(c):
    return (f"{c['num_tiles']}t/{c['tiles_per_group']}g/{c['shared_mb']}M/"
            f"{c['topology'][:1]}/W{c['W']}/D{c['dram_density']}")


def fig_convergence():
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    for pat, color, label in (("mx_bo_seed*", BLUE, "BO (qLogNEI, weighted)"),
                              ("mx_random_seed*", AQUA, "random search")):
        for d in sorted(glob.glob(os.path.join(RES, pat))):
            f = os.path.join(d, "log.jsonl")
            if not os.path.exists(f):
                continue
            recs = [json.loads(l) for l in open(f)]
            xs = np.arange(1, len(recs) + 1)
            ys = [r["best_weighted"] for r in recs]
            ax.plot(xs, ys, color=color, lw=1.8, label=label)
            label = None
            ax.annotate(f"{ys[-1]:.3f}", xy=(xs[-1], ys[-1]),
                        xytext=(3, 0), textcoords="offset points",
                        fontsize=8, color=color)
    ax.set_xlabel("evaluations")
    ax.set_ylabel("best weighted speedup vs baseline")
    ax.set_title("workload-matrix objective — BO vs random", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mx_convergence.png"))
    plt.close(fig)


def fig_pareto(P, marked=()):
    dec = np.array([p["objectives"]["decode_agg"] for p in P])
    pre = np.array([p["objectives"]["prefill_agg"] for p in P])
    area = np.array([p["objectives"]["area_mm2"] for p in P])
    # 2D pareto on (dec, pre) for the scatter; area via size
    order = np.argsort(-dec)
    front = []
    best_pre = -1
    for i in order:
        if pre[i] > best_pre:
            front.append(i)
            best_pre = pre[i]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(dec, pre, s=8 + 120 / area, color=GRAY, alpha=0.5, lw=0,
               label="evaluated (size ∝ 1/area)")
    fi = sorted(front, key=lambda i: dec[i])
    ax.plot(dec[fi], pre[fi], "-", color=BLUE, lw=1.2, alpha=0.6)
    ax.scatter(dec[fi], pre[fi], s=30, color=BLUE, zorder=3,
               label="decode/prefill Pareto")
    for i in fi:
        ax.annotate(cfg_str(P[i]["config"]), xy=(dec[i], pre[i]),
                    xytext=(4, 3), textcoords="offset points", fontsize=6.2,
                    color=TXT2)
    for lbl, c in marked:
        m = next((p for p in P if p["config"] == c), None)
        if m:
            ax.scatter([m["objectives"]["decode_agg"]],
                       [m["objectives"]["prefill_agg"]], marker="x", s=70,
                       color=VIOLET, zorder=4)
            ax.annotate(lbl, xy=(m["objectives"]["decode_agg"],
                                 m["objectives"]["prefill_agg"]),
                        xytext=(4, -10), textcoords="offset points",
                        fontsize=7, color=VIOLET, fontweight="bold")
    ax.axhline(1.0, color="#d8d7d2", lw=0.8, ls="--")
    ax.axvline(1.0, color="#d8d7d2", lw=0.8, ls="--")
    ax.set_xlabel("decode aggregate speedup (geomean of 9 anchors)")
    ax.set_ylabel("prefill aggregate speedup (geomean of 3 anchors)")
    ax.set_title("decode-optimal vs prefill-optimal architectures", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mx_pareto.png"))
    plt.close(fig)
    return [P[i] for i in fi]


def fig_anchor_heat(P, rows):
    """rows = [(label, cfgdict)]; heat of per-anchor speedups."""
    M = []
    labels = []
    for lbl, c in rows:
        m = next((p for p in P if p["config"] == c), None)
        if not m:
            continue
        M.append([m["anchors"][aid]["speedup"] for aid in AIDS])
        labels.append(f"{lbl}  {cfg_str(c)}")
    if not M:
        return
    M = np.array(M)
    fig, ax = plt.subplots(figsize=(8.6, 0.5 * len(M) + 1.8))
    im = ax.imshow(M, cmap="RdBu_r", vmin=2 - M.max(), vmax=M.max(),
                   aspect="auto")
    ax.set_xticks(range(len(AIDS)), AIDS)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.6,
                    color="#ffffff" if abs(M[i, j] - 1) > 0.25 else TXT)
    ax.set_title("per-anchor speedup vs baseline (WL-A … WL-L)", fontsize=10)
    ax.grid(False)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mx_anchor_heat.png"))
    plt.close(fig)


def fig_rank(P):
    from scipy.stats import spearmanr
    dec = np.array([p["objectives"]["decode_agg"] for p in P])
    pre = np.array([p["objectives"]["prefill_agg"] for p in P])
    rd = (-dec).argsort().argsort() + 1
    rp = (-pre).argsort().argsort() + 1
    rho, pv = spearmanr(dec, pre)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.scatter(rd, rp, s=16, color=BLUE, alpha=0.65, lw=0)
    lim = max(rd.max(), rp.max()) + 1
    ax.plot([0, lim], [0, lim], color="#d8d7d2", lw=0.8, ls="--")
    ax.set_xlabel("decode rank (1 = best)")
    ax.set_ylabel("prefill rank")
    ax.set_title(f"config ranking: decode vs prefill\n"
                 f"Spearman ρ = {rho:.3f} (p = {pv:.2g}, n = {len(P)})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "mx_rank_scatter.png"))
    plt.close(fig)
    return dict(spearman_rho=float(rho), p=float(pv), n=len(P))


def main():
    os.makedirs(FIGS, exist_ok=True)
    P = pool()
    if len(P) < 5:
        print(f"only {len(P)} matrix points — run campaigns first")
        return
    report = {"n_points": len(P)}

    fig_convergence()

    # top configs by each aggregate
    by = {}
    for key in ("weighted", "minmax", "decode_agg", "prefill_agg"):
        top = sorted(P, key=lambda p: -p["objectives"][key])[:5]
        by[key] = [dict(cfg=cfg_str(p["config"]),
                        **{k: round(p["objectives"][k], 4)
                           for k in ("weighted", "minmax", "decode_agg",
                                     "prefill_agg", "area_mm2")})
                   for p in top]
    report["top_by_objective"] = by

    # old single-corner recommendation (from the ctx512 4-objective DSE):
    # best decode_tput config in the old cache pool
    old = []
    for f in glob.glob(os.path.join(HERE, "cache", "*.json")):
        d = json.load(open(f))
        if d.get("valid") and d.get("ctx", 512) == 512:
            old.append(d)
    old_best = max(old, key=lambda d: d["objectives"]["decode_tput"])["config"] \
        if old else None
    report["old_corner_recommendation"] = old_best

    marked = [("baseline", BASELINE)]
    if old_best:
        marked.append(("old-DSE best", old_best))
    front = fig_pareto(P, marked)
    report["decode_prefill_front"] = [cfg_str(p["config"]) for p in front]

    rows = [("baseline", BASELINE)]
    if old_best:
        rows.append(("old-DSE best (decode ctx512)", old_best))
    for key in ("weighted", "minmax", "decode_agg", "prefill_agg"):
        best = max(P, key=lambda p: p["objectives"][key])
        rows.append((f"best {key}", best["config"]))
    # dedupe rows by config
    seen, uniq = set(), []
    for lbl, c in rows:
        k = json.dumps(c, sort_keys=True)
        if k not in seen:
            seen.add(k)
            uniq.append((lbl, c))
    fig_anchor_heat(P, uniq)
    report["rank_correlation"] = fig_rank(P)

    # robustness verdict: old recommendation's anchor table
    if old_best:
        m = next((p for p in P if p["config"] == old_best), None)
        if m:
            report["old_best_on_matrix"] = {
                aid: round(m["anchors"][aid]["speedup"], 4) for aid in AIDS}
            report["old_best_objectives"] = {
                k: round(v, 4) for k, v in m["objectives"].items()}

    json.dump(report, open(os.path.join(RES, "matrix_analysis.json"), "w"),
              indent=1)
    print(json.dumps(report, indent=1)[:3500])


if __name__ == "__main__":
    main()
