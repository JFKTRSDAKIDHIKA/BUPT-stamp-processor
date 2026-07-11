#!/usr/bin/env python3
"""Analysis + figures for the DSE campaign.

Inputs : results/{bo_seed*,rand_seed*}/log.jsonl, cache/*.json
Outputs: results/figs/*.png, results/analysis.md (numbers cited in the report)

Figures follow the dataviz method: BO = blue #2a78d6, random = aqua #1baf7a
(validated pair; aqua carries direct labels), thin marks, single axis,
recessive grid.
"""
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIGS = os.path.join(RES, "figs")
sys.path.insert(0, HERE)
from bo_engine import OBJECTIVES, hv_of  # noqa: E402

BLUE, AQUA, GRAY = "#2a78d6", "#1baf7a", "#9a9992"
YELLOW, VIOLET, RED = "#eda100", "#4a3aa7", "#e34948"
TXT, TXT2 = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 9.5, "axes.edgecolor": "#d8d7d2",
    "axes.labelcolor": TXT, "text.color": TXT, "xtick.color": TXT2,
    "ytick.color": TXT2, "axes.grid": True, "grid.color": "#eceae5",
    "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
})


def load_runs(pattern):
    runs = {}
    for d in sorted(glob.glob(os.path.join(RES, pattern))):
        f = os.path.join(d, "log.jsonl")
        if os.path.exists(f):
            runs[os.path.basename(d)] = [json.loads(l) for l in open(f)]
    return runs


def load_cache():
    out = []
    for f in glob.glob(os.path.join(HERE, "cache", "*.json")):
        out.append(json.load(open(f)))
    return out


# ── 1. Convergence: hypervolume vs evaluations ────────────────
def fig_convergence(bo_runs, rand_runs):
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    stats = {}
    for runs, color, label in ((bo_runs, BLUE, "BO (Ax/BoTorch qNEHVI)"),
                               (rand_runs, AQUA, "random search")):
        if not runs:
            continue
        curves = [[r["hypervolume"] for r in v] for v in runs.values()]
        n = min(len(c) for c in curves)
        M = np.array([c[:n] for c in curves])
        x = np.arange(1, n + 1)
        for c in M:
            ax.plot(x, c, color=color, lw=0.8, alpha=0.35)
        ax.plot(x, M.mean(0), color=color, lw=2.0, label=label)
        stats[label] = M
        ax.annotate(label, xy=(x[-1], M.mean(0)[-1]),
                    xytext=(-4, 8 if color == BLUE else -14),
                    textcoords="offset points", ha="right",
                    color=color, fontsize=9, fontweight="bold")
    nb = len(next(iter(bo_runs.values()))) if bo_runs else 0
    ax.axvline(12.5, color="#d8d7d2", lw=1, ls="--")
    ax.text(12.8, ax.get_ylim()[0], "Sobol → GP+qNEHVI", fontsize=8,
            color=TXT2, va="bottom")
    ax.set_xlabel("evaluations")
    ax.set_ylabel("dominated hypervolume (4 objectives)")
    ax.set_title("BO vs same-budget random search — convergence", fontsize=11)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "convergence_hv.png"))
    plt.close(fig)
    return stats


def convergence_stats(stats):
    """Final-HV comparison + sample-efficiency ratio + Mann-Whitney U."""
    out = {}
    if len(stats) < 2:
        return out
    (lb, B), (lr, R) = stats.items()
    bo_final = B[:, -1]
    rd_final = R[:, -1]
    out["bo_final_hv"] = bo_final.tolist()
    out["rand_final_hv"] = rd_final.tolist()
    # evaluations BO needs to reach random's best final HV
    target = rd_final.max()
    reach = [int(np.argmax(b >= target)) + 1 if (b >= target).any() else -1
             for b in B]
    out["bo_evals_to_beat_random_best"] = reach
    try:
        from scipy.stats import mannwhitneyu
        u, p = mannwhitneyu(bo_final, rd_final, alternative="greater")
        out["mannwhitney_U"] = float(u)
        out["mannwhitney_p_onesided"] = float(p)
    except Exception as e:
        out["stat_error"] = str(e)
    return out


# ── 2. Pareto front ───────────────────────────────────────────
def pareto_mask(pts, keys, sense):
    P = np.array([[p[k] * (-1 if sense[k] else 1) for k in keys] for p in pts])
    n = len(P)
    mask = np.ones(n, bool)
    for i in range(n):
        if not mask[i]:
            continue
        dom = ((P >= P[i]).all(1) & (P > P[i]).any(1))
        if dom.any():
            mask[i] = False
    return mask


def fig_pareto(cache):
    valid = [c for c in cache if c.get("valid") and c.get("ctx", 512) == 512]
    pts = [dict(c["objectives"], **c["config"]) for c in valid]
    if not pts:
        return []
    sense = {k: v["minimize"] for k, v in OBJECTIVES.items()}
    mask = pareto_mask(pts, list(OBJECTIVES), sense)
    front = sorted((p for p, m in zip(pts, mask) if m),
                   key=lambda p: p["area_mm2"])

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter([p["area_mm2"] for p in pts],
               [p["decode_tput"] for p in pts],
               s=14, color=GRAY, alpha=0.55, lw=0, label="evaluated")
    fx = [p["area_mm2"] for p in front]
    fy = [p["decode_tput"] for p in front]
    ax.scatter(fx, fy, s=34, color=BLUE, zorder=3, lw=0,
               label="4-objective Pareto set")
    for p in front:
        ax.annotate(f"{p['num_tiles']}t/{p['tiles_per_group']}g/"
                    f"{p['shared_mb']}M/{p['topology'][:1]}/W{p['W']}/"
                    f"D{p['dram_density']}",
                    xy=(p["area_mm2"], p["decode_tput"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6.3, color=TXT2)
    ax.set_xlabel("area proxy (mm², base+buffer die)")
    ax.set_ylabel("decode throughput (geomean tok/s)")
    ax.set_title("Pareto set projected on throughput vs area", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "pareto_tput_area.png"))
    plt.close(fig)
    return front


# ── 3. Sensitivity: refit an ARD GP on ALL observations ───────
KNOB_ORDER = ["num_tiles", "tiles_per_group", "shared_mb", "W",
              "dram_density"]
TOPOS = ["ring", "mesh", "torus"]


def encode_X(cfgs):
    """Ordinal knobs log2-normalized to [0,1]; topology one-hot. 8 dims."""
    lo_hi = {k: (math.log2(min(v)), math.log2(max(v)))
             for k, v in (("num_tiles", (8, 32)), ("tiles_per_group", (2, 8)),
                          ("shared_mb", (4, 16)), ("W", (1, 4)),
                          ("dram_density", (512, 8192)))}
    rows = []
    for c in cfgs:
        r = [(math.log2(c[k]) - lo_hi[k][0]) / (lo_hi[k][1] - lo_hi[k][0])
             for k in KNOB_ORDER]
        r += [1.0 if c["topology"] == t else 0.0 for t in TOPOS]
        rows.append(r)
    return np.array(rows)


def fig_sensitivity(cache):
    """ARD lengthscale importance + LOO-R2 per objective, from a fresh
    SingleTaskGP fit on every valid ctx-512 observation (§0.2: surrogate
    quality evidence independent of the in-loop logs)."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood

    valid = [c for c in cache if c.get("valid") and c.get("ctx", 512) == 512]
    if len(valid) < 15:
        return None
    X = torch.tensor(encode_X([c["config"] for c in valid]), dtype=torch.double)
    dims = KNOB_ORDER + [f"topo:{t}" for t in TOPOS]
    out = {}
    imp_by_obj = {}
    for obj in OBJECTIVES:
        y = torch.tensor([[c["objectives"][obj]] for c in valid],
                         dtype=torch.double)
        ystd = (y - y.mean()) / (y.std() + 1e-9)
        gp = SingleTaskGP(X, ystd)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)
        with torch.no_grad():
            lml = mll(gp(*gp.train_inputs), gp.train_targets).item()
            # closed-form LOO residuals (Rasmussen & Williams eq. 5.12)
            K = gp.covar_module(X).to_dense() \
                + gp.likelihood.noise * torch.eye(len(X), dtype=torch.double)
            Ki = torch.linalg.inv(K)
            yv = ystd.squeeze(-1)
            mu_loo = yv - (Ki @ yv) / Ki.diagonal()
            ss_res = ((yv - mu_loo) ** 2).sum()
            ss_tot = ((yv - yv.mean()) ** 2).sum()
            loo_r2 = float(1 - ss_res / ss_tot)
        ker = getattr(gp.covar_module, "base_kernel", gp.covar_module)
        ls = ker.lengthscale.detach().squeeze().numpy()
        imp = 1.0 / np.maximum(ls, 1e-6)
        imp_by_obj[obj] = imp / imp.sum()
        out[obj] = dict(loo_r2=round(loo_r2, 3), lml=round(lml, 2),
                        lengthscales={d: round(float(l), 3)
                                      for d, l in zip(dims, ls)})
    # aggregate: topology = max of its one-hot dims, then mean over objectives
    agg = {}
    for k in KNOB_ORDER:
        agg[k] = float(np.mean([imp_by_obj[o][dims.index(k)]
                                for o in OBJECTIVES]))
    agg["topology"] = float(np.mean([
        max(imp_by_obj[o][dims.index(f"topo:{t}")] for t in TOPOS)
        for o in OBJECTIVES]))
    items = sorted(agg.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(5.8, 3.2))
    ax.barh([k for k, _ in items], [v for _, v in items], color=BLUE,
            height=0.55)
    for y, (k, v) in enumerate(items):
        ax.text(v, y, f" {v:.2f}", va="center", fontsize=8, color=TXT2)
    ax.set_xlabel("ARD importance (normalized 1/lengthscale, mean over objectives)")
    ax.set_title(f"Knob sensitivity — GP refit on {len(valid)} observations",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "sensitivity_ard.png"))
    plt.close(fig)
    return dict(importance=agg, per_objective=out, n=len(valid))


# ── 4. Insight figures from the cache grids ───────────────────
def _cfg_get(cache, **want):
    ctx = want.pop("ctx", 512)
    for c in cache:
        if not c.get("valid") or c.get("ctx", 512) != ctx:
            continue
        if all(c["config"][k] == v for k, v in want.items()):
            return c
    return None


def fig_q1_density(cache):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=False)
    dens = (512, 1024, 2048, 4096, 8192)
    colors = {8: AQUA, 16: BLUE, 32: VIOLET}
    for nt in (8, 16, 32):
        ys = []
        for d in dens:
            c = _cfg_get(cache, num_tiles=nt, tiles_per_group=4, shared_mb=8,
                         topology="ring", W=1, dram_density=d)
            ys.append(c["metrics"]["llama_b16"]["tokens_per_s"] if c else np.nan)
        axes[0].plot(dens, ys, "o-", ms=4, lw=1.6, color=colors[nt],
                     label=f"{nt} tiles")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(dens, [str(d) for d in dens])
    axes[0].set_xlabel("bond density (bits/column/cycle)")
    axes[0].set_ylabel("Llama-1B b16 tok/s")
    axes[0].set_title("density knee vs num_tiles (tpg=4)", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    colors2 = {2: YELLOW, 4: BLUE, 8: RED}
    for tpg in (2, 4, 8):
        ys = []
        for d in dens:
            c = _cfg_get(cache, num_tiles=16, tiles_per_group=tpg, shared_mb=8,
                         topology="ring", W=1, dram_density=d)
            ys.append(c["metrics"]["llama_b16"]["tokens_per_s"] if c else np.nan)
        axes[1].plot(dens, ys, "o-", ms=4, lw=1.6, color=colors2[tpg],
                     label=f"tpg={tpg} ({16 // tpg} columns)")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(dens, [str(d) for d in dens])
    axes[1].set_xlabel("bond density (bits/column/cycle)")
    axes[1].set_title("density knee vs tiles_per_group (16 tiles)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Q1 — where the 2048 bits/col knee moves", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "q1_density_knee.png"))
    plt.close(fig)


def fig_q2_kv(cache):
    pts = defaultdict(list)  # model -> (capacityMB, AI, tok/s)
    for c in cache:
        if not c.get("valid") or c.get("ctx", 512) != 2048:
            continue
        cfg = c["config"]
        cap = cfg["num_tiles"] // cfg["tiles_per_group"] * cfg["shared_mb"]
        for tag, lbl in (("llama_b16", "Llama-1B f16 (KV 67 MB)"),
                         ("mistral_b16", "Mistral-7B INT4 (KV 33.5 MB)")):
            m = c["metrics"][tag]
            pts[lbl].append((cap, m["arith_intensity"], m["tokens_per_s"],
                             m["kv_resident"]))
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    for lbl, color in ((list(pts)[0], BLUE), (list(pts)[1], YELLOW)):
        arr = sorted(pts[lbl])
        caps = sorted({a[0] for a in arr})
        ai = [np.mean([x[1] for x in arr if x[0] == c]) for c in caps]
        tk = [np.mean([x[2] for x in arr if x[0] == c]) for c in caps]
        axes[0].plot(caps, ai, "o-", ms=4, lw=1.6, color=color, label=lbl)
        axes[1].plot(caps, tk, "o-", ms=4, lw=1.6, color=color, label=lbl)
    for ax, yl in ((axes[0], "arithmetic intensity (FLOP/DRAM byte)"),
                   (axes[1], "tok/s (batch 16, ctx 2048)")):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("total shared SRAM = num_banks × shared_mb (MB)")
        ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Q2 — KV-cache residency knee vs shared capacity (ctx 2048)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "q2_kv_knee.png"))
    plt.close(fig)


def fig_q3_q4(cache):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    colors = {1: AQUA, 2: BLUE, 4: VIOLET}
    for w in (1, 2, 4):
        xs, ys, us = [], [], []
        for nt in (8, 16, 32):
            c = _cfg_get(cache, num_tiles=nt, tiles_per_group=4, shared_mb=8,
                         topology="ring", W=w, dram_density=2048)
            if c:
                xs.append(nt)
                ys.append(c["metrics"]["llama_b16"]["tokens_per_s"])
                us.append(c["metrics"]["llama_b16"]["mxu_util_pct"])
        axes[0].plot(xs, ys, "o-", ms=4, lw=1.6, color=colors[w], label=f"W={w}")
    axes[0].set_xticks((8, 16, 32))
    axes[0].set_xlabel("num_tiles")
    axes[0].set_ylabel("Llama-1B b16 tok/s")
    axes[0].set_title("Q3 — do more tiles get fed? (tok/s)", fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)

    for w in (1, 2, 4):
        xs, ys = [], []
        for tpg in (2, 4, 8):
            c = _cfg_get(cache, num_tiles=16, tiles_per_group=tpg, shared_mb=8,
                         topology="ring", W=w, dram_density=2048)
            if c:
                xs.append(tpg)
                ys.append(c["metrics"]["llama_b16"]["tokens_per_s"])
        axes[1].plot(xs, ys, "o-", ms=4, lw=1.6, color=colors[w], label=f"W={w}")
    axes[1].set_xticks((2, 4, 8))
    axes[1].set_xlabel("tiles_per_group (bank:tile = 1:tpg)")
    axes[1].set_title("Q4 — column sharing (16 tiles)", fontsize=10)
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "q3_q4.png"))
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    bo_runs = load_runs("bo_seed*")
    rand_runs = load_runs("rand_seed*")
    cache = load_cache()
    report = {}

    stats = fig_convergence(bo_runs, rand_runs)
    report["convergence"] = convergence_stats(stats)
    front = fig_pareto(cache)
    report["pareto_front"] = [
        {k: p[k] for k in ("num_tiles", "tiles_per_group", "shared_mb",
                           "topology", "W", "dram_density", "decode_tput",
                           "mxu_util", "arith_intensity", "area_mm2")}
        for p in front]
    report["sensitivity_ard"] = fig_sensitivity(cache)
    try:
        fig_q1_density(cache)
        fig_q2_kv(cache)
        fig_q3_q4(cache)
    except Exception as e:
        report["insight_fig_error"] = str(e)
    report["n_cache"] = len(cache)
    report["n_valid"] = sum(1 for c in cache if c.get("valid"))

    with open(os.path.join(RES, "analysis.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1)[:4000])


if __name__ == "__main__":
    main()
