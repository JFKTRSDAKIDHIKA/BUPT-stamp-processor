#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_layout.py — 为每个 run 生成两类图片:

  floorplan.png  布局规划图:die/core、宏阵列、HB pad 簇(按 KOZ 深度着色)、
                 重分布通道、边缘带 —— 展示策略 A/B 的物理意图(注记版)
  layout.png     物理版图:DEF 真实数据(宏 + 标准单元 + 分层布线 + HB 面阵
                 引脚)渲染;本机 OpenROAD 无 GUI/save_image、无 KLayout,
                 故用 matplotlib 直接渲染 DEF

  另为每个 (die × 变体) 生成 A/B 并排对比图 → out/plots/<die>__<variant>_AB.png

用法:
  python3 scripts/plot_layout.py                 # 所有已完成(有 DEF)的 run
  python3 scripts/plot_layout.py --runs runs/smoke_die__baseline__no_redist
"""
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Rectangle

PR_ROOT = Path(__file__).resolve().parents[1]

# 图例配色(KOZ 深度语义见 floorplan_hb.tcl)
MODE_COLOR = {"hard": "#d62728", "punch": "#7f0000", "shallow": "#ff9f40"}
MODE_LABEL = {"hard": "KOZ hard (pad, top-2 metal blocked)",
              "punch": "KOZ punch (feed-through, all metal)",
              "shallow": "shallow pad (M8/M9 capacity only)"}
LAYER_GROUP = [  # 布线按层分组着色
    (re.compile(r"^M[123]$"), "#4c72b0", 0.25, "M1-M3"),
    (re.compile(r"^M[4567]$"), "#2ca02c", 0.35, "M4-M7"),
    (re.compile(r"^M[89]$"), "#d62728", 0.6, "M8-M9 (redistribution)"),
]
MAX_SEGS_PER_GROUP = 400_000   # 大 die 布线段抽样上限(绘图性能)


# ── 解析 ──────────────────────────────────────────────────────────────
def tcl_cfg(rd: Path):
    cfg = {}
    for m in re.finditer(r"^set cfg\((\w+)\) (.*)$",
                         (rd / "run_cfg.tcl").read_text(), re.M):
        v = m.group(2).strip()
        if v.startswith("{") and v.endswith("}"):
            v = v[1:-1]          # 只剥最外层大括号,嵌套列表保留在内
        cfg[m.group(1)] = v
    return cfg


def lef_sizes(*lef_paths):
    sizes = {}
    for lp in lef_paths:
        txt = Path(lp).read_text()
        for m in re.finditer(r"^MACRO (\S+).*?SIZE\s+([\d.]+)\s+BY\s+([\d.]+)",
                             txt, re.S | re.M):
            sizes[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    return sizes


def parse_def(def_file: Path, want_nets=True):
    """流式解析 DEF:元件(placed/fixed)、M9 面阵引脚、布线段。"""
    comps, pins, segs = [], [], {}   # segs: group_idx -> [(x1,y1,x2,y2)]
    dbu = 20000.0
    sec = None
    cur_layer = None
    re_units = re.compile(r"UNITS DISTANCE MICRONS (\d+)")
    re_comp = re.compile(r"^\s*-\s+(\S+)\s+(\S+)")
    re_place = re.compile(r"\+\s+(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)\s+(\w+)")
    re_pinlayer = re.compile(r"\+\s+LAYER\s+(M9)\b")
    re_route = re.compile(r"^\s*(?:\+\s+ROUTED|NEW)\s+(M\d+)")
    re_pts = re.compile(r"\(\s*(-?\d+|\*)\s+(-?\d+|\*)(?:\s+-?\d+)?\s*\)")

    comp_buf = None
    pin_is_m9 = False
    with open(def_file, errors="replace") as f:
        for line in f:
            um = re_units.search(line)
            if um:
                dbu = float(um.group(1))
            if line.startswith("COMPONENTS"):
                sec = "comp"; continue
            if line.startswith("PINS"):
                sec = "pins"; continue
            if line.startswith("NETS"):
                sec = "nets" if want_nets else None
                continue
            if line.startswith(("END COMPONENTS", "END PINS", "END NETS")):
                sec = None; continue

            if sec == "comp":
                cm = re_comp.match(line)
                if cm:
                    comp_buf = cm.group(2)   # master 名
                pm = re_place.search(line)
                if pm and comp_buf:
                    comps.append((comp_buf, int(pm.group(1)) / dbu,
                                  int(pm.group(2)) / dbu, pm.group(3)))
                    comp_buf = None
            elif sec == "pins":
                if line.lstrip().startswith("-"):
                    pin_is_m9 = False
                if re_pinlayer.search(line):
                    pin_is_m9 = True
                pm = re_place.search(line)
                if pm and pin_is_m9:
                    pins.append((int(pm.group(1)) / dbu, int(pm.group(2)) / dbu))
                    pin_is_m9 = False
            elif sec == "nets":
                rm = re_route.match(line)
                if rm:
                    cur_layer = rm.group(1)
                if cur_layer and "(" in line:
                    gi = next((i for i, (rx, *_r) in enumerate(LAYER_GROUP)
                               if rx.match(cur_layer)), None)
                    if gi is not None:
                        px = py = None
                        prev = None
                        for xm, ym in re_pts.findall(line):
                            x = px if xm == "*" else int(xm) / dbu
                            y = py if ym == "*" else int(ym) / dbu
                            if prev is not None and (x, y) != prev:
                                segs.setdefault(gi, []).append((prev[0], prev[1], x, y))
                            prev = (x, y); px, py = x, y
                if line.strip().endswith(";"):
                    cur_layer = None
    return comps, pins, segs


# ── 绘图 ──────────────────────────────────────────────────────────────
def draw_frame(ax, cfg, geo):
    dw, dh = float(cfg["die_w"]), float(cfg["die_h"])
    ax.add_patch(Rectangle((0, 0), dw, dh, fill=False, ec="black", lw=1.2))
    x1, y1, x2, y2 = geo["core"]
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                           ec="gray", ls="--", lw=0.8))
    ax.set_xlim(-dw * 0.02, dw * 1.02)
    ax.set_ylim(-dh * 0.02, dh * 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("µm")


def draw_floorplan(ax, rd, cfg, geo, macro_rects):
    draw_frame(ax, cfg, geo)
    x1, y1, x2, y2 = geo["core"]
    # 策略 B 的边缘带底色
    if cfg["strategy"] == "edge_redist":
        b = geo["edge_band_w"]
        outer = Rectangle((x1, y1), x2 - x1, y2 - y1)
        inner = Rectangle((x1 + b, y1 + b), x2 - x1 - 2 * b, y2 - y1 - 2 * b)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fc="#9ecae1",
                               alpha=0.25, ec="none"))
        ax.add_patch(Rectangle((x1 + b, y1 + b), (x2 - x1) - 2 * b,
                               (y2 - y1) - 2 * b, fc="white", ec="none"))
    # 重分布通道
    for c in geo.get("channels", []):
        cx1, cy1, cx2, cy2 = c["rect"]
        ax.add_patch(Rectangle((cx1, cy1), cx2 - cx1, cy2 - cy1,
                               fc="#2ca02c", alpha=0.25, ec="#2ca02c", lw=0.5))
    # 宏
    for (mx, my, mw, mh) in macro_rects:
        ax.add_patch(Rectangle((mx, my), mw, mh, fc="#bbbbbb", ec="#555555", lw=0.6))
    # HB 簇(KOZ)
    for cl in geo["clusters"]:
        rx1, ry1, rx2, ry2 = cl["rect"]
        col = MODE_COLOR.get(cl["mode"], "red")
        ax.add_patch(Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1,
                               fc=col, alpha=0.55, ec=col, lw=0.8))
        if rx2 - rx1 > float(cfg["die_w"]) * 0.02:
            ax.text((rx1 + rx2) / 2, (ry1 + ry2) / 2, cl["name"],
                    ha="center", va="center", fontsize=5, color="white")
    strat = "A no_redist (through-stack KOZ)" if cfg["strategy"] == "no_redist" \
        else "B edge_redist (edge HB, clean core)"
    ax.set_title(f"Floorplan  {cfg['die_name']} / {cfg['variant']}\n{strat}", fontsize=9)
    # 图例:只列出本图实际出现的元素
    handles = [Rectangle((0, 0), 1, 1, fc="#bbbbbb", ec="#555555", label="SRAM macro")]
    for mode in sorted({cl["mode"] for cl in geo["clusters"]}):
        handles.append(Rectangle((0, 0), 1, 1, fc=MODE_COLOR.get(mode, "red"),
                                 alpha=0.55, label=MODE_LABEL.get(mode, mode)))
    if geo.get("channels"):
        handles.append(Rectangle((0, 0), 1, 1, fc="#2ca02c", alpha=0.25,
                                 label="redistribution channel"))
    if cfg["strategy"] == "edge_redist":
        handles.append(Rectangle((0, 0), 1, 1, fc="#9ecae1", alpha=0.25,
                                 label="edge HB band"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              fontsize=6, ncol=2, framealpha=0.9)


def draw_layout(ax, cfg, geo, comps, pins, segs, sizes):
    draw_frame(ax, cfg, geo)
    # 标准单元 / 宏
    std_patches, mac_patches = [], []
    for master, x, y, _o in comps:
        w, h = sizes.get(master, (0.2, 0.144))
        (mac_patches if w > 5 else std_patches).append(
            Rectangle((x, y), w, h))
    ax.add_collection(PatchCollection(std_patches, fc="#4c72b0", ec="none",
                                      alpha=0.6, rasterized=True))
    ax.add_collection(PatchCollection(mac_patches, fc="#999999", ec="#444444",
                                      lw=0.6))
    # 布线(按层分组;大 run 抽样)
    for gi, (rx, color, alpha, label) in enumerate(LAYER_GROUP):
        ss = segs.get(gi, [])
        if len(ss) > MAX_SEGS_PER_GROUP:
            ss = ss[:: len(ss) // MAX_SEGS_PER_GROUP + 1]
        if ss:
            ax.add_collection(LineCollection(
                [((a, b), (c, d)) for a, b, c, d in ss],
                colors=color, alpha=alpha, linewidths=0.15, rasterized=True))
    # HB 面阵引脚(M9)
    if pins:
        xs, ys = zip(*pins)
        ax.scatter(xs, ys, s=0.5, c="#ff9f40", marker="s", rasterized=True)
    nseg = sum(len(v) for v in segs.values())
    ax.set_title(f"Physical layout  {cfg['die_name']} / {cfg['variant']} / "
                 f"{cfg['strategy']}\n{len(comps):,} insts · {nseg:,} route segs · "
                 f"{len(pins):,} HB face pins", fontsize=9)
    from matplotlib.lines import Line2D
    handles = [Rectangle((0, 0), 1, 1, fc="#999999", ec="#444444", label="SRAM macro"),
               Rectangle((0, 0), 1, 1, fc="#4c72b0", alpha=0.6, label="std cells")]
    for gi, (_rx, color, alpha, label) in enumerate(LAYER_GROUP):
        if segs.get(gi):
            handles.append(Line2D([], [], color=color, alpha=min(1.0, alpha + 0.3),
                                  lw=1.5, label=f"routing {label}"))
    if pins:
        handles.append(Line2D([], [], color="#ff9f40", marker="s", ls="none",
                              ms=4, label="HB face pins (M9)"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              fontsize=6, ncol=3, framealpha=0.9)


def plot_run(rd: Path, sizes):
    cfg = tcl_cfg(rd)
    geo_f = rd / "clusters.json"
    def_f = rd / f"{cfg['run_id']}.def"
    if not (geo_f.exists() and def_f.exists()):
        return None
    geo = json.loads(geo_f.read_text())
    comps, pins, segs = parse_def(def_f)
    macro_rects = [(x, y, *sizes.get(m, (0, 0))) for m, x, y, _o in comps
                   if sizes.get(m, (0, 0))[0] > 5]

    fig, ax = plt.subplots(figsize=(8, 8 * float(cfg["die_h"]) / float(cfg["die_w"])))
    draw_floorplan(ax, rd, cfg, geo, macro_rects)
    fig.savefig(rd / "floorplan.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 9 * float(cfg["die_h"]) / float(cfg["die_w"])))
    draw_layout(ax, cfg, geo, comps, pins, segs, sizes)
    fig.savefig(rd / "layout.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {rd.name}: floorplan.png + layout.png")
    return cfg


def combo_ab(runs_done):
    """每 (die×变体) 输出 A/B 并排对比(floorplan 与 layout 各一张)。"""
    out = PR_ROOT / "out" / "plots"
    out.mkdir(parents=True, exist_ok=True)
    by = {}
    for rd, cfg in runs_done:
        by.setdefault((cfg["die_name"], cfg["variant"]), {})[cfg["strategy"]] = rd
    from PIL import Image  # 若无 PIL 用 matplotlib 拼图
    for (die, var), s in by.items():
        if "no_redist" not in s or "edge_redist" not in s:
            continue
        for kind in ("floorplan", "layout"):
            ims = [Image.open(s[st] / f"{kind}.png")
                   for st in ("no_redist", "edge_redist")]
            h = max(im.height for im in ims)
            ims = [im.resize((int(im.width * h / im.height), h)) for im in ims]
            cat = Image.new("RGB", (sum(im.width for im in ims) + 20, h), "white")
            x = 0
            for im in ims:
                cat.paste(im, (x, 0)); x += im.width + 20
            f = out / f"{die}__{var}_{kind}_AB.png"
            cat.save(f)
            print(f"[plot] A/B 对比 → {f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    args = ap.parse_args()
    run_dirs = ([Path(r) for r in args.runs] if args.runs
                else sorted((PR_ROOT / "runs").glob("*__*")))
    # LEF 尺寸表(标准单元 + SRAM 宏)
    any_cfg = tcl_cfg(run_dirs[0])
    sizes = lef_sizes(any_cfg["std_lef"], *re.findall(r"\{([^{}]+)\}", any_cfg["macro_lefs"])
                      if "{" in any_cfg["macro_lefs"] else [any_cfg["macro_lefs"]])
    done = []
    for rd in run_dirs:
        if not (rd / "run_cfg.tcl").exists():
            continue
        try:
            cfg = plot_run(rd, sizes)
            if cfg:
                done.append((rd, cfg))
        except Exception as e:
            print(f"[plot] {rd.name} 失败: {e}")
    try:
        combo_ab(done)
    except ImportError:
        print("[plot] 无 PIL,跳过 A/B 拼图(单图已生成)")


if __name__ == "__main__":
    main()
