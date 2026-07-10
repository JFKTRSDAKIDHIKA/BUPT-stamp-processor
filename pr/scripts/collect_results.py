#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
collect_results.py — Task 4:结果收集与 A/B 策略对比。

数据源(每个 run 目录):
  metrics.json     OpenROAD -metrics 输出(utl::metric:时序/面积/HB 几何/DRV)
  openroad.log     GRT 最终拥堵表、report_power 功耗表
  congestion.rpt   GRT 拥堵热点(带坐标)→ 按 clusters.json 的核心区/边缘带
                   几何分区统计 —— 直接回答"策略 B 是否改善核心区拥堵、
                   边缘通道是否成为新瓶颈"
输出:
  pr/out/results.csv   全量指标平表
  pr/out/summary.md    每 (die × 变体) 的 A vs B 对比 + Δ
"""
import csv
import json
import re
from pathlib import Path

PR_ROOT = Path(__file__).resolve().parents[1]


# ── 解析器 ────────────────────────────────────────────────────────────
def load_metrics(rd: Path):
    """utl::metric JSON;键可能是 'A::b' 原样。数值化。"""
    f = rd / "metrics.json"
    if not f.exists():
        return {}
    try:
        raw = json.loads(f.read_text())
    except json.JSONDecodeError:
        return {}
    out = {}
    for k, v in raw.items():
        k = k.replace("__", "::") if "::" not in k else k
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def parse_grt_log(log: str):
    """GRT 最终拥堵表的 Total 行:资源利用率 % 与总溢出。"""
    m = re.search(
        r"Final congestion report.*?^\s*Total\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+"
        r"(\d+)\s*/\s*(\d+)\s*/\s*(\d+)", log, re.S | re.M)
    if not m:
        return {}
    return dict(grt_resource=int(m.group(1)), grt_demand=int(m.group(2)),
                grt_usage_pct=float(m.group(3)),
                grt_overflow=int(m.group(6)))


def parse_power_log(log: str):
    """report_power 的 Total 行(W)。宏内部功耗未建模,见 gen_sram_lib.py。"""
    tail = log[log.rfind("===== FINAL POWER ====="):]
    m = re.search(r"^Total\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                  r"([\d.eE+-]+)", tail, re.M)
    if not m:
        return {}
    return dict(power_internal_w=float(m.group(1)),
                power_switching_w=float(m.group(2)),
                power_leakage_w=float(m.group(3)),
                power_total_w=float(m.group(4)))


def parse_congestion_regions(rd: Path):
    """拥堵热点按区域分类:核心内部 vs 边缘带(含通道)。

    congestion.rpt 每条违例带 bbox 坐标;宽容解析所有 4 数值坐标组。
    分类依据 clusters.json:core bbox 向内收 edge_band_w 得到"核心内圈",
    热点中心落在内圈 = 核心区拥堵,否则计作边缘带拥堵。
    """
    geo_f, rpt_f = rd / "clusters.json", rd / "congestion.rpt"
    if not (geo_f.exists() and rpt_f.exists()):
        return {}
    geo = json.loads(geo_f.read_text())
    x1, y1, x2, y2 = geo["core"]
    b = geo["edge_band_w"]
    inner = (x1 + b, y1 + b, x2 - b, y2 - b)

    core_n = edge_n = 0
    for m in re.finditer(r"\(\s*([\d.]+)[, ]+([\d.]+)\s*\)\s*[-, ]*"
                         r"\(\s*([\d.]+)[, ]+([\d.]+)\s*\)",
                         rpt_f.read_text()):
        cx = (float(m.group(1)) + float(m.group(3))) / 2
        cy = (float(m.group(2)) + float(m.group(4))) / 2
        if inner[0] <= cx <= inner[2] and inner[1] <= cy <= inner[3]:
            core_n += 1
        else:
            edge_n += 1
    return dict(cong_hotspots_core=core_n, cong_hotspots_edge=edge_n)


def collect_run(entry):
    rd = Path(entry["run_dir"])
    row = dict(entry)
    row["done"] = (rd / "DONE").exists()
    mt = load_metrics(rd)
    # 关键指标改名为报告友好列
    key_map = {
        "IFP::instance_count": "instances",
        "DPL::utilization": "util_pct",
        "DPL::design_area_um2": "cell_area_um2",
        "HB::n_clusters": "hb_clusters",
        "HB::koz_area_um2": "koz_area_um2",
        "HB::koz_core_frac": "koz_core_frac",
        "HB::n_channels": "n_channels",
        "HB::channel_w_um": "channel_w_um",
        "GRT::congested": "grt_diverged",
        "DRT::drv": "drc_violations",
        "DRT::failed": "drt_failed",
        "DRT::worst_slack_max": "wns_ps",
        "DRT::worst_slack_min": "hold_wns_ps",
        "DRT::tns_max": "tns_ps",
        "DRT::clock_skew": "clk_skew_ps",
    }
    for src, dst in key_map.items():
        if src in mt:
            row[dst] = mt[src]
    log_f = rd / "openroad.log"
    if log_f.exists():
        log = log_f.read_text(errors="replace")
        row.update(parse_grt_log(log))
        row.update(parse_power_log(log))
    row.update(parse_congestion_regions(rd))
    return row


# ── 汇总报告 ──────────────────────────────────────────────────────────
COMPARE_COLS = [  # (列名, 表头, 格式, 越小越好?)
    ("wns_ps", "WNS(ps)", "{:.0f}", False),
    ("tns_ps", "TNS(ps)", "{:.0f}", False),
    ("util_pct", "利用率%", "{:.1f}", None),
    ("cell_area_um2", "单元面积µm²", "{:.0f}", True),
    ("grt_usage_pct", "GRT利用%", "{:.1f}", True),
    ("grt_overflow", "GRT溢出", "{:.0f}", True),
    ("cong_hotspots_core", "核心热点", "{:.0f}", True),
    ("cong_hotspots_edge", "边缘热点", "{:.0f}", True),
    ("drc_violations", "DRC", "{:.0f}", True),
    ("power_total_w", "功耗(W)", "{:.4f}", True),
]


def fmt(row, col, spec):
    v = row.get(col)
    if v is None or v == "":
        return "—"
    try:
        return spec.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def summary_md(rows):
    by_key = {}
    for r in rows:
        by_key.setdefault((r["die"], r["variant"]), {})[r["strategy"]] = r
    L = ["# MOBOL 3D 堆叠 PR:策略 A(直穿/无重分布) vs 策略 B(边缘重分布)",
         "",
         "- A = `no_redist`:DRAM 直穿 → base/buffer 核心区密集 KOZ",
         "- B = `edge_redist`:buffer 高层金属重分布 → HB 全部在边缘,核心区零 KOZ",
         "- Δ 为 B−A(时序 Δ>0 好;拥堵/面积/功耗 Δ<0 好)", ""]
    for (die, variant), s in sorted(by_key.items()):
        a, b = s.get("no_redist"), s.get("edge_redist")
        L.append(f"## {die} · 变体 {variant}")
        L.append("| 指标 | A 直穿 | B 边缘重分布 | Δ (B−A) |")
        L.append("|---|---|---|---|")
        for col, hdr, spec, _ in COMPARE_COLS:
            va = fmt(a or {}, col, spec)
            vb = fmt(b or {}, col, spec)
            d = "—"
            try:
                d = spec.format(float(b[col]) - float(a[col]))
            except (TypeError, KeyError, ValueError):
                pass
            L.append(f"| {hdr} | {va} | {vb} | {d} |")
        for tag, r in (("A", a), ("B", b)):
            if r and not r.get("done"):
                L.append(f"\n> ⚠ 策略 {tag} 的 run 未完成(见 {r['run_id']}/openroad.log)")
        L.append("")
    return "\n".join(L)


def main():
    manifest = json.loads((PR_ROOT / "runs" / "manifest.json").read_text())
    rows = [collect_run(e) for e in manifest]

    out = PR_ROOT / "out"
    out.mkdir(exist_ok=True)
    cols = sorted({k for r in rows for k in r})
    with open(out / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    (out / "summary.md").write_text(summary_md(rows))

    n_done = sum(r["done"] for r in rows)
    print(f"[collect] {n_done}/{len(rows)} run 完成 → {out/'results.csv'}, {out/'summary.md'}")
    print(summary_md(rows))


if __name__ == "__main__":
    main()
