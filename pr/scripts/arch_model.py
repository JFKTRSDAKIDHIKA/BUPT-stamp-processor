#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arch_model.py — Task 1:架构变体参数提取 + 参数化物理模型。

数据流:
  config/mobol_arch.yaml (compiler.dse)     → 架构变体(W 旋钮 / DRAM 键合密度 / NMC)
  ARCH_SPEC.md §7 垂直互连语义               → HB 接口数量/位宽公式(编码在 variant_hb)
  synth/build/<top>/design.v + PDK LEF      → OpenROAD 逐实例精确面积(area_probe.tcl,缓存)
  pr/config/experiments.yaml                → 工艺假设(HB pitch / KOZ / 通道利用率)与 die 绑定

输出:
  每 (die × variant) 的物理模型:die/core 尺寸、HB 簇数量与位宽、簇/KOZ 几何、
  策略 B 的边缘带宽度与重分布通道宽度。gen_runs.py 直接 import 本模块;
  独立运行则打印 Task-1 模型表并写 pr/out/arch_model.json。

HB 接口建模(源自 ARCH_SPEC §7 + mobol_arch.yaml vbond):
  * tile↔bank 链路:全芯片 16 条(每 tile 一条私有键合),
      每条位宽 = W × 512b(64B flit)× 2 方向 + 控制冗余
      W = 变体的 tile_link(baseline=1, W2=2 …)
  * bank列↔DRAM:全芯片 4 列,每列每方向 dram_density bit
      (2048=默认拐点,8192=dense8192 变体)。DRAM die 为 16 通道(§8),
      故按 16 个"通道簇"落 pad:每簇位宽 = 4×density/16 × 2 方向 + 控制。
  * 策略 A(no_redist):通道簇直接穿透 buffer die 落在 base die 核心区
      (feed-through)→ 两个 die 的核心区都被 KOZ 切碎。
  * 策略 B(edge_redist):buffer die 用 M8/M9 把 DRAM 信号重分布到边缘,
      base die 所有 HB(tile 链路 + DRAM 通道)只出现在边缘带,核心区零 KOZ。
"""
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

PR_ROOT = Path(__file__).resolve().parents[1]      # …/pr
REPO_ROOT = PR_ROOT.parent


# ──────────────────────────────────────────────────────────────────────
# 配置装载
# ──────────────────────────────────────────────────────────────────────
def load_experiments(cfg_path=None):
    """读 experiments.yaml;paths 支持环境变量覆盖(MOBOL_ROOT/PDK_ROOT)。"""
    cfg_path = Path(cfg_path or PR_ROOT / "config" / "experiments.yaml")
    cfg = yaml.safe_load(cfg_path.read_text())
    cfg["paths"]["mobol_root"] = os.environ.get(
        "MOBOL_ROOT", cfg["paths"]["mobol_root"])
    cfg["paths"]["pdk_root"] = os.environ.get(
        "PDK_ROOT", cfg["paths"]["pdk_root"])
    return cfg


def load_variants(cfg):
    """从 mobol_arch.yaml(单一真值源)读架构变体列表与结构常量。"""
    arch = yaml.safe_load(
        (Path(cfg["paths"]["mobol_root"]) / "config" / "mobol_arch.yaml").read_text())
    structural = arch["structural"]
    variants = arch["compiler"]["dse"]
    for v in variants:
        v["id"] = re.sub(r"[^\w.-]", "_", str(v["label"]))  # 文件系统安全 id
    return structural, variants


def pdk_paths(cfg):
    """PDK 各视图的绝对路径(与 3nm.vars 同构,但不依赖其 Tcl)。"""
    root = Path(cfg["paths"]["pdk_root"])
    orun = root / "PnR-OpenROAD" / "Front-side Version"
    return dict(
        tech_lef=orun / "TECH-LEF" / "3nm_GAA_FSPR.tech.lef",
        std_lef=orun / "LEF" / "3nm_GAA_FSPR.lef",
        std_lib=orun / "LIB" / "3nm_GAA_FSPR_rvt_nldm.lib",
        rc_tcl=orun / "RC" / "setRC.tcl",
        tracks=orun / "OpenROAD_Runfiles" / "3nm" / "3nm.tracks",
        pdn=orun / "OpenROAD_Runfiles" / "3nm" / "3nm.pdn.tcl",
    )


def macro_views(cfg):
    """SRAM 宏的 LEF 与(自动生成的)Liberty 路径列表。"""
    mroot = Path(cfg["paths"]["mobol_root"]) / "synth" / "macros"
    lib_dir = PR_ROOT / "macros_lib"
    lefs, libs = [], []
    for mdir in sorted(mroot.iterdir()):
        lef = mdir / f"{mdir.name}.lef"
        if lef.exists():
            lefs.append(lef)
            libs.append(lib_dir / f"{mdir.name}.lib")
    return lefs, libs


# ──────────────────────────────────────────────────────────────────────
# 网表精确面积(OpenROAD 探测 + 缓存)
# ──────────────────────────────────────────────────────────────────────
def netlist_area(cfg, top, force=False):
    """调 area_probe.tcl 得到 {n_std, std_um2, n_macro, macro_um2, n_ports}。"""
    cache = PR_ROOT / "out" / "netlist_area.json"
    cache.parent.mkdir(parents=True, exist_ok=True)
    db = json.loads(cache.read_text()) if cache.exists() else {}
    if top in db and not force:
        return db[top]

    # 宏 liberty 不存在时先生成(area_probe link 需要)
    lefs, libs = macro_views(cfg)
    if not all(p.exists() for p in libs):
        subprocess.run([sys.executable, str(PR_ROOT / "scripts" / "gen_sram_lib.py")],
                       check=True)

    pdk = pdk_paths(cfg)
    # 注意:build/<top>/design.v 是综合前 RTL,后端一律用门级网表 <top>.gate.v
    netlist = Path(cfg["paths"]["mobol_root"]) / "synth" / "build" / top / f"{top}.gate.v"
    out = PR_ROOT / "out" / f"_probe_{top}.json"
    env = dict(os.environ,
               AP_TECH_LEF=str(pdk["tech_lef"]), AP_STD_LEF=str(pdk["std_lef"]),
               AP_MACRO_LEFS=" ".join(map(str, lefs)),
               AP_STD_LIB=str(pdk["std_lib"]),
               AP_MACRO_LIBS=" ".join(map(str, libs)),
               AP_NETLIST=str(netlist), AP_TOP=top, AP_OUT=str(out))
    print(f"[arch_model] 探测网表面积: {top} …", file=sys.stderr)
    r = subprocess.run([cfg["openroad"]["bin"], "-no_init", "-no_splash", "-exit",
                        str(PR_ROOT / "scripts" / "openroad" / "area_probe.tcl")],
                       env=env, capture_output=True, text=True, timeout=1800)
    if "AREA_PROBE_OK" not in r.stdout:
        raise RuntimeError(f"area probe 失败({top}):\n{r.stdout[-3000:]}\n{r.stderr[-1000:]}")
    db[top] = json.loads(out.read_text())
    out.unlink()
    cache.write_text(json.dumps(db, indent=2))
    return db[top]


# ──────────────────────────────────────────────────────────────────────
# HB 接口模型
# ──────────────────────────────────────────────────────────────────────
def variant_hb(variant, structural, tech, share):
    """某变体在"占整层 share 比例"的 die 上分到的 HB 接口规模。"""
    W = int(variant["tile_link"])
    density = int(variant["dram_density"])
    ctrl = tech["hb_ctrl_bits"]
    num_tiles = structural["num_tiles"]          # 16
    num_banks = structural["num_banks"]          # 4
    n_dram_ch_total = 16                         # DRAM die 16 通道(SPEC §8)

    bits_tl = W * 512 * 2 + ctrl                             # 每条 tile 链路 pad 数
    bits_dc = (num_banks * density // n_dram_ch_total) * 2 + ctrl  # 每 DRAM 通道 pad 数
    return dict(
        n_tl=round(num_tiles * share), bits_tl=bits_tl,
        n_dc=round(n_dram_ch_total * share), bits_dc=bits_dc,
    )


def cluster_side(nbits, pitch):
    """nbits 个 pad 排成正方阵的边长(µm)。"""
    return math.ceil(math.sqrt(nbits)) * pitch


def macro_dims(cfg):
    """synth/macros 下各宏的 LEF 尺寸 {name: (w, h)}。"""
    out = {}
    mroot = Path(cfg["paths"]["mobol_root"]) / "synth" / "macros"
    for mdir in sorted(mroot.iterdir()):
        lef = mdir / f"{mdir.name}.lef"
        if not lef.exists():
            continue
        m = re.search(rf"MACRO\s+{re.escape(mdir.name)}\b.*?SIZE\s+([\d.]+)\s+BY\s+([\d.]+)",
                      lef.read_text(), re.S)
        if m:
            out[mdir.name] = (float(m.group(1)), float(m.group(2)))
    return out


# ──────────────────────────────────────────────────────────────────────
# 每 (die, variant) 的完整物理模型
# ──────────────────────────────────────────────────────────────────────
def build_model(cfg, die_name, die, variant, structural):
    tech = cfg["tech"]
    pitch = tech["hb_pitch_um"]
    kmarg = tech["koz_margin_um"]

    area = netlist_area(cfg, die["top"])
    std_um2, macro_um2 = area["std_um2"], area["macro_um2"]

    # NMC 变体:buffer die 每 bank 侧挂 1 个 NMC 引擎 + 1 个预取引擎(SPEC §6.4/6.5)。
    # 现有 netlist 未集成它们,这里为其预留面积(die 尺寸真实,布局区留白)。
    nmc_um2 = 0.0
    if variant.get("nmc") and die["role"] == "buffer":
        n_banks_here = round(structural["num_banks"] * die["share"] * 4) or 1
        n_banks_here = max(1, round(4 * die["share"]))
        for eng in ("nmc_engine", "prefetch_engine"):
            a = netlist_area(cfg, eng)
            nmc_um2 += (a["std_um2"] + a["macro_um2"]) * n_banks_here

    hb = variant_hb(variant, structural, tech, die["share"])
    side_tl = cluster_side(hb["bits_tl"], pitch)
    side_dc = cluster_side(hb["bits_dc"], pitch)
    koz_tl = side_tl + 2 * kmarg
    koz_dc = side_dc + 2 * kmarg
    n_clusters = hb["n_tl"] + hb["n_dc"]

    # ── 宏阵列排布(tile SRAM / bank SRAM 与架构分组对齐) ──
    # cols×rows 选取:块状排布的长宽比最接近目标 aspect
    aspect = die.get("aspect", 1.0)
    halo = cfg["flow"]["macro_halo_um"]
    mw = mh = 0.0
    cols = rows = 0
    if area["n_macro"]:
        mname = max(area["macros"], key=area["macros"].get)
        mw, mh = macro_dims(cfg)[mname]
        n = area["n_macro"]
        best = None
        for c in range(1, n + 1):
            r = math.ceil(n / c)
            if c * r - n >= c:      # 不允许整行空缺
                continue
            score = abs((c * mw) / (r * mh) - aspect)
            if best is None or score < best[0]:
                best = (score, c, r)
        cols, rows = best[1], best[2]

    # ── 策略无关的基础核心面积需求 ──
    base_need = (std_um2 / die["target_util"]
                 + macro_um2 / tech["macro_util"] + nmc_um2 / die["target_util"])

    # ── 策略 A 的 KOZ 面积(核心区被切走的部分) ──
    a_koz = hb["n_tl"] * koz_tl ** 2 + hb["n_dc"] * koz_dc ** 2
    koz_big = max(koz_tl, koz_dc)

    # ── 策略 B:边缘带 + (buffer)重分布通道 ──
    band_w = koz_big + tech["edge_gap_um"]
    # 重分布通道:该 die 上所有垂直信号都要从核心走到边缘
    redist_bits = hb["n_tl"] * hb["bits_tl"] + hb["n_dc"] * hb["bits_dc"]
    ch_total_per_edge = 0.0
    if die["role"] == "buffer":
        # 每边分担 1/4 的重分布位宽;单层金属(垂直 M9 / 水平 M8),
        # 通道内利用率 channel_util —— "留够空间"的量化保证
        ch_total_per_edge = (redist_bits / 4.0) * tech["redist_pitch_um"] / tech["channel_util"]

    # ── 宏阵列决定的最小核心尺寸(A/B 取更严者,保证同变体 A/B 同尺寸可比) ──
    w_min = h_min = 0.0
    if cols:
        gap_a = koz_big + 2 * halo          # A:宏间通道要装下 KOZ 簇
        gap_b = 2 * halo                    # B:宏间只留 halo
        w_min = max(cols * mw + (cols + 1) * gap_a,
                    cols * mw + (cols + 1) * gap_b + 2 * band_w)
        h_min = max(rows * mh + (rows + 1) * gap_a,
                    rows * mh + (rows + 1) * gap_b + 2 * band_w)

    # 固定点迭代:core 面积 = 基础需求 + max(A 的 KOZ, B 的边缘带+通道软损耗)
    core_area = base_need + a_koz
    for _ in range(4):
        cw = math.sqrt(core_area * aspect)
        ch_ = core_area / cw
        ring = 2 * (cw + ch_) * band_w
        chan = 0.75 * (2 * ch_total_per_edge * cw + 2 * ch_total_per_edge * ch_) \
            if ch_total_per_edge else 0.0  # 软阻挡按 75% 面积损耗计
        core_area = base_need + max(a_koz, ring + chan)

    # 尺寸下限(宏阵列)与面积需求联合满足
    cw = max(cw, w_min)
    ch_ = max(ch_, h_min)
    if cw * ch_ < core_area:
        s = math.sqrt(core_area / (cw * ch_))
        cw *= s
        ch_ *= s

    # 边缘容纳性检查:B 策略所有簇沿四边排开必须放得下,否则线性放大核心
    per_edge = math.ceil(n_clusters / 4)
    need_len = per_edge * koz_big * 1.15 + 2 * band_w
    if need_len > min(cw, ch_):
        scale = need_len / min(cw, ch_)
        cw *= scale
        ch_ *= scale

    m = tech["core_margin_um"]

    def q(x):  # 对齐到 0.1µm 网格,避免奇异坐标
        return round(x, 1)

    model = dict(
        die=die_name, top=die["top"], role=die["role"],
        variant=variant["label"], variant_id=variant["id"],
        W=variant["tile_link"], dram_density=variant["dram_density"],
        nmc=bool(variant.get("nmc")),
        # 面积
        n_std=area["n_std"], n_macro=area["n_macro"],
        std_um2=std_um2, macro_um2=macro_um2, nmc_reserve_um2=nmc_um2,
        macro_cols=cols, macro_rows=rows, macro_w=mw, macro_h=mh,
        core_w=q(cw), core_h=q(ch_),
        die_w=q(cw + 2 * m), die_h=q(ch_ + 2 * m), core_margin=m,
        # HB 接口
        grid_x=die["grid"][0], grid_y=die["grid"][1],
        hb_n_tile_links=hb["n_tl"], hb_bits_tile_link=hb["bits_tl"],
        hb_n_dram_ch=hb["n_dc"], hb_bits_dram_ch=hb["bits_dc"],
        hb_pitch=pitch, koz_margin=kmarg, edge_gap=tech["edge_gap_um"],
        side_tl=q(side_tl), side_dc=q(side_dc),
        koz_tl=q(koz_tl), koz_dc=q(koz_dc),
        koz_area_um2=q(a_koz),
        koz_core_frac=round(a_koz / core_area, 4),
        # 策略 B 几何
        edge_band_w=q(band_w),
        channels_per_edge=die["grid"][0],
        channel_w=q(ch_total_per_edge / die["grid"][0]) if ch_total_per_edge else 0.0,
        offchannel_adj=tech["offchannel_adj"],
        shallow_pad_adj=tech["shallow_pad_adj"],
        # 引脚映射
        hb_pin_patterns=die["hb_pin_patterns"],
        target_util=die["target_util"],
    )
    return model


def all_models(cfg, dies=None, variants_sel=None):
    structural, variants = load_variants(cfg)
    exp = cfg["experiment"]
    die_names = dies or exp["dies"]
    sel = variants_sel or exp["variants"]
    if sel != "all" and not isinstance(sel, str):
        variants = [v for v in variants if v["label"] in sel or v["id"] in sel]
    out = []
    for dn in die_names:
        for v in variants:
            out.append(build_model(cfg, dn, cfg["dies"][dn], v, structural))
    return out


# ──────────────────────────────────────────────────────────────────────
# Task-1 报告
# ──────────────────────────────────────────────────────────────────────
def report(models):
    hdr = ("| die | 变体 | W | DRAM密度 | NMC | 标准单元 | 宏 | die (µm) | "
           "tile链路 (数×位) | DRAM通道 (数×位) | KOZ簇边长 tl/dc | A:KOZ占核心 | "
           "B:边缘带 | B:通道宽/边 |")
    sep = "|" + "---|" * 14
    rows = [hdr, sep]
    for m in models:
        rows.append(
            f"| {m['die']} | {m['variant']} | {m['W']} | {m['dram_density']} | "
            f"{'√' if m['nmc'] else '—'} | {m['n_std']:,} ({m['std_um2']/1e6:.3f}mm²) | "
            f"{m['n_macro']} ({m['macro_um2']/1e6:.3f}mm²) | "
            f"{m['die_w']:.0f}×{m['die_h']:.0f} | "
            f"{m['hb_n_tile_links']}×{m['hb_bits_tile_link']} | "
            f"{m['hb_n_dram_ch']}×{m['hb_bits_dram_ch']} | "
            f"{m['koz_tl']:.0f}/{m['koz_dc']:.0f}µm | "
            f"{m['koz_core_frac']*100:.1f}% | {m['edge_band_w']:.0f}µm | "
            f"{m['channels_per_edge']}×{m['channel_w']:.0f}µm |")
    return "\n".join(rows)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dies", nargs="*", default=None,
                    help="覆盖 experiments.yaml 的 die 列表(可含 smoke_die)")
    args = ap.parse_args()
    cfg = load_experiments(args.config)
    models = all_models(cfg, dies=args.dies)
    print(report(models))
    out = PR_ROOT / "out" / "arch_model.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(models, indent=2, ensure_ascii=False))
    print(f"\n[arch_model] 模型已写入 {out}", file=sys.stderr)
