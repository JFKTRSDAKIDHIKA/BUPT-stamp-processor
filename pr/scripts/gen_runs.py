#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_runs.py — Task 2/3:展开实验矩阵(die × 变体 × 策略),为每个 run 生成
run_cfg.tcl(pr_main.tcl 的唯一输入)与 manifest.json(批跑/收集的索引)。

用法:
  python3 gen_runs.py                        # experiments.yaml 的默认全集
  python3 gen_runs.py --dies smoke_die       # 冒烟
  python3 gen_runs.py --variants baseline dense8192 --strategies edge_redist
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import arch_model as am


def tcl_str(v):
    """python 值 → Tcl 字面量。"""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "{" + " ".join(tcl_str(x) for x in v) + "}"
    return "{" + str(v) + "}"


def emit_run_cfg(path: Path, kv: dict):
    lines = ["# ── 自动生成(gen_runs.py),勿手改;参数语义见 experiments.yaml ──"]
    for k, v in kv.items():
        lines.append(f"set cfg({k}) {tcl_str(v)}")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--dies", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--strategies", nargs="*", default=None)
    args = ap.parse_args()

    cfg = am.load_experiments(args.config)
    pdk = am.pdk_paths(cfg)
    macro_lefs, macro_libs = am.macro_views(cfg)

    # 宏 liberty 兜底生成
    if not all(p.exists() for p in macro_libs):
        subprocess.run([sys.executable,
                        str(am.PR_ROOT / "scripts" / "gen_sram_lib.py")], check=True)

    strategies = args.strategies or cfg["experiment"]["strategies"]
    models = am.all_models(cfg, dies=args.dies, variants_sel=args.variants)

    runs_dir = am.PR_ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    manifest = []
    build = Path(cfg["paths"]["mobol_root"]) / "synth" / "build"

    for m in models:
        top = m["top"]
        netlist = build / top / f"{top}.gate.v"   # 门级网表(design.v 是综合前 RTL)
        for strat in strategies:
            run_id = f"{m['die']}__{m['variant_id']}__{strat}"
            rd = runs_dir / run_id
            rd.mkdir(exist_ok=True)
            # 参数化 SDC:与综合口径一致(500MHz、IO 延迟=20% 周期),
            # 不用 synth 生成的 .sdc(其 remove_from_collection 在本
            # OpenROAD build 的 SDC 读取器中不可用)
            period = cfg["flow"]["clock_period_ps"]
            io_dly = int(period * 0.2)
            sdc = rd / "constraints.sdc"
            sdc.write_text(f"""# 自动生成(gen_runs.py):{run_id}
create_clock -name clk -period {period} [get_ports clk]
set nonclk {{}}
foreach p [all_inputs] {{
  if {{[get_full_name $p] ne "clk"}} {{ lappend nonclk $p }}
}}
set_input_delay  {io_dly} -clock clk $nonclk
set_output_delay {io_dly} -clock clk [all_outputs]
""")
            kv = dict(
                run_id=run_id,
                die_name=m["die"], role=m["role"], strategy=strat,
                variant=m["variant"], top=top, design=top,
                # 路径(全部绝对,不硬编码在 Tcl 内)
                netlist=str(netlist), sdc=str(sdc),
                tech_lef=str(pdk["tech_lef"]), std_lef=str(pdk["std_lef"]),
                std_lib=str(pdk["std_lib"]),
                macro_lefs=[str(p) for p in macro_lefs],
                macro_libs=[str(p) for p in macro_libs],
                tracks_file=str(pdk["tracks"]), pdn_tcl=str(pdk["pdn"]),
                rc_tcl=str(pdk["rc_tcl"]),
                script_dir=str(am.PR_ROOT / "scripts" / "openroad"),
                # 几何(arch_model 的变体物理模型)
                die_w=m["die_w"], die_h=m["die_h"], core_margin=m["core_margin"],
                grid_x=m["grid_x"], grid_y=m["grid_y"],
                macro_cols=m["macro_cols"], macro_rows=m["macro_rows"],
                macro_w=m["macro_w"], macro_h=m["macro_h"],
                hb_n_tile_links=m["hb_n_tile_links"],
                hb_bits_tile_link=m["hb_bits_tile_link"],
                hb_n_dram_ch=m["hb_n_dram_ch"],
                hb_bits_dram_ch=m["hb_bits_dram_ch"],
                hb_pitch=m["hb_pitch"], hb_pad=0.8,
                hb_layer="M9",
                koz_margin=m["koz_margin"], edge_gap=m["edge_gap"],
                edge_band_w=m["edge_band_w"],
                channels_per_edge=m["channels_per_edge"],
                channel_w=m["channel_w"],
                offchannel_adj=m["offchannel_adj"],
                shallow_pad_adj=m["shallow_pad_adj"],
                hb_pin_patterns=m["hb_pin_patterns"],
                # 流程旋钮(与 PDK 3nm.vars 口径一致)
                place_density=cfg["flow"]["place_density"],
                slew_margin=cfg["flow"]["slew_margin"],
                cap_margin=cfg["flow"]["cap_margin"],
                grt_congestion_iterations=cfg["flow"]["grt_congestion_iterations"],
                droute_end_iter=cfg["flow"]["droute_end_iter"],
                macro_halo=cfg["flow"]["macro_halo_um"],
                cts_buffer="BUFx4", cts_cluster_diameter=100,
                io_hor_layer="M2", io_ver_layer="M3",
                wire_rc_layer="M3", wire_rc_layer_clk="M6",
                routing_layers="M1-M9", clock_layers="M6-M9",
                layer_adjustments=["M1", 0.5, "M2", 0.5, "M3", 0.5, "M4", 0.5,
                                   "M5", 0.5, "M6", 0.5, "M7", 0.5, "M8", 0.5,
                                   "M9", 0.5],
            )
            emit_run_cfg(rd / "run_cfg.tcl", kv)
            manifest.append(dict(run_id=run_id, run_dir=str(rd), **{
                k: m[k] for k in ("die", "role", "variant", "variant_id", "top",
                                  "die_w", "die_h", "edge_band_w", "koz_core_frac")},
                strategy=strat))

    mf = am.PR_ROOT / "runs" / "manifest.json"
    mf.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"[gen_runs] 生成 {len(manifest)} 个 run → {runs_dir}")
    for r in manifest:
        print("  ", r["run_id"])


if __name__ == "__main__":
    main()
