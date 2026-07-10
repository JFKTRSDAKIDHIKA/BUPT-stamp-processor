#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_sram_lib.py — 为 synth/macros 下的 SRAM 宏生成占位 Liberty(NLDM)视图。

背景:synth/macros/<name>/ 只有 .lef/.v(fakeram 的 .lib 生成依赖原作者本机的
FakeRAM2.0 checkout,本服务器不可用)。OpenROAD link + STA + resizer 需要宏的
liberty,否则宏引脚上的路径无法计时。本脚本从宏的 blackbox verilog(.bb.v,
若无则 .v)解析端口位宽,从 .lef 解析几何面积,生成与 PDK 单位制一致
(time=1ps, cap=1fF, 0.7V)的占位 NLDM。

时序占位值沿用 fakeram 口径(synth/README.md):tcq=218ps、tsu=th=50ps。
拿到 memory compiler 实测数据后,替换下方 TIMING 常量即可。

用法: python3 gen_sram_lib.py [--macros-dir DIR] [--out DIR]
"""
import argparse
import re
from pathlib import Path

# ── 占位时序(单位 ps / fF;流片前需以实测替换) ─────────────────────
TIMING = dict(
    tcq_ps=218.0,        # clk→输出 延迟(fakeram 占位值 0.218 ns)
    tran_ps=80.0,        # 输出转换时间
    tsu_ps=50.0,         # 输入建立
    th_ps=50.0,          # 输入保持
    cin_ff=2.0,          # 输入引脚电容
    max_cap_ff=250.0,    # 输出引脚最大负载
    max_tran_ps=160.0,   # 与 PDK default_max_transition 一致
)


def parse_ports(lef: Path, macro: str):
    """从 LEF 解析宏引脚(权威来源,保证 liberty 与 LEF 逐 pin 一致)。

    返回 [(name, dir, msb|None)]:总线位以 name[i] 形式出现在 LEF 中,
    按基名聚合,msb = 最大下标。
    """
    txt = lef.read_text()
    m = re.search(rf"MACRO\s+{re.escape(macro)}\b(.*?)END\s+{re.escape(macro)}",
                  txt, re.S)
    if not m:
        raise RuntimeError(f"{lef}: 找不到 MACRO {macro}")
    body = m.group(1)
    buses = {}   # base -> [dir, msb]
    for pm in re.finditer(
            r"PIN\s+(\S+)\s*(.*?)END\s+\1", body, re.S):
        name, pbody = pm.group(1), pm.group(2)
        if re.search(r"USE\s+(POWER|GROUND)", pbody):
            continue
        dm = re.search(r"DIRECTION\s+(\w+)", pbody)
        d = (dm.group(1).lower() if dm else "input")
        d = {"inout": "input"}.get(d, d)
        bm = re.match(r"([A-Za-z_]\w*)\[(\d+)\]$", name)
        if bm:
            base, idx = bm.group(1), int(bm.group(2))
            ent = buses.setdefault(base, [d, idx])
            ent[1] = max(ent[1], idx)
        else:
            buses.setdefault(name, [d, None])
    return [(n, d, msb) for n, (d, msb) in buses.items()]


def parse_lef_size(lef: Path, macro: str):
    """从 LEF 抓 MACRO 的 SIZE W BY H。"""
    txt = lef.read_text()
    m = re.search(rf"MACRO\s+{re.escape(macro)}\b.*?SIZE\s+([\d.]+)\s+BY\s+([\d.]+)",
                  txt, re.S)
    if not m:
        raise RuntimeError(f"{lef}: 找不到 MACRO {macro} 的 SIZE")
    return float(m.group(1)), float(m.group(2))


def emit_lib(name: str, ports, area_um2: float) -> str:
    t = TIMING
    L = []
    L.append(f"""/* 占位 NLDM — 由 pr/scripts/gen_sram_lib.py 生成。
 * 时序为 fakeram 占位值(tcq={t['tcq_ps']}ps, tsu/th={t['tsu_ps']}ps),
 * 单位制与 3nm_GAA_FSPR_rvt_nldm.lib 对齐(1ps / 1fF / 0.7V)。
 * 注意:未建 SRAM 内部功耗模型 —— report_power 中宏内部功耗为 0,
 * A/B 策略对比不受影响(两侧同样低估),绝对功耗需宏实测数据。 */
library ({name}_lib) {{
  delay_model : table_lookup;
  time_unit : "1ps";
  capacitive_load_unit (1,ff);
  current_unit : "1mA";
  leakage_power_unit : "1pW";
  pulling_resistance_unit : "1kohm";
  voltage_unit : "1V";
  nom_process : 1; nom_temperature : 25; nom_voltage : 0.7;
  operating_conditions (PVT_0P7V_25C) {{ process : 1; temperature : 25; voltage : 0.7; }}
  default_operating_conditions : PVT_0P7V_25C;
  voltage_map (VDD, 0.7); voltage_map (VSS, 0);
  default_max_transition : {t['max_tran_ps']};

  cell ({name}) {{
    area : {area_um2:.3f};
    is_macro_cell : true;
    interface_timing : true;
    dont_touch : true;
    dont_use : true;
    pg_pin (VDD) {{ pg_type : primary_power;  voltage_name : VDD; }}
    pg_pin (VSS) {{ pg_type : primary_ground; voltage_name : VSS; }}""")

    def bus_or_pin(pname, msb):
        return (f"bus ({pname}) {{\n      bus_type : bus_{msb + 1};\n", "    }")

    bus_types = set()
    for pname, d, msb in ports:
        if pname in ("VDD", "VSS"):
            continue
        indent = "    "
        if msb is not None:
            bus_types.add(msb + 1)
            L.append(f"{indent}bus ({pname}) {{")
            L.append(f"{indent}  bus_type : bus_{msb + 1};")
            inner = indent + "  "
        else:
            L.append(f"{indent}pin ({pname}) {{")
            inner = indent + "  "
        if pname == "clk":
            L.append(f"{inner}direction : input; clock : true; capacitance : {t['cin_ff']};")
        elif d == "input":
            L.append(f"{inner}direction : input; capacitance : {t['cin_ff']};")
            # 输入相对 clk 的建立/保持约束
            L.append(f"{inner}timing () {{ related_pin : \"clk\"; timing_type : setup_rising;")
            L.append(f"{inner}  rise_constraint (scalar) {{ values(\"{t['tsu_ps']}\"); }}")
            L.append(f"{inner}  fall_constraint (scalar) {{ values(\"{t['tsu_ps']}\"); }} }}")
            L.append(f"{inner}timing () {{ related_pin : \"clk\"; timing_type : hold_rising;")
            L.append(f"{inner}  rise_constraint (scalar) {{ values(\"{t['th_ps']}\"); }}")
            L.append(f"{inner}  fall_constraint (scalar) {{ values(\"{t['th_ps']}\"); }} }}")
        else:  # output: clk→Q 传播延迟
            L.append(f"{inner}direction : output; max_capacitance : {t['max_cap_ff']};")
            L.append(f"{inner}timing () {{ related_pin : \"clk\"; timing_type : rising_edge;")
            L.append(f"{inner}  cell_rise (scalar) {{ values(\"{t['tcq_ps']}\"); }}")
            L.append(f"{inner}  rise_transition (scalar) {{ values(\"{t['tran_ps']}\"); }}")
            L.append(f"{inner}  cell_fall (scalar) {{ values(\"{t['tcq_ps']}\"); }}")
            L.append(f"{inner}  fall_transition (scalar) {{ values(\"{t['tran_ps']}\"); }} }}")
        L.append(f"{indent}}}")
    L.append("  }")
    L.append("}")

    # bus_type 定义插到 library 头部之后
    types = "\n".join(
        f"  type (bus_{w}) {{ base_type : array; data_type : bit;\n"
        f"    bit_width : {w}; bit_from : {w - 1}; bit_to : 0; downto : true; }}"
        for w in sorted(bus_types))
    out = "\n".join(L)
    return out.replace("  cell (", types + "\n\n  cell (", 1)


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve()
    ap.add_argument("--macros-dir", type=Path,
                    default=here.parents[2] / "synth" / "macros")
    ap.add_argument("--out", type=Path, default=here.parents[1] / "macros_lib")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    for mdir in sorted(args.macros_dir.iterdir()):
        name = mdir.name
        lef = mdir / f"{name}.lef"
        if not lef.exists():
            continue
        w, h = parse_lef_size(lef, name)
        ports = parse_ports(lef, name)
        lib = emit_lib(name, ports, w * h)
        out = args.out / f"{name}.lib"
        out.write_text(lib)
        print(f"[gen_sram_lib] {out}  ({len(ports)} ports, area={w * h:.0f} um^2)")


if __name__ == "__main__":
    main()
