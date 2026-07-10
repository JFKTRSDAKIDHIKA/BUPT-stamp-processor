# MOBOL 三层堆叠 PR 实验框架(策略 A/B 对比)

面向 MOBOL 加速器(`ARCH_SPEC.md`)的多架构变体物理实现与
**混合键合(HB)重分布策略 A/B 对比**的全自动 OpenROAD 流程。

```
Top    DRAM die(16 通道 HBM3 级,不做 PR,只建模其 HB 落点)
Mid    buffer die : 4×8MB SSP bank(代表实现:ssp_bank = 1 bank)
Base   base die   : 16 tile + 环网(代表实现:noc_tile_sys)
```

- **策略 A `no_redist`(直穿)**:DRAM 通道经 HB 直穿 buffer die 落到
  base die 核心区 → 两个 die 的核心区都被 KOZ(禁布区)切碎。
- **策略 B `edge_redist`(边缘重分布)**:buffer die 用 M8/M9 高层金属把
  DRAM 信号重分布到 die 边缘,base die 的全部 HB(tile 链路 + DRAM 通道)
  只出现在边缘带,**核心区零 KOZ**;代价是 buffer die 的边缘布线通道。

## 目录结构

```
pr/
├── config/experiments.yaml       实验矩阵配置(工艺假设/die 绑定/流程旋钮)
├── scripts/
│   ├── arch_model.py             Task 1:变体 → 物理模型(面积/HB 数量/几何)
│   ├── gen_runs.py               Task 2/3:展开矩阵 → runs/<id>/run_cfg.tcl
│   ├── gen_sram_lib.py           SRAM 宏占位 Liberty(fakeram 的 .lib 缺失)
│   ├── collect_results.py        Task 4:metrics/日志/拥堵 → CSV + 对比表
│   ├── plot_layout.py            出图:floorplan.png(布局规划)+ layout.png
│   │                             (DEF 物理版图;本机无 GUI/KLayout,matplotlib 渲染)
│   └── openroad/
│       ├── pr_main.tcl           PR 主流程(共用;读 run_cfg.tcl)
│       ├── floorplan_hb.tcl      策略 A/B 的全部几何:宏阵列、KOZ、
│       │                         面阵 HB 引脚(up: 区域)、重分布通道
│       └── area_probe.tcl        网表精确面积探测(结果缓存)
├── macros_lib/                   生成的宏 .lib
├── runs/<die>__<变体>__<策略>/    每个 run 的配置/日志/DEF/ODB/指标
└── out/                          arch_model.json / results.csv / summary.md
```

## 使用

```bash
cd pr
./run_all.sh --smoke              # 冒烟:tile_node × baseline × A/B(约 15 分钟)
./run_all.sh                      # 全矩阵:5 变体 × 2 策略 × {base_die, buffer_die}
./run_all.sh -j 2                 # 2 路并行(每 run 16 线程,注意内存)
./run_all.sh --dies buffer_die --variants baseline dense8192
python3 scripts/arch_model.py     # 只看 Task-1 物理模型表(不跑 PR)
python3 scripts/collect_results.py  # 随时重新汇总(跑批中途也可)
python3 scripts/plot_layout.py      # 出图(跑批中途也可,只画有 DEF 的 run)
```

图片输出:每个 run 目录下 `floorplan.png`(布局规划图:die/core、宏阵列、
KOZ 簇按深度着色、重分布通道、边缘带)与 `layout.png`(物理版图:DEF 真实
标准单元/宏/分层布线/M9 HB 面阵引脚);`out/plots/` 下每 (die×变体) 一张
A/B 并排对比图。

- 断点续跑:完成的 run 会落 `DONE` 标记,重复执行自动跳过;`--force` 重跑。
- 变体真值源是 `config/mobol_arch.yaml` 的 `compiler.dse`
  (baseline / W2 / nmc / nmc+W2 / dense8192),新增变体无需改任何脚本。
- 路径可用环境变量覆盖:`MOBOL_ROOT`、`PDK_ROOT`。

## 物理建模要点(与 ARCH_SPEC §7/§8 对应)

| 对象 | 建模 |
|---|---|
| tile↔bank 链路 | 16 条(每 tile 一条),位宽 = W×512b×2向 + 16 控制;W 随变体(baseline=1,W2=2) |
| DRAM 通道 | 16 个(DRAM die 16 通道),位宽 = 4列×密度/16×2向 + 控制;密度随变体(2048/8192) |
| HB pad | 2µm 间距方阵簇,0.8µm pad,KOZ 外扩 2µm(`experiments.yaml: tech`) |
| KOZ 深度 | `hard`=顶两层+禁布;`punch`=全层直穿(A 的 buffer die);`shallow`=仅 M8/M9 削减(B 的 DRAM 顶面落点) |
| HB 引脚 | OpenROAD 面键合机制:`define_pin_shape_pattern` + `up:{…}` 区域约束,真实落 M9 面阵 |
| 重分布通道 | 宽度 = 位宽×0.064µm ÷ 通道利用率 0.5(**保证策略 B 不把 buffer die 自己堵死**);通道外 M8/M9 GRT 容量收紧 → 等效 `add_routing_guide` 把长线导入通道 |
| 宏阵列 | 参数化摆放(tile SRAM 按 tile 网格 / bank SRAM 按 bank 阵列);策略 A 的 KOZ 簇落宏间通道交叉处 |
| die 尺寸 | 同 (die×变体) 下 A/B 共用同一 die 尺寸(取两策略需求的较严者)→ 公平对比 |
| NMC 变体 | buffer die 为 NMC+预取引擎预留面积(netlist 未集成,见下"诚实性") |

## 输出指标(`out/summary.md` 每格 A vs B vs Δ)

- 时序:WNS / TNS / 时钟偏斜(detailed route 后,GRT 寄生)
- 面积:die 面积、单元面积、利用率、KOZ 占核心区比例
- 布线:GRT 资源利用率与总溢出、**核心区 vs 边缘带拥堵热点分区统计**
  (来自 `congestion.rpt` 坐标 × `clusters.json` 几何 —— 直接回答
  "B 是否清空了 base 核心区拥堵、边缘通道是否成为新瓶颈")
- 功耗:report_power 总功耗(注意宏内部功耗未建模,见下)

## 3nm 工艺下的局限与 DRC 建议(重要)

1. **简化 PDK**:USC-3N-2D 只有基础 WIDTH/PITCH/SPACING,没有 3nm 真实
   LEF58 规则(SADP 着色、EUV 最小面积、cut 阵列间距)。OpenROAD 的 DRC
   数只能做 **A/B 同口径趋势比较**;签核 DRC 需厂商 runset。
2. **无 filler/tap/tie/天线二极管**:密度填充、阱连续、天线修复缺席;
   常量网(tie-high `ce_in` 等)被标 special 跳过布线,流片前需补 tie cell。
3. **单角 NLDM、全库 area:0**:hold 修复在该组合下会崩溃已禁用
   (只修 setup);hold/skew 数字仅作趋势。SRAM 宏 Liberty 为占位时序
   (tcq=218ps),**宏内部功耗为 0** —— 功耗绝对值偏低,A/B 差值可信。
4. **无 GDS**:本 OpenROAD build 无 `write_gds` 且 PDK 不带单元 GDS。
   输出 DEF/ODB;拿到单元 GDS 后:
   `klayout -zz -rd design=<run>.def -rd out=<run>.gds -rm def2gds.py`。
5. **诚实性声明**:W2/密度变体改变的是 **HB 接口规模与 KOZ/通道几何**,
   netlist 仍是 baseline 综合结果(W=2 的宽端口需重综合);nmc 变体只预留
   引擎面积。这正是本实验的目的 —— 隔离"重分布策略×接口规模"的物理影响。
