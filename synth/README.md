# MOBOL synth/ — SRAM 宏化 + Yosys 逻辑综合流程

面向 Yosys（逻辑综合）+ OpenROAD（后端）的自动化流程：把 RTL 中的行为级
SRAM 替换为 fakeram 生成的宏模型（Verilog / Liberty / LEF），并综合到
USC-3N-2D（SCCAD 3 nm GAA）标准单元。

```
synth/
├── fakeram/gen_sram_macros.py   fakeram 配置生成 + 批量调用 + 2R1W 扩展视图
├── fakeram/mobol_sram.cfg       生成的 FakeRAM2.0 输入配置（USC-3N-2D 工艺参数）
├── macros/<name>/               生成的 .lib / .lef / .v / .bb.v
├── pdk/                         指向 USC-3N-2D OpenROAD 视图的符号链接（原路径含空格）
├── filelists/<top>.f            每个综合 top 的文件清单（行为级/宏级二选一）
├── yosys/synth.ys.in            Yosys 综合脚本模板（run_synth.sh 实例化）
├── run_synth.sh                 sv2v → Yosys 一键综合driver
├── verif/                       行为级 vs 宏级 逐周期等价性 TB（Verilator）
├── scripts/sram_audit.py        SRAM 实例审计（防止行为级存储混入综合清单）
└── build/<top>/                 design.v / synth.ys / yosys.log / 报告 / 门级网表
```

## SRAM 清单（来自 ARCH_SPEC + config/mobol_arch.yaml）

| 存储 | 规格 | 端口 | RTL 现状 | 宏化方案 |
|:--|:--|:--|:--|:--|
| LOCAL scratchpad (LSP) | 256 KB/tile = 4096 × 512 b | 2R1W，同步读 1 拍，写同址读旧值 | `sram_2r1w` 行为模型，`tile_top`/`tile_node` 实例化 | 1 × `mobol_sram_4096x512_2r1w`（drop-in wrapper） |
| SHARED bank (SSP) | 8 MB/bank × 4 = 131072 × 512 b | 同 2R1W 口径（SYNTH_NOTES §1） | 无 RTL 实例（NMC/prefetch 经 valid/ready 通道） | 新模块 `rtl/buffer/ssp_bank.sv` = 32 × 同款宏 |
| 其余（MXU/VPU 操作数寄存器、join 计数器、NoC 缓冲） | ≤ 数 KB | — | 寄存器堆 | 保持触发器实现（不宏化） |

单一宏几何：`mobol_sram_4096x512_2r1w`，471.9 × 294.9 µm（0.139 mm²/256 KB），
全芯片 LSP(16) + SSP(128) = 144 实例。

## 关键决策

1. **FakeRAM2.0 只支持 1RW**（`class_memory.py` 硬编码 `rw_ports=1`），而 RTL
   需要 2R1W。`gen_sram_macros.py` 复用 FakeRAM2.0 的工艺/面积模型
   （`Process` + `get_macro_dimensions`），额外产出 2R1W 的 .lib/.lef/.v/.bb.v；
   面积按 1RW 阵列 ×2 计（双副本复制——真实 memory compiler 构造 2R1W 的
   常规方式）。同时保留 stock 1RW 宏做参考。时序数字沿用 FakeRAM2.0 的占位
   值（tcq 0.218 ns 等），拿到实测数据后在脚本顶部替换。
2. **替换 = 文件清单切换，零 RTL 改动**：`rtl/sram/sram_2r1w_macro.sv` 与行为
   模型同名同参同端口（内部例化宏）。仿真清单用行为版，综合清单用宏版；
   `scripts/sram_audit.py` 保证二者绝不混编。
3. **接口时序完全一致，无需桥接逻辑**：两者都是同步写 + 1 拍同步读 +
   写同址读旧值。等价性由 `make equiv` 逐周期证明（2 万随机周期、高冲突
   地址流，覆盖 LSP wrapper 与 ssp_bank 组合两条路径）。
4. **PDK 无异步复位触发器**（只有 DFFHx1/DFFHQNx1），综合脚本用
   `async2sync + dfflegalize -cell $_DFF_P_ 01` 把 RTL 的异步 rst_n 折叠为
   同步逻辑。门级网表的复位行为是同步的——门级仿真时复位需保持至少 1 拍。
5. **Yosys 前端不认 unpacked-array 端口**（mxu16/vpu16 等），统一经 sv2v
   预转换；另外 `logic'()` 类型转换 Yosys 不支持，已把 fp32_mul/fp32_add 中
   3 处改写为等价位选（tb_fp 117 万次比对、tb_mxu 300 例回归通过）。
6. **fakeram 行为模型的写 OR bug**（`mem[addr] <= wd | mem[addr]`，写掩码
   注释残留）由生成脚本自动修补，否则宏仿真必错。

## 综合覆盖矩阵（8 个 top，全部经 QC 硬检查）

| Top | 层级 | 内容 | 单元数（含子模块） | DFF（展平） | SRAM 宏（展平） |
|:--|:--|:--|--:|--:|--:|
| `noc_tile_sys` | **系统级（base die fabric）** | 16 × tile_node + 双向环 NoC（32 路由器、2 VC） | 476k | 44 224 | 16（2.23 mm²） |
| `tile_top` | tile 集成 | Sequencer FSM + MXU(16³) + join 计数器 + LSP | 2.76M | 54 034 | 1 |
| `tile_node` | tile 集成 | LSP + DMA 引擎 + NoC 注入/弹出 | 5.2k | 564 | 1 |
| `ssp_bank` | buffer die | 8 MB SHARED bank（32 宏 + 解码/复用） | 39k | 10 | 32（4.45 mm²） |
| `vpu16` | 计算单元 | 16-lane f32 VPU（含 exp2/recip/rsqrt 超越通路） | 2.72M | 8 714 | 0 |
| `nmc_engine` | buffer die | 近存归约引擎（256 宽 f32 加法阵列 + 控制） | 768k | 8 204 | 0 |
| `prefetch_engine` | buffer die | 权重流式预取 DMA（2D strided 地址生成） | 12k | 103 | 0 |
| `ring_noc` | 互连 | 独立环网（含 ring_router 缓冲/仲裁） | 393k | 35 200 | 0 |

展平计数来自 `scripts/count_cells.py`（网表层次 DAG 展开；`stat`/grep 只给
每模块体内计数，对层次网表会严重低估——PnR 宏清点必须用展平数）。

每个 `build/<top>/` 含：`design.v`（sv2v 后 RTL）、`synth.ys`、`yosys.log`、
`synth_stat.txt`（含 SRAM 宏面积追加）、`<top>.gate.v`（门级网表）、
`<top>.sdc`（OpenROAD 约束，500 MHz/2000 ps）。

**QC 硬门槛**（run_synth.sh 内置，任一不过即退出非零）：
0 个未映射 generic cell、0 个锁存器（DHLx1）、0 个 X 字面量、
`yosys check` 0 problems。

**集成边界（如实说明）**：RTL 中不存在单一 `base_die_top`——
`tile_top`（计算集成）与 `noc_tile_sys`（fabric 集成）是两个已被
Verilator TB 验证的最大集成点；把 MXU/VPU/join 挂进 tile_node 属于
RTL 集成设计工作（sequencer 与 DMA/NoC 端口仲裁），不应在综合阶段
临时拼接未验证的胶水逻辑。对 3D 堆叠芯片，推荐的 PnR 组织方式本来
就是分层：tile / bank 分别 PnR 成硬宏，再做 die 级组装。

## 使用

```bash
cd synth
make macros            # 重新生成 SRAM 宏视图（改几何/时序参数后）
make equiv             # Verilator 等价性验证（行为级 vs 宏级）
make synth             # Yosys 综合 tile_top
./run_synth.sh <top>   # tile_top | tile_node | ssp_bank
python3 scripts/sram_audit.py   # 审计 SRAM 使用与文件清单
python3 scripts/count_cells.py build/<top>/<top>.gate.v <top> [type…]
                                # 展平实例计数（宏/DFF/锁存器）
```

## 对接 OpenROAD

- 网表：`build/<top>/<top>.gate.v`
- 标准单元 LEF/lib：`pdk/stdcell.lef` `pdk/stdcell.lib` `pdk/tech.lef`
- SRAM 宏 LEF/lib：`macros/mobol_sram_4096x512_2r1w/…{lef,lib}`
- SDC 起点见 `rtl/SYNTH_NOTES.md` §6（500 MHz，clk 周期 2 ns）；abc 已按
  2 ns（`-D 2000`，lib 时间单位 1 ps）约束映射。
- **注意**：USC-3N-2D 的 .lib 中全部标准单元 `area : 0`（面积只在 LEF 中），
  所以 `stat -liberty` 只有单元计数没有逻辑面积——面积/利用率要在 OpenROAD
  读 LEF 后统计；SRAM 宏面积已由 run_synth.sh 追加到报告末尾。
- 注意 FP 超越函数/MXU MAC 的流水化建议（SYNTH_NOTES §2-3）仍然成立，
  500 MHz 时序收敛前置条件。
