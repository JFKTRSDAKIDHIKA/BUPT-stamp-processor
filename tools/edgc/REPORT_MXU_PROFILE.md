# MXU 利用率 Profiling 报告 — base die Compute Tile

日期：2026-07-10 · 工具：`src/cycle/` 周期精准模拟器 + `tools/edgc/` 编译器
调度：`baseline`（tile_link=1, wr=1, rd=2, dma_rate=1, dram_density=2048）
DRAM：`config/ramulator_3d_dram.yaml`（HBM3 级，Ramulator2）

复现：`python3 tools/edgc/profile_mxu.py --sim build/src/trace_sim`

---

## 1. 结论（TL;DR）

**在现有的 edge-LLM 工作负载上，base die 16 个 tile 的 MXU 芯片级占用率只有
0.66%–0.82%，即便是最忙的单个 tile 的 MXU 也只占用约 2.5% 的时间。MXU 完全
不是瓶颈——它几乎一直空闲。** 瓶颈是**全局 barrier 同步**（占总周期预算的
69%–78%），叠加两个结构性问题：**一半 tile（T8–T15）根本不跑 MXU**，以及
**每个 MXU 算子都被 DMA 搬运 + fence + wait 串行包裹**，喂不满 4 级流水。

这符合预期：这些负载是**功能验证用的微缩模型**（seq=16, d_model=64），
用途是逐位校验编译器与模拟器的正确性，而非吞吐。它们的 GEMM 太小，无法摊薄
同步/搬运开销，也填不满 tile 阵列。

---

## 2. 实测数据

### 2.1 MXU 占用率

| 模型 | 逻辑拍 | MXU ops | 芯片 MXU 占用 | 跑 MXU 的 tile 数 | 最忙 tile 占用 |
|:--|--:|--:|--:|--:|--:|
| edge_dense   | 10 387 | 272 | **0.655%** | 8/16 | 2.31% |
| edge_gqa     | 11 055 | 304 | **0.687%** | 8/16 | 2.46% |
| edge_mqa     | 10 884 | 288 | **0.662%** | 8/16 | 2.50% |
| edge_sliding | 11 055 | 304 | **0.687%** | 8/16 | 2.46% |
| edge_moe     | 26 983 | 880 | **0.815%** | 8/16 | 2.43% |
| spec_target  | 22 316 | 608 | **0.681%** | 8/16 | 2.44% |

占用率定义（同 `chip.cpp`）：`Σ mxu_busy_cycles / (16 × total_cycles)`。
`mxu_busy_cycles = mxu_ops × mxu_latency(4)`，即 MXU 流水里有 ≥1 个算子的拍数。

### 2.2 停顿归因（全芯片 cycle-sum 占 `16 × 拍数` 预算的比例）

| 模型 | issue(busy) | **acquire 停顿** | dma 停顿 | mxu_busy | vpu_busy | mxu 停顿 | vpu 停顿 |
|:--|--:|--:|--:|--:|--:|--:|--:|
| edge_dense   | 2.07% | **77.94%** | 16.06% | 0.66% | 3.54% | 0.49% | 3.40% |
| edge_gqa     | 2.31% | **75.20%** | 18.39% | 0.69% | 3.71% | 0.52% | 3.55% |
| edge_mqa     | 2.29% | **77.04%** | 16.60% | 0.66% | 3.70% | 0.50% | 3.54% |
| edge_sliding | 2.31% | **75.20%** | 18.39% | 0.69% | 3.71% | 0.52% | 3.55% |
| edge_moe     | 2.33% | **69.15%** | 24.34% | 0.82% | 3.74% | 0.61% | 3.55% |
| spec_target  | 2.28% | **75.67%** | 18.00% | 0.68% | 3.67% | 0.51% | 3.51% |

阵列平均只有 ~2.3% 的拍在发射指令，其余 ~97% 在 stall——其中绝大部分是
`stall_acquire`（在 barrier 上等）。

---

## 3. 根因分析

### 根因 A：一半阵列是暗的（结构上限先砍一半）
输出块驻留（output-block-stationary）GEMM 映射把输出块按 `idx % 16` 轮转到
tile。但这些模型的 GEMM 太小：seq=16→1 个行块，投影维度 64→4 列块、FFN
128→8 列块，所以任何一个 GEMM 最多产生 8 个输出块，全部落到 T0–T7。以
edge_gqa 为例，每 tile 的 MXU 算子分布：

```
T0=68 T1=68 T2=52 T3=52 T4=16 T5=16 T6=16 T7=16 T8..T15=0
```

**T8–T15 永远不碰 MXU。** 芯片级占用率的理论上限一开始就被砍到 50%。

### 根因 B：全局 barrier 串行化（停顿主因，~75%）
编译器每个原语（gemm / norm / rope / residual / softmax / silu…）结尾都插一次
**全局 barrier**：16 个 tile 全部 RELEASE 到 tile0，tile0 ACQUIRE arity=16，再
广播回落。整个程序被切成几十个 barrier 相隔的相位，每相位以**最慢 tile**为准
推进，先算完的 tile 全卡在 `stall_acquire`。

| 模型 | barrier 数 |
|:--|--:|
| edge_dense | 24 |
| edge_gqa / mqa / sliding | 30 |
| spec_target | 60 |
| edge_moe | 62 |

因为跑 MXU 的只有 8 个 tile、每相位算子又少，barrier 到来时大部分 tile 早已
空转等待——这就是 69%–78% 的 acquire 停顿来源。

### 根因 C：每个 MXU 算子被 DMA + fence + wait 串行包裹
GEMM 内层循环对每个 K 步都：DMA 载入 A 块、DMA 载入 B 块、`DMA_FENCE`、发一条
**4 拍**的 MXU、`WAIT_MXU`。算子数量对比（edge_gqa）：

```
MXU_F16F16=304   DMA=848   DMA_FENCE=856   WAIT_MXU=304
```

即每 1 个 MXU 算子对应 ~2.8 次 DMA + ~2.8 次 fence。MXU 流水本可 II=1 每拍
吃一个算子，实际每几十拍才喂一个（要等 DRAM 往返 + fence + wait）。所以哪怕
在最忙的 T0/T1 上，MXU 占用也只有 ~2.5%——**流水一直饿着**。

---

## 4. 把 0.7% 拆开看

芯片 MXU 空闲的 ~99.3% 大致这样分掉：

1. **~50%**：T8–T15 结构性暗置（根因 A），直接没了一半上限。
2. **活跃那一半里**：~75% 的预算耗在 barrier 等待（根因 B），只有 ~2.3% 的拍
   在发指令。
3. **发指令的拍里**：MXU 又被 DMA/fence/wait 稀释（根因 C），最忙 tile 峰值
   占用仅 2.5%。

三层相乘 → 芯片级 ~0.7%。

---

## 5. 判断与建议

- **这些数字不代表架构的 MXU 吞吐能力**，只说明微缩验证负载喂不动 MXU。要测
  真实 MXU 利用率，需要 GEMM 维度足够大（输出块 ≥16、K 足够长）来填满 16 tile
  并摊薄 barrier/DMA 开销——本轮按用户要求用现有负载，未新增大模型。
- 若要在**现有负载**上提高 MXU 占用，杠杆按收益排序：
  1. **减少 barrier**：把逐原语的全局 barrier 换成生产者-消费者点对点同步
     （只在真正有数据依赖的 tile 间 release/acquire），可直接吃掉 ~75% 的
     acquire 停顿。
  2. **double-buffer GEMM 内层**：预取下一 K 步的 A/B 块，去掉每步的
     `DMA_FENCE`/`WAIT_MXU` 串行点，让 MXU 连续累加（output-stationary 的在飞
     转发已支持 `acc=1`，缺的是操作数预取）。
  3. **填满阵列**：小 GEMM 换更细的并行切分（如按 K 或 batch 切到 T8–T15），
     或把多个小 GEMM 打包并行。

---

## 6. 附：本轮为跑通链路所做的修复

1. `src/cycle/flit.h`：补 `#include <stdexcept>`（`std::logic_error` 在
   Linux/libstdc++ 下不再随其它头文件间接可见，macOS 上曾侥幸编过）。
2. Ramulator2 `generic_dram_controller.cpp`：外部**写**请求在发出末条 WR 命令
   后触发其 callback（原版只处理读的完成回调，写请求的 callback 永不触发→
   `trace_sim` 里所有 DRAM 写的 `WRITE_ACK` 永不返回→仿真死锁）。修复后
   `ctest` 6/6 全过，5 个 edge 模型 golden==ref==sim 逐位校验全过。
3. `src/trace_sim.cpp`：`--json` 增加 per-tile 明细（instrs / busy / 各类
   stall / mxu_ops / mxu_busy / vpu_busy / dma_bytes），供 profiling 用。
