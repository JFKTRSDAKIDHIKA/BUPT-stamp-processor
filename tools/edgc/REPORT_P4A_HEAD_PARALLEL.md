# Priority 4-A — Head-Parallel Attention Tiling（decode）

Branch: `opt/p1-barrier-elim` · 日期：2026-07-10 · 数据：`results/p4a_head_parallel.csv`
复现：`python3 tools/edgc/bench_attention.py --sim build/src/trace_sim`

> 纯时序。decode 阶段真实注意力形状（真实 n_heads / d_head / context）。

---

## 1. 问题与做法

decode 每个 head 的注意力很小：QK^T = (1×d_head)·(d_head×L)，PV = (1×L)·(L×d_head)。
若把所有 head **串行放在一个 tile** 上，16 个 tile 只有 1 个在动（利用率 1/16）。

**Head-parallel（Strategy A）**：head h → tile (h mod 16)，各 head 相互独立、**无需
tile 间通信**，最多让 min(n_heads,16) 个 tile 同时算。实现是编译期的 tile 指派——把
每个 head 的 QK^T + softmax + PV 整段发到它自己的 tile。

---

## 2. 实测（decode，M=1 token，L=512）

| 配置 | heads | d_head | 单 tile: 活跃/cyc | head-parallel: 活跃/cyc | 加速 | tile 利用率 |
|:--|--:|--:|--:|--:|--:|--:|
| Llama-1B   | 32 | 64  | 1 / 520 055 | **16** / 95 181  | **5.46×** | 6.25% → **100%** |
| Llama-3B   | 24 | 128 | 1 / 644 766 | **16** / 165 041 | 3.91× | 6.25% → **100%** |
| Mistral-7B | 32 | 128 | 1 / 859 630 | **16** / 195 518 | 4.40× | 6.25% → **100%** |
| edge (4h)  |  4 | 16  | 1 / 34 501  |  **4** / 9 099   | 3.79× | 6.25% → **25%** |

**tile 利用率从 ~1/16 提升到 num_heads/16**（n_heads≥16 → 填满 16 个 tile；n_heads=4
→ 4/16，受 head 数上限约束，正是 spec 说的 "limited by num_heads"）。causality 通过。

---

## 3. 为什么加速是 ~4–5× 而非 16×

Llama-1B 有 32 head，head-parallel 后每 tile 处理 2 个 head，理想相对单 tile 应 ~16×。
实测 5.46×，因为**填满阵列后瓶颈从"tile 空置"转移到"垂直键合带宽"**：每个 head 要从
DRAM 载入 K、V（各 L×d_head），16 个 tile 同时抢 4 条分组垂直键合 → 访存受限。所以
head-parallel 把 decode 注意力从**计算并行不足**推到**访存带宽**这道墙——利用率满了，
下一步得靠更宽键合 / KV 复用 / 更低精度 KV。

这与 Tier-1 报告一致：decode 整体是访存受限；head-parallel 解决了注意力这一段的 tile
空置，但没改变访存是最终天花板这一事实。

---

## 4. 对照 spec 成功标准

| 指标 | 目标 | 实测 |
|:--|:--|:--|
| tile utilization 从 ~1/16 → 接近 num_heads/16 | 必须 | ✅ 6.25% → 100%（n_heads≥16）/ 25%（n_heads=4）|
| 时序正确（causality） | 必须 | ✅ 通过 |
| 加速 | — | 3.8–5.5×（受垂直键合带宽限）|

## 5. 与其他策略的关系（本轮只做 A）

- **Strategy A（head-parallel）**：已实现。上限 = min(n_heads,16) 个 tile。对
  n_heads<16 的模型（如 GQA 的 KV 侧、或 head 少的小模型）填不满——需 B/C 补。
- **Strategy B（K-split）/ C（multi-GEMM packing）**：本轮未做（decode 完整代码路径按
  约定搁置）。当 n_heads<16 或要进一步切分单个大 head 时用得上。
