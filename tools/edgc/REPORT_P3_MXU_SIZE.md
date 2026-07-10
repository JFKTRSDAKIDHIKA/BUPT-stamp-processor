# Priority 3 — MXU 尺寸分析（analytical DSE）

Branch: `opt/p1-barrier-elim` · 日期：2026-07-10
数据：`results/p3_mxu_size_dse.csv` · 图：`results/p3_mxu_util.png`
复现：`python3 tools/edgc/mxu_dse.py --csv ... --plot ...`

> 纸面分析（不改模拟器硬件参数）。从 Tier-1 真实模型提取所有 GEMM 的 M/N/K，
> 对 MXU 尺寸 S ∈ {8,16,32} 计算 tile-pass 数与利用率。

定义：`passes = ⌈M/S⌉·⌈N/S⌉·⌈K/S⌉`；
`util = (真实 M·N·K MAC) / (passes · S³ 峰值 MAC 槽)`，按层内各 GEMM 的 MAC 量加权。

---

## 1. 结论（一句话）

**16×16×16 对 edge decode 是正确尺寸**：在 batch=16（现实可达的 decode 批量）达到
100% 利用率，而 32×32×32 在 batch=16 只有 50%（要 batch=32 才填满，边端通常没有那么
多并发 token）；8×8×8 虽然在小 batch 时利用率更高，但达到同样 100% 需要 **~8× 的
tile-pass 数**（控制/发射/同步开销翻数倍）。**单 token（M=1）时利用率恒为 1/S，谁都
救不了——正解是 batching（P4），不是改 MXU 尺寸。**

---

## 2. Decode 利用率 vs. MXU 尺寸（ctx=512）

| batch M | 8×8×8 | 16×16×16 | 32×32×32 |
|--:|--:|--:|--:|
| **1** (单 token) | 12.5% | **6.2%** | 3.1% |
| **16** | 100% | **100%** | 50% |
| **32** | 100% | **100%** | 100% |

（所有 5 个模型数值相同——利用率只由 M 维 padding 决定，与 d_model/d_ffn/层数无关。）

- **M=1**：util = 1/S 精确成立（12.5=1/8, 6.25=1/16, 3.1=1/32）。decode 每步 M=1，MXU
  的 S×S 输出块只有 1 行是真 token，其余 S−1 行是 padding → 利用率 = 1/S。**越小的
  MXU 浪费越少**。
- **M=16**：8³ 与 16³ 都满（16 是二者的整数倍），**32³ 只有 50%**（16<32，半空）。
- **M=32**：三者全满。

图 `p3_mxu_util.png`：每种 MXU 在 **batch = S** 处饱和（8³@M8、16³@M16、32³@M32）——
这就是 M-padding 的阶梯。

---

## 3. tile-pass 数（控制/发射开销）

同一 GEMM，越小的 S → 越多 pass（∝ 1/S³）。以 16×16×16 为基准（Llama-1B 单层
decode ≈ 245,760 passes）：**8×8×8 约 8× passes，32×32×32 约 1/8**。所以：

- 8³ 在 M=1 的利用率优势（12.5% vs 6.25%）是**用 8× 的发射/同步开销换来的**——在
  访存受限的 decode 里（MXU 本就闲，见 Tier-1 报告 util 1–4%）这点计算利用率提升
  不划算。
- 32³ 在 prefill（M=seq 很大）几乎总是满，pass 最少、开销最低 → **prefill 偏爱大
  MXU**；但 decode（M 小）它半空 → **decode 偏爱不超过 batch 的 MXU**。

---

## 4. 建议（含 ≥3 条 Tier-3 支撑）

**推荐维持 16×16×16。** 支撑证据：

1. **M-padding 阶梯**（本报告 §2 + 图）：16³ 在 batch=16 满载，32³ 需 batch=32；边端
   decode 批量通常 ≤16 → 32³ 会长期半空。
2. **tile-pass 开销**（§3）：8³ 要 8× 发射才达到同等利用率；decode 访存受限下这点
   计算利用率不值那么多控制开销。
3. **P2 的访存受限证据**（`REPORT_P2_DOUBLE_BUFFER.md` §4）：16³ 下 GEMM 算术强度
   ~4 MAC/B，已是访存受限；放大到 32³ 会进一步提高算术强度、缓解访存压力——**但仅
   当 M 够大（prefill 或 batch≥32）**才用得上，否则被 M-padding 吃掉。

**折中方案（若要兼顾 prefill 吞吐与 decode 利用率）**：16×16×16 是 decode 与 prefill
的平衡点。若工作负载以 prefill 为主可考虑 32³；以低批 decode 为主则 16³ 最稳，且真正
的杠杆是**把 batch 提到 16 填满 M**（Tier-1 报告已量化：batch 1→16 免费换 16× 吞吐）。

---

## 5. 对照 spec 成功标准

| 指标 | 目标 | 实测 |
|:--|:--|:--|
| shape→utilization 表（含 8/16/32 对比） | 必须 | ✅ §2 表 + `p3_mxu_size_dse.csv` + 图 |
| Pareto/推荐配置 + ≥3 条支撑 | 必须 | ✅ §4：推荐 16³，三条支撑（M-padding 阶梯 / pass 开销 / P2 访存受限）|
