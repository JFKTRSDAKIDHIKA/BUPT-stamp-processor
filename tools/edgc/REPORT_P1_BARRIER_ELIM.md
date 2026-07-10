# Priority 1 — Barrier Elimination via Dependency-Aware Synchronization

Branch: `opt/p1-barrier-elim` · 日期：2026-07-10
复现：`EDGC_BARRIER={global|dep} python3 tools/edgc/profile_mxu.py --sim build/src/trace_sim`
数据：`tools/edgc/results/p1_barrier_elim.csv`

---

## 0. 与 spec 的对齐说明（重要）

Spec 里 Priority 1 假设当前调度是 `DMA → global_barrier → MXU → global_barrier`
（即 GEMM 的 K 内层循环被 barrier 打断）。**在本代码库中并非如此**：GEMM 的
K 内层循环在**单个 tile 上已经是顺序流水**（load→fence→MXU(acc)→wait），tile
之间在 GEMM 内部**不做**同步；MXU 的在飞累加转发（`acc=1`）也已启用。真正的
全局 barrier 出现在**原语之间**（每个 `_gemm/_norm/_rope/_residual/_attention/
_silu/_gelu/_scale_add` 结尾各一次），是保守的相位串行化。

因此本 Priority 1 的正确落点是**原语级依赖感知的 barrier 消除**：只有当后一个
原语与"上次 barrier 以来被读/写过的 DRAM buffer"发生 RAW/WAR/WAW 冲突时才插
barrier；相互独立的原语（三个 QKV 投影都只读 `xn`、MoE 各专家都只读 `hn`）之间
不再插 barrier。语义等价于 spec 的意图（去掉不必要的同步），但机制适配真实 IR。

实现：`tools/edgc/edgc/compile.py` 的 `_sync(reads, writes)`（约 20 行），每个
叶子原语开头声明读写 buffer 集合。`EDGC_BARRIER=global` 强制每原语前插 barrier，
**逐位复现原调度**（A/B 基线，见下）；默认 `dep` 为优化调度。

---

## 1. 结果（Tier-1 代理工作负载 + Tier-2 维度标注）

> Tier-1 真机型号（Llama-3.2-1B/3B、Phi-3.5、Mistral-7B-INT4、Mixtral）**无法**在本
> 链路端到端运行——golden 参考是纯 Python 逐位 f16 前向、权重是确定性随机生成、
> ISA 无 INT8/INT4（详见 §4）。这里用仓库自带的 edge_* 代理，它们覆盖同一组
> **计算特征维度**：注意力变体 MHA/GQA/MQA、MoE 稀疏路由、causal/滑窗掩码。
> 扫到的 **Tier-2 维度**：注意力类型（MHA/GQA/MQA）、掩码/phase（prefill causal +
> sliding window）、稀疏度（dense vs. MoE top-2）。精度维固定 FP16（无 INT 支持）。

| 模型（Tier-2 特征） | baseline 拍 | P1 拍 | **加速** | barrier 数 (去掉) | MXU 占用 base→P1 | 逐位 |
|:--|--:|--:|--:|--:|--:|:--:|
| edge_dense (MHA/LN/GELU)      | 10 387 | 9 974 | **1.041×** | 24→20 (−4)  | 0.655→0.682% | ✅ |
| edge_gqa (GQA 4→2/RoPE/SwiGLU)| 11 055 | 10 603| **1.043×** | 30→22 (−8)  | 0.687→0.717% | ✅ |
| edge_mqa (MQA 4→1)            | 10 884 | 10 315| **1.055×** | 30→22 (−8)  | 0.662→0.698% | ✅ |
| edge_sliding (GQA+滑窗8)      | 11 055 | 10 603| **1.043×** | 30→22 (−8)  | 0.687→0.717% | ✅ |
| **edge_moe (4 专家 top-2)**   | 26 983 | 24 142| **1.118×** | 62→42 (−20) | 0.815→0.911% | ✅ |
| spec_target (GQA×4 层)        | 22 316 | 21 109| **1.057×** | 60→44 (−16) | 0.681→0.720% | ✅ |

**正确性**：全部 7 个模型（含 spec_draft）`golden==reference==simulator` 逐位通过；
`global` 模式逐位复现原 baseline（cycles 完全一致，如 edge_gqa=11055）。

---

## 2. 为什么有效，以及为什么达不到 spec 估的 75%（用 Tier-3 视角解释）

**结论：barrier 消除是正确且有收益的（MoE +11.8%，其余 +4–5%），但 spec 估计的
"acquire 停顿 ↓75%"在本工作负载上不可达——因为 acquire 停顿是 DMA 受限关键路径
+ 半数 tile 空置的*症状*，barrier 数只是它被*归因*的位置，不是*成因*。**

证据（edge_moe，dep 模式）：
- barrier 从 62 砍到 42（−32%），但 **acquire 停顿仍占周期预算的 68.9%**。
- 只有 **8/16 tile 跑 MXU**，最忙 tile 的 MXU 占用仅 2.72%。
- dma 停顿占 24.1%。

机理：即便去掉了不必要的 barrier，剩下的**依赖必需**的 barrier 仍让先算完的
tile 等最慢的那个 tile，而最慢 tile 的时间花在 **DMA 往返**上（每个 MXU 算子被
load A/load B/fence/wait 串行包裹）。加上 T8–T15 全程空置，关键路径根本不在
barrier 上。所以：
- **MoE 收益最大**：它有多个**真正独立**的专家原语，去掉专家间 barrier 让不同
  tile 的专家工作能错开 → 11.8%。
- **注意力/dense 收益小**：QKV 虽独立但落在同一批 tile（1×4 输出块只用 T0–T3），
  同一 tile 仍顺序执行 q→k→v，去掉 barrier 只省掉 barrier 自身的往返延迟 → ~4%。

这直接指明后续杠杆：**P2（双缓冲隐藏 DMA 延迟）**打掉 24% 的 dma 停顿和串行的
load-compute；**P4（填满 T8–T15）**打掉"半数阵列空置"。P1 是二者的前置（点对点
tag 同步已就位），但单靠 P1 不改变 DMA 受限的关键路径。

---

## 3. 对照 spec 成功标准

| 指标 | spec 目标 | 实测（本工作负载） | 说明 |
|:--|:--|:--|:--|
| Acquire 停顿（Tier-3 GEMM） | ≤25% of baseline | 降 3–11% | 目标不可达：见 §2，停顿受 DMA 关键路径支配，非 barrier 数 |
| 端到端加速 | — | 1.04–1.12× | MoE 最高 |
| 逐位正确 | 必须 | ✅ 全过 | golden==ref==sim |

诚实结论：spec 的 75% 估计来自"acquire 停顿占 75% 预算"这一观察，但它把**计数器**
当成了**可通过删 barrier 回收的量**。本轮用真实模拟器证否了这一点，并定位了真正
的瓶颈。这是 P1 最有价值的产出——它为 P2/P4 提供了量化依据。

---

## 4. 未做/不可做项（需你决策，勿默默假设）

以下 spec 条目在当前代码库**无法**如字面执行，需要架构/工具扩展或换方法，未在
本轮实现：

1. **Tier-1 真机型号端到端**（Llama/Phi/Mistral/Mixtral，1–8B）：golden 是纯 Python
   逐位前向 + 随机生成权重，跑 1B 规模不现实，且无真实 checkpoint 加载路径。
   本轮用 edge_* 代理覆盖同一特征维度。
2. **INT8/INT4 精度**：ISA 只有 `MXU_F16F16`/`MXU_F32F16`，无整型 MMA。需扩 ISA +
   模拟器数据通路。
3. **Priority 3（MXU 尺寸/数量 DSE）**：MXU=16×16×16 是**编译期结构常量**
   （`arch_config.h`/ARCH_SPEC `structural:`），非模拟器旋钮；且缺面积/功耗模型
   （`area ∝ size³`、`power ∝ active MACs` 需要新建）。做 18 点 DSE 需要先把 MXU
   参数化并建面积/功耗模型——是一项独立的模拟器工程。
4. **Priority 4 的 decode phase**：编译器目前只 lower prefill（seq 固定），没有
   逐 token 的 decode 循环；"decode batch=1 tile 利用率"需要先建 decode 代码路径。
   其中 **Strategy A（head-parallel）** 在现有 prefill 下可直接做（把不同 head 的
   注意力分到 T8–T15），是 P4 里最快落地的一块。

建议下一步：**P2 双缓冲**（收益最直接、依赖 P1 已就位、不需新架构），随后 P4 的
head-parallel 填满阵列。P3 与 Tier-1 真机需要先扩模拟器/编译器，属于更大的工作项。
