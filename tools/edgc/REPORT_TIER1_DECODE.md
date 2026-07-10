# Tier-1 真实模型结构 · Decode 阶段时序

Branch: `opt/p1-barrier-elim` · 日期：2026-07-10
数据：`results/tier1_decode_b1_c512.csv`、`results/tier1_llama1b_tier2.csv`
复现：`python3 tools/edgc/decode_tier1.py --sim build/src/trace_sim --batch B --ctx L`

> 纯时序、shape-driven。真实网络结构（真实层数/hidden/heads/FFN），随机/零数据。
> **层采样外推**：每层 shape 相同，故每个唯一 GEMM shape 只精确仿真一次，组合出
> 单层代价再 ×层数。含 P1+P2 优化，decode（单 token 对 KV cache）。

---

## 1. 真实模型结构（decode，batch=1，ctx=512）

| 模型 | 精度 | 层 | cyc/层 | cyc/token | tok/s@500MHz | MXU 利用率 | 活跃 tile | DRAM 读/token |
|:--|:--:|--:|--:|--:|--:|--:|--:|--:|
| Llama-3.2-1B    | f16 | 16 | 1 781 817 | 28.5 M  | 17.5 | 3.45% | 16/16 | 4.0 GB |
| Llama-3.2-3B    | f16 | 28 | 2 906 087 | 81.4 M  |  6.1 | 3.49% | 16/16 | 11.6 GB |
| Phi-3.5-mini    | i8  | 32 | 2 321 775 | 74.3 M  |  6.7 | 2.44% | 16/16 | 7.5 GB |
| Mistral-7B-INT4 | i4  | 32 | 5 048 039 | 161.5 M |  3.1 | 1.08% | 16/16 | 7.1 GB |
| Mixtral-8x7B    | f16 | 32 |10 364 520 | 331.7 M |  1.5 | 3.75% | 16/16 | 51.0 GB |

模型结构（来自各自 config）：Llama-1B d=2048/32h/8kv/dh64/ffn8192；Llama-3B
d=3072/24h/8kv/dh128/ffn8192；Phi-3.5-mini d=3072/32h/dh96/ffn8192；Mistral-7B
d=4096/32h/8kv/dh128/ffn14336；Mixtral 同 Mistral + 8 专家 top-2。

---

## 2. 核心结论：decode 是**极度访存受限**，MXU 利用率 1–4%

- **N 维并行填满了 16 个 tile**（投影/FFN 的输出维 d_model/d_ffn 很大），所以不是
  "tile 空置"问题——16/16 tile 都在动。
- 但 **MXU 利用率仅 1–4%**：decode 每步 M=1（单 token），MXU 的 16×16 块**只有 1/16
  行是真数据**，另外 15 行是 padding；且每 token 要把**整套权重**从 DRAM 流一遍
  （4–51 GB/token），MXU 算得快、等数据等得久。
- **Mistral-INT4 利用率最低（1.08%）**：INT4 打包 4×K/发射 → MXU 发射更少 → 相对
  权重流的 MXU 占用更低。INT4 省的是 DRAM（7.1GB，比同规模 f16 少 4×）和发射数，
  不是利用率。
- **Mixtral 每 token 读 51 GB**：即便 top-2 稀疏，2 个专家的大 FFN（d_ffn=14336）
  权重流仍主导；数字还含输出块驻留映射对小 A 的重复重载（真实的搬运浪费）。

---

## 3. Tier-2 维度扫描（Llama-3.2-1B）

**batch 扫描（ctx=512）——batch≤16 免费提吞吐：**

| batch | cyc/token | tok/s | MXU 利用率 |
|--:|--:|--:|--:|
| 1  | 28.5 M | 17.5 | 3.45% |
| 8  | 28.5 M | 140.3 | 3.45% |
| 16 | 28.5 M | **280.6** | 3.45% |
| 32 | 55.8 M | 286.7 | 3.52% |

**关键**：batch 1→16 **cycle 完全不变**，tok/s 却 16×（17.5→280）——因为 decode 的
M-block 本来就有 16 行、只填了 1 行；batch 到 16 把那 15 行空 padding 填满，**零额外
成本换 16× 吞吐**。batch>16 需要更多 M-block，成本转为线性（batch32 ≈ batch16）。
利用率的原始占用值不变（同样的 MXU 发射覆盖 16 个 token 而非 1 个），但**有效计算**
提高了 16×——这正是 decode 的正解：**用 batch 填 M-padding**（连到 P4 的动机）。

**context 扫描（batch=1）——decode 对 context 不敏感：**

| ctx | cyc/token | tok/s |
|--:|--:|--:|
| 128  | 27.6 M | 18.1 |
| 512  | 28.5 M | 17.5 |
| 2048 | 32.1 M | 15.6 |

context 128→2048 只让 cycle +16%：注意力（QK^T/PV ∝ L）随 context 增长，但相对
**权重流（FFN 主导，与 context 无关）**是小头。decode 直到极长 context 前都是权重
受限，不是注意力受限。

覆盖的 **Tier-2 维度 ≥3**：注意力类型（GQA/MQA/MHA，跨模型）、精度（f16/i8/i4，
跨模型）、batch（1/8/16/32）、sequence length（128/512/2048）、phase（decode）。

---

## 4. 对照 spec 成功标准

| 指标 | 目标 | 实测 |
|:--|:--|:--|
| Tier-1 真实 shape 端到端 | cycle + 利用率 | ✅ 5 个真实结构，cyc/token + 利用率 + tok/s + DRAM/token |
| 时序正确（causality） | 必须 | ✅ 全部通过 |

**给后续的判断**：decode 的瓶颈是 (a) 权重流带宽、(b) M=1 的 15/16 行浪费。
(b) 靠 **batch≤16**（本报告已量化 16× 免费）或 P4 的多 token 打包解决；(a) 靠更低
精度（INT4，已实现）和权重复用/更宽键合缓解。MXU 尺寸对这两点的影响见 P3 分析。
