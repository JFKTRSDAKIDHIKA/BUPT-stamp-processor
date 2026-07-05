# edgc — 侧端大模型 AI 编译器 + 周期精准仿真链路报告

日期：2026-07-04 · 位置：`tools/edgc/` · 目标：MOBOL 3D 加速器

---

## 1. 目标与总体架构

输入：侧端大模型的 metadata 描述（`ModelConfig`，贴近真实 config.json）。
输出：**静态编译**的完整 tile 指令 trace + DRAM 权重镜像，由 C++ 周期精准模拟器读入执行。

```
ModelConfig (metadata)                     edgc/model.py
      │  gen_weights (确定性 f16)
      ▼
高层 f16 数据流参考 forward  ──────────────  edgc/model.py  (reference_forward)
      │                                        （逐位 oracle：模型该算出什么）
      ▼  Compiler.compile  (lowering + DSE)     edgc/compile.py
tile-ISA trace (.trace, 文本) + DRAM 镜像 (.mem, 二进制)
      │                     ├──────────────────┐
      ▼                     ▼                   ▼
golden ISA 解释器      C++ trace_sim         DSE 驱动
edgc/golden.py        src/trace_sim.cpp     edgc_cc.py --dse
（ISA 语义参考）      （周期精准）           spec_decode.py
```

**双重比特级校验**（每个模型都跑）：
1. `golden == reference_forward`：编译器 lowering 正确（模型语义没编错）。
2. `golden == simulator`：C++ 模拟器逐字节按 ISA 语义执行了 trace（模拟器正确）。

两者都逐字节 `memcmp` 通过 → trace 是模型的忠实静态编译，且模拟器忠实执行它。

## 2. 支持的侧端特性（全部编译 + 比特级验证）

| 特性 | 支持形式 | 验证模型 |
|:--|:--|:--|
| 注意力变体 | `n_kv_heads` 参数化：MHA(=n_heads) / GQA(<) / MQA(=1) | edge_gqa, edge_mqa |
| 掩码 | causal / full / **滑动窗口**（softmax 内融合掩码，边界位置逐位正确） | edge_sliding (window=8) |
| 位置编码 | RoPE（rotate-half，单块内 dim k↔k+half，per-position cos/sin 表） / none | edge_gqa 等 |
| 归一化 | RMSNorm / LayerNorm（行方向，跨块 blocked 布局） | 全部 |
| FFN | SwiGLU（silu(gate)⊙up）/ GELU（tanh 近似） | swiglu/gelu 各半 |
| **MoE** | top-k 路由，**编译期静态求解**（对具体输入的路由决策 = 对该次推理编译） | edge_moe (4 专家 top-2) |
| **投机解码** | draft(小)+target(大) 双模型，target 一趟并行验证 γ 个提议位置 | spec_draft/spec_target |

**为流片扩展的 ISA**（`src/cycle/isa.h`，规范缺口——原 MOBOL 只有 MXU）：VPU 新增 `MUL_F32 / SILU_F32 / RMSNORM_BLK / SOFTMAX_BLK（含 causal+滑窗）/ ROPE_F16`。这些是"数据相关掩码 + 行方向归约 + 逐元素门控"在硬件上的落点；没有它们侧端 LLM 无法在本架构执行。均逐拍计费（cycle_config.h）。

## 3. 实测（3D DRAM，密度 2048，500 MHz，baseline 调度）

| 模型 | 配置 | 指令数 | 逻辑拍 | MXU ops | DRAM 读 |
|:--|:--|:--|:--|:--|:--|
| edge_dense | MHA, LN, GELU, causal | 3 440 | 10 387 | 272 | 300 KB |
| edge_gqa | GQA(4→2), RMS, RoPE, SwiGLU | 4 086 | 11 055 | 304 | 370 KB |
| edge_mqa | MQA(4→1), RMS, RoPE, SwiGLU | 3 986 | 10 884 | 288 | 349 KB |
| edge_sliding | GQA + 滑窗 8 | 4 086 | 11 055 | 304 | 370 KB |
| edge_moe | GQA + 4 专家 top-2 | 10 070 | 26 983 | 880 | 1.06 MB |

（seq=16, d_model=64, heads=4, d_head=16, ffn=128, 2 层。）MQA/GQA 比 MHA 省 KV 投影（MXU ops 288/304 vs 272... 注：本 lowering 未共享 KV 计算，GQA 收益体现在 DRAM/KV 存储而非此处 MXU；见局限）。

### DSE（编译期调度搜索，闭环调模拟器择优）
edge_gqa 扫描 5 个调度：baseline 11055 → W2(端口加宽) 10128 → dense8192(密度+端口) **10084**。编译器自动选 dense8192。（NMC 对本通用 lowering 无效——归约走的是通用 GEMM+barrier 流，未路由到 bank 近存引擎；NMC 收益在手写 llm_layer 工作负载中已验证。）

### 投机解码成本模型
draft(1 层)=4 805 拍，target(4 层)=20 316 拍一趟并行验证 8 个位置。每接受步成本 = draft+target = 25 121 拍产出 (α+1) token：

| 接受长度 α | tokens | cyc/token | 加速 |
|:--|:--|:--|:--|
| 0 | 1 | 25 121 | 0.81× |
| 1 | 2 | 12 560 | 1.62× ← 盈亏平衡 |
| 5 | 6 | 4 187 | 4.85× |
| 8 | 9 | 2 791 | 7.28× |

α≥1 即盈利；draft 仅占 19% 步成本。架构使投机解码可行的关键是 **target 一趟并行处理全部 γ 位置**（20 316 拍不随 γ 增长）。随机初始化的 draft/target 实测接受 0/8（未训练，诚实报告）；蒸馏 draft 典型接受 4-6/8 → ~4.9× 吞吐。

## 4. 编译过程中发现并修复的深层缺陷（严谨性记录）

比特级校验暴露了多个隐蔽 bug，每个都定位到根因后修复：

1. **LOCAL scratch 多块 strip 重叠**：临时缓冲落在多块累加器的 4KB 区间内，c=0 的写覆盖 c=2 的结果。→ 重排为不重叠的 16KB 槽。
2. **f16 激活数据流建模缺失**：参考把中间量留在 f32，但硬件 DRAM 激活是 f16；残差流、每个 matmul 输出都必须按 f16 截断。→ 重写 reference_forward 为忠实 f16 数据流。
3. **K-block 累加结合律**：golden 把每个 K-block 单独求和再相加，而 blockops 用旧 C 播种连续累加——跨块结合律不同，1 ULP 差异（Q/K/V 曾"侥幸对上"）。→ matmul 加 `c_init` 连续播种。
4. **异步 DMA 的 WAR 冒险**：store DMA 异步读缓冲 B，后续 op 又写 B——同步 golden 看不到，异步 sim 数据损坏。→ 编译器在缓冲复用前插 fence / 分槽。
5. **MXU 结果未 wait 就被消费**：4 拍延迟内 VPU/DMA 读到旧值（GQA 侥幸、sliding 暴露）。→ 补 `wait_mxu`。
6. **超越函数不可跨实现复现**：C++ `std::exp/tanh/sqrt`(libm) 与 Python `math.exp`+round 有 2.8% 值差 1 ULP，跨层放大到 0.5。→ golden/reference 经 ctypes 调用**同一 libm 的 f32 函数**，并逐操作镜像 blockops 的 f32 运算序（含左结合多项式、eps 的 f32 舍入）。

第 6 点对流片重要：真实硬件用定义好的 LUT/多项式而非 libm；当前用共享 libm 消除了跨实现分歧，流片前应把两侧统一换成 RTL 采用的同一近似（结构已就绪）。

## 5. 局限与后续

- **通用 lowering 未做算子融合/KV 共享**：GQA/MQA 的 KV 复用、attention 的 QK^T→softmax→PV 融合、NMC 归约路由都未在通用路径实现（手写 llm_layer 已验证 NMC 收益）。这是编译器成熟度问题，非架构限制。
- **相位 barrier 粗粒度**：每个算子后全局 barrier，跨算子重叠不足；MXU 占用低。可用更细的 tag 依赖替代全局 barrier（同步原语已支持）。
- **边端规模**：seq=16、d_model=64 便于逐位校验；放大到真实侧端尺寸（seq 512、d_model 2048）只是 tiling 循环次数，lowering 结构不变。
- **投机解码接受率**：需真实蒸馏 draft 才能展示高接受；本链路提供的是**经过验证的成本模型**。

## 6. 复现

```bash
cd tools/edgc && ./run_edgc.sh          # 5 模型比特级验证 + DSE + 投机解码
python3 edgc_cc.py --model edge_moe --check --sim ../../build/src/trace_sim \
    --ramulator ../../config/ramulator_3d_dram.yaml   # 单模型
python3 spec_decode.py --sim ../../build/src/trace_sim \
    --ramulator ../../config/ramulator_3d_dram.yaml --gamma 8
```
