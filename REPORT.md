# MOBOL 周期精准模拟器 — 验证与架构评估报告

日期：2026-07-04（rev4：+ 权重流式预取〔bank DMA 直收 DRAM〕）· 模拟器：`src/cycle/`（执行驱动、逐周期 tick）

---

## 1. 模拟器覆盖范围（全部逐周期建模）

| 子系统 | 模型 | 关键参数 |
|:---|:---|:---|
| Tile 序列器 | 顺序单发射，取指/译码/派发 1 条/拍，停顿原因分类统计 | — |
| MXU | 流水线 II=1、延迟 4 拍；在飞累加器转发（output-stationary） | 16×16×16, f16/f32×f16→f32 |
| VPU（规范扩展） | 16 lane f32；add/cvt/transpose/scale/softmax/gelu/layernorm 按拍计费 | softmax 128 拍/块 等 |
| LOCAL SRAM | 2R1W，64B/端口/拍，逐拍仲裁，争用计数 | 256 KB/tile |
| DMA 引擎 | 1 通道/tile，2 拍 setup，1×64B 子事务/拍，2D strided，在飞上限 64，逐子事务 ack | 队列深 16 |
| NoC 双向环 | flit 级（64B），1 flit/链路/向/拍，每跳 1 拍，缓冲背压，**2 虚拟通道**（请求/响应分离、响应优先——协议级无死锁），确定性仲裁 | 16 节点 |
| 3D 垂直互联 | 每 tile 私有 hybrid-bond 至本组 bank（64B/拍/向）；**每组独立 bank↔DRAM 列，密度为 DSE 旋钮（bits/列/拍）**；远 bank 经环上网关中转 | 列队列深 64 |
| SHARED bank | 轮询服务 4 上行链路，端口争用可开关，访问计数常开 | 8 MB×4 |
| **DRAM（3D 堆叠 die）** | **Ramulator2 HBM3 级核心：16 通道×128-bit、bank group×bank 并行、32B 原生访问粒度、OpenRow FRFCFS、REFab 刷新（自写 AllBankNoRank——Ramulator2 原生 AllBank 对无 rank 器件静默不刷新）**；16 GB（=规范）；核心峰值 256 GB/s | 逻辑:DRAM = 1:2（500 MHz 基底 / 1 GHz 控制器），yaml 单一来源 |
| 同步 | release token 走 NoC、arity-k join、acquire 物理阻塞、代际复位 | tag 表 1024 |
| Trap | 洞位/越界/远 bank 直访——release 版常开 | — |

**3D 封装建模**（rev2 重点）：DRAM die 直接混合键合在 buffer die 之上，不再是平面封装的单通道 DDR4。DRAM 侧多 bank/多通道并行由 Ramulator2 HBM3 模型承担（RoBaRaCoCh 把通道位放最低位，连续 32B 行自动跨通道交织）；垂直方向每组一条键合列，密度（512–8192 bits/逻辑拍）独立扫描。64B fabric flit 在 DRAM 控制器处拆成 2×32B 器件事务，全部完成才放行响应——链路满载计费与器件时序粒度两不误。

## 2. 验证（6/6 测试，`./run_all.sh` 一键复现）

- 双 GEMM 与 2 层 transformer decoder prefill 均与 CPU 参考 **memcmp 比特一致**；C 矩阵零 DRAM 往返；因果校验全过。
- 已知答案微基准逐拍锁定：邻居 push 9 拍、近/远 bank 3/13 拍、MXU 8 连发 8+4 拍、token 飞行 3 拍、**HBM3 冷读经 3D 列往返 14 拍**（≈18 tCK ACT+RD+RL = 9 逻辑拍 + 5 级垂直/入口流水）、非法访问全部 trap。
- 确定性：任意次运行周期数与数值逐位一致（40/40 压力零失败；ASan/UBSan 全插桩干净，含 Ramulator2 源码）。
- 修复的关键缺陷：Ramulator2 `source_id` 越界堆破坏（前任冒烟测试同样命中）、请求/响应共享缓冲的协议死锁（2-VC 解决）、release 版边界检查失效、**HBM 器件下刷新静默缺失**（自写刷新管理器，开销 +4.3%，证明此前确被乐观化）。

## 3. 垂直互联密度 DSE（本次核心实验）

LLM 2 层 prefill，HBM3 级 3D DRAM，扫描每列 hybrid-bond 密度：

| 密度 (bits/列/拍) | 每列带宽 @500MHz | 全片垂直带宽 | LLM 2 层逻辑拍 | 相对增益 |
|:---|:---|:---|:---|:---|
| 512  | 32 GB/s  | 128 GB/s | 8 080 | 基准 |
| 1024 | 64 GB/s  | 256 GB/s | 7 387 | +8.6 % |
| **2048** | **128 GB/s** | **512 GB/s** | **7 247** | **+1.9 %** |
| 4096 | 256 GB/s | 1 TB/s   | 7 178 | +0.9 % |
| 8192 | 512 GB/s | 2 TB/s   | 7 082 | +1.3 % |

（dual_gemm：512→493 拍，2048→418 拍，8192→418 拍——2048 即完全饱和。）

**DSE 结论：拐点在 2048 bits/列/拍。** 物理原因：base die 消费端上限——每 tile 的 SRAM 写口 64B/拍 × 每组 4 tile = 恰好 2048 bits/列/拍。垂直密度超过 2048 后，瓶颈转移到 tile 端口与 DRAM 核心（256 GB/s），继续加密度只是浪费键合资源。**若要利用 4096+ 密度，需先：(a) 加宽 tile ingress 端口，或 (b) 让权重直接 DRAM→SHARED bank 流式预取（buffer die 直收，绕过 tile 端口）——后者是最值得做的架构演进。**

## 4. 实测数据（3D DRAM，密度 2048，500 MHz 逻辑）

| 负载 | 逻辑拍 | 墙钟 | MXU 占用 | NoC 平均/峰值 | DRAM 平均读延迟 |
|:---|:---|:---|:---|:---|:---|
| 双 GEMM | 418 | 0.84 µs | ~1 % | 10.4 % / 47.8 % | 38 tCK |
| LLM 1 层 | 3 512 | 7.0 µs | 0.63 % | ~9 % / ~19 % | ~90 tCK |
| LLM 2 层 | 7 247 | 14.5 µs | 0.63 % | 9.1 % / 19.1 % | 92 tCK |
| LLM 4 层 | 14 834 | 29.7 µs | 0.63 % | — | — |
| LLM 8 层 | 28 167 | 56.3 µs | 0.64 % | — | — |

对比平面封装 DDR4 基线（用户指出的错误建模）：LLM 2 层 28 551 拍 @400MHz = 71.4 µs → **3D 堆叠 + 键合列 = 4.9× 墙钟加速**；DRAM 平均读延迟 322 tCK → 92 tCK。层扩展保持线性（~3.6 k 拍/层）。

## 5. 架构优点（更新）

1. **3D 堆叠 DRAM 是本架构的正确性支柱**：DSE 证明 2048 bits/列的键合密度即可把 DRAM 排队延迟从 322 tCK 压到 92 tCK、端到端 4.9×——这是平面封装拿不到的。
2. PGAS one-hot 编址：路由/近远/越界全部位比较，洞位 trap 零成本。
3. tag/arity 同步：无一致性协议、NoC 乱序免疫、arity-2/16 均验证正确，硬件近乎免费。
4. SHARED 融合 + NUMA 感知放置有效（近 3 拍 vs 远 13 拍）。
5. 静态调度 → 比特级确定性，流片 sign-off 友好。

## 6. 架构局限（更新后按严重性）

1. **base die 消费端口是新瓶颈**：DRAM 供给能力（256 GB/s 核心 + 512 GB/s 垂直@2048）已超过 tile 端口消费能力（每 tile 32 GB/s）。做大模型必须让 SHARED bank 直收 DRAM 权重流（buffer-die DMA），或加宽 tile ingress。
2. **DRAM 快了之后，LN 串行化与同步开销浮出水面**：tile0 的 VPU 停顿 1636 拍在 7247 拍总时长中占 22.6%（DDR4 时代仅 6%）。跨 tile 行归约硬件/树形归约的优先级上升。
3. 无组播：激活广播仍 3× 冗余 DRAM 读；NoC 峰值链路利用已到 19%（dual_gemm 47.8%），组播同时省 DRAM 和 NoC。
4. VPU 是规范缺口（必须回写规范）；release 依赖 ack 往返；flat arity-16 归约 240 拍；环直径随 tile 数增长——同 rev1 分析。
5. **DRAM 参数需与工艺对齐**：HBM3 时序是堆叠核心的替身；实际混合键合 DRAM（无 TSV/PHY serdes）行为可能更快、且 bank 数/页大小可定制——流片前需用供应商 die 的实测时序替换 `config/ramulator_3d_dram.yaml`（结构已支持任意 Ramulator2 器件模型）。

## 7. 建模假设

- HBM3_2Gbps 时序作为混合键合堆叠 DRAM 核心的保守替身（真实键合 die 无 PHY 串化开销，延迟只会更低——不会高估性能）。
- 控制 flit 按整 flit 计费（保守）；MXU/VPU 操作数专用通路（Q1，可开关）；VPU 阻塞式（保守）。
- 逻辑 die 500 MHz、DRAM 控制器 1 GHz（yaml `clock_ratio: 2` 为单一时钟域来源）。
- 刷新：REFab 每 nREFI 每通道（自写 AllBankNoRank）。

## 8. 消费端口 DSE 与三范式架构研究（rev3）

### 8.1 base die 消费端口 DSE（问题 1）

W =（tile 垂直链路宽度 = SRAM 写口数 = DMA 发射率），LLM 2 层：

| | density 2048 | 4096 | 8192 |
|:--|:--|:--|:--|
| W=1（64B/拍/tile） | 7 247 | 7 178 | 7 082 |
| W=2（128B/拍/tile） | 6 709 | 6 566 | 6 408 |
| W=4（256B/拍/tile） | 6 290 | 6 103 | **6 047** |

结论：**端口与密度互补**——W=1 时密度几乎无用（tile 端口封顶），W=2 后密度收益打开；W1→W4+8192 共提速 16.6%，收益递减点在 W=2~4（再往上受 DRAM 核心 256 GB/s、LN 串行化与同步开销制约）。加宽消费端口的硬件代价：SRAM 写口 ×W、垂直键合 ×W、DMA 引擎发射带宽 ×W——W=2 是面积效益比最好的档位。

### 8.2 buffer die SRAM 建模与"无 buffer die"研究（问题 2）

SHARED bank 建模内容：bank 控制器逐拍轮询服务 4 条 tile 上行链路（宽度随 W）、SRAM 端口争用可开关且访问计数常开、DRAM 流量作 passthrough 转发、远 bank 经环上网关路由；本套工作负载下 4 端口即无争用（计数为 0）。

去掉 buffer die（SHARED→DRAM 往返，2-die 堆叠）：

| 负载 | 3-die baseline | 2-die no-buffer | 差异 |
|:--|:--|:--|:--|
| LLM 2 层 | 7 247 | 7 998 | **+10.4 %** |
| LLM 8 层 | 28 167 | 31 029 | +10.2 % |
| 双 GEMM | 418 | 416 | ≈0 |
| DRAM 读流量（LLM 2 层） | 244 KB | 308 KB | +64 KB（=LN 2KB×16 tile×2 层，逐字节对账） |

结论：**在 3D 快 DRAM 前提下，buffer die 的价值主要来自"多读者复用"而非单纯低延迟**——dual_gemm 的 C 只有单读者，走 DRAM 几乎无损；LLM 的 LN 激活被 16 个 tile 读，SRAM 驻留省 26% DRAM 读流量并快 10%。若砍掉 buffer die 省成本，代价是每层 +10% 时延且 DRAM 流量随读者数线性膨胀（大模型下更糟）。

### 8.3 buffer die 近存计算（问题 3）

在每个 bank 旁加一个 16-lane NMC 引擎（命令式：固定序 f32 归约 / 融合 LayerNorm+转换，完成后发 release token，与 tag 同步原语原生集成）。**关键设计教训（DSE 发现）：归约必须随 bank 分布**——4 个输出块的归约放同一 bank 时单引擎串行（比 baseline 慢 10%），分散到 4 bank 后并行（快 10%）。

| 指标（LLM 2 层, W=1） | baseline | NMC | 变化 |
|:--|:--|:--|:--|
| 总周期 | 7 247 | **6 513** | **−10.1 %** |
| tile0 VPU 停顿（LN 串行化） | 1 636 | 556 | **−66 %** |
| NoC flit-hops | 21 204 | 16 072 | −24 % |
| 指令数 | 2 162 | 2 040 | −6 % |

释放的 base die 资源：每 reducer 的 16 KB psum slot 缓冲 ×4 + tile0 的 ~9 KB LN 工作区（共 ~73 KB LOCAL SRAM），VPU 归约/LN 负载 −66%（可缩减 VPU 或让位）。**再投资实验**：把省下的面积投给消费端口加宽（W=2），NMC+W2 = 5 814 拍，较 baseline W1 **−19.8%**。

### 8.4 三范式对比总表（问题 4）

| 范式 | LLM 2 层 | LLM 8 层 | 相对 | 成本/风险 |
|:--|:--|:--|:--|:--|
| B：2-die 无 buffer | 7 998 | 31 029 | +10 % | 最低成本（省一层 die + 32MB SRAM）；DRAM 流量与读者数线性膨胀；能耗↑（DRAM 访问 vs SRAM） |
| A：3-die baseline | 7 247 | 28 167 | 基准 | 规范现状 |
| C：3-die + NMC | 6 513 | 25 548 | −10 % | bank 侧 4 个 16-lane 引擎（面积小）；bank SRAM 读写口复用 |
| **C+：NMC + W2 端口** | **5 814** | **23 037** | **−19.8 %** | C 的面积 + 释放面积再投端口 |

**推荐：范式 C+**。理由：(i) NMC 消除了两个已量化的 base die 串行化热点（LN 22.6%→~8%、arity-16 归约链）；(ii) 释放的 SRAM/VPU 面积恰好覆盖端口加宽的开销；(iii) 归约/规约类算子天然靠近数据（psum 只上行一次，不再经环网二次搬运，NoC 流量 −24%）；(iv) 三范式全部通过同一 CPU 参考的比特级校验——数值语义完全不变，纯粹是物理位置与执行引擎的迁移，编译器工作量可控。

### 8.5 权重流式预取：DRAM→SHARED bank 直收（rev4）

机制：buffer die 每个 bank 增加一个**预取 DMA 引擎**（与 NMC 引擎并行），由组长 tile（0/4/8/12）用 `NMC PREFETCH` 命令驱动：引擎经**本组自己的垂直列**直接从 DRAM 流式拉取 2D 权重区域写入 bank SRAM，完成后回 release token——权重数据全程不经过 base die。按层双缓冲（奇偶区域），组长提前一层发起下一层预取（与本层计算完全重叠），收齐 token 后向组内广播 WPREF。各 bank 只装本组 tile 消费的权重切片（Wq/Wk/Wv 各归一 bank、W1 列切片、W2 行切片）——**零复制，DRAM 总读字节与 baseline 严格相等（249 856 B，逐字节对账）**。

| 配置（LLM 2 层） | 周期 | vs baseline W1 | LLM 8 层 | vs baseline |
|:--|:--|:--|:--|:--|
| baseline W1 | 7 247 | — | 28 167 | — |
| baseline + stream W1 | 6 599 | −8.9 % | 26 526 | −5.8 % |
| baseline + stream W2 | 5 836 | −19.5 % | 23 388 | −17.0 % |
| NMC + stream W1 | 5 725 | −21.0 % | 24 669 | −12.4 % |
| **NMC + stream + W2（范式 C++）** | **5 089** | **−29.8 %** | **22 157** | **−21.3 %** |

观测：(i) DRAM 请求数 7808→4736、bank↔DRAM 列流量 −39%（预取用满 64B 块、顺序流、开行友好）；平均读延迟 92→120 tCK 但已**不在关键路径**（完全被上一层计算掩盖）。(ii) streaming 与端口加宽、NMC 三者正交叠加。(iii) streaming 下垂直密度拐点仍为 2048 bits/列/拍（512 档 −9%），因为权重最终仍要经 tile 链路进 LOCAL——若未来 MXU 能直接从 bank 读操作数（权重驻留计算），密度需求才会进一步上移。

**最终推荐架构（范式 C++）**：3-die + 分布式 NMC + bank 预取引擎 + W2 消费端口，LLM 2 层 **5 089 拍（10.2 µs）**，较规范基线 −29.8%，较最初的平面 DDR4 错误建模快 **7.0×**（墙钟）。

## 9. 复现

```bash
./run_all.sh     # 构建 + 验证 + 层扩展 + 密度/端口 DSE + 三范式对比
./build/src/mobol_sim --workload llm --layers 8 --arch nmc --stream-weights \
    --ramulator config/ramulator_3d_dram.yaml \
    --tile-link 2 --wr-ports 2 --rd-ports 4 --dma-rate 2   # 范式 C++
```
