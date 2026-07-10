# MOBOL 加速器架构规范（Architecture Specification）

> 面向侧端大模型推理的三层堆叠加速器。
> 版本 v1.1 · 2026-07-10 · 状态：周期精准模拟器验证通过，面向流片
>
> 本文是权威架构规范：把原始非形式化描述（`MOBOL.md`）落实为经周期精准
> 模拟器（`src/cycle/`）验证的、可综合参数化的定义。所有可调参数的单一
> 真值源是 `config/mobol_arch.yaml`；本文正文给出语义，数值以 YAML 为准。
>
> **v1.1 变更（基于周期精准模拟器的实测扩展）**：新增低精度 MMA
> （`MXU_I8I8`/`MXU_I4I4`，§3.2/§4/§10）、DMA 部分排空 fence（`DMA_FENCE keep=N`）
> 与按发射序退休（§3.4）、双/多缓冲 GEMM 内层与 overlap 计数器（§3.4/§3.7）、
> 依赖感知 barrier 消除（§9.1）、以及 decode 阶段与 Tier-1 真实模型结构的时序
> 数据（§15–§18）。§15 起给出全部微基准/端到端实测数值。
>
> **两类保真度**：f16 主链路是**逐位确定**的（golden==reference==sim，§10/§12）；
> 低精度（INT8/INT4）与 decode/Tier-1 大模型仿真是**纯时序模型**（随机/零数据，
> 只保 cycle / stall / 利用率 / 因果性正确，不验数值，§15 起注明）。

---

## 0. 术语与约定

| 记号 | 含义 |
|:--|:--|
| tile | base die 上的计算单元，本规范 16 个 |
| bank | buffer die 上的共享 SRAM 体，本规范 4 个 |
| group | 4 个 tile + 其正上方 1 个 bank + 1 条垂直列，共 4 组 |
| flit | 互连流控单元，64 B（= 链路位宽） |
| 逻辑周期 | base/buffer die 时钟域（参考 500 MHz） |
| DRAM 周期 | DRAM 控制器时钟域（参考 1 GHz，比值见 §8） |
| f16 / f32 | IEEE-754 binary16 / binary32 |
| block | 16×16 元素块（MXU/VPU 原生粒度） |

数值全部小端。地址 40 位，字节寻址。

---

## 1. 物理组织：三层混合键合堆叠

```
        ┌─────────────────────────────────────┐
 顶层   │  DRAM die：HBM3 级核心，16 通道       │  ← §8
        │  16 GB，bank-group×bank 并行          │
        └───┬────────┬────────┬────────┬───────┘
     垂直列 │        │        │        │  混合键合，每组一条（§7）
        ┌───┴────┬───┴────┬───┴────┬───┴────┐
 中层   │ bank0  │ bank1  │ bank2  │ bank3  │  buffer die：4×8 MB SRAM
        │ +NMC   │ +NMC   │ +NMC   │ +NMC   │  每 bank 侧近存引擎（§6.4）
        └─┬─┬─┬─┬┴─┬─┬─┬─┬┴─┬─┬─┬─┬┴─┬─┬─┬─┬┘  每 tile 一条垂直键合（§7）
        ┌─┴─┴─┴─┴──┴─┴─┴─┴──┴─┴─┴─┴──┴─┴─┴─┴─┐
 底层   │  base die：16 个 compute tile        │  §3
        │  双向环 NoC（2 虚拟通道）            │  §5
        └─────────────────────────────────────┘
   分组：tile i 属组 g = i>>2；组 g 的 4 个 tile 共享 bank g。
```

结构常量（`structural:`，编译期固定，YAML 校验一致性）：
`num_tiles=16, num_banks=4, tiles_per_group=4, mxu=16×16×16`。

---

## 2. 地址空间（PGAS，无缓存一致性）

40 位物理地址，高 3 位 one-hot 段标识，其余为选择器 + 段内偏移。所有路由/
近远/越界判定均由位比较完成，无查表。

```
39 38 37 | 36                                    0
[ SEG   ]|[  选择器（高位） + 偏移（低位）        ]
```

| SEG[39:37] | 段 | 位置 | 选择器 | 偏移 | 容量 |
|:--|:--|:--|:--|:--|:--|
| `001` | LOCAL | base die | TILE[36:33] (4b) | [17:0] (18b) | 256 KB/tile |
| `010` | SHARED | buffer die | BANK[36:35] (2b) | [22:0] (23b) | 8 MB/bank |
| `100` | DRAM | 顶层 | — | [33:0] (34b) | 16 GB |
| `000` | NULL | — | — | — | 访问即 trap |

- 基址：LOCAL tile *j* = `0x20_0000_0000 + j×0x2_0000_0000`；SHARED bank *b* =
  `0x40_0000_0000 + b×0x8_0000_0000`；DRAM = `0x80_0000_0000`。
- **洞位铁律**：偏移中超出物理容量的高位必须恒 0，否则硬件 trap（零成本越界
  检查）。模拟器 release 版常开此 trap（`memory_model.cpp`）。
- **近/远判定**：SHARED 近 ⟺ `addr.BANK == TILE_ID[3:2]`（2 位比较）。近 = 正
  上方 bank，垂直一跳；远 = 经环网关中转（§7）。
- **双重命名 + 自别名**：`local[off]` 走硬连线本地口，不进 NoC；全局名
  `GLOBAL(LOCAL,tile=j,off)` 用于跨 tile 推送；当 TILE 字段 == 本 TILE_ID 时
  硬件折叠为本地快路径。

### 2.1 访问能力约束

| 目标 | 直接 load/store | 异步 DMA |
|:--|:--:|:--:|
| 本 tile LOCAL | ✅ 每周期 | — |
| 本组 SHARED（近） | ✅（`LOAD/STORE_SHARED`） | ✅ |
| 远 tile LOCAL / 远 SHARED / DRAM | ❌ | ✅ |

计算数据通路只直读本地；跨单元搬运一律显式 DMA，由编译器静态排程。远 bank
直接 load/store 触发 trap（§4 `LOAD_SHARED` 校验）。

---

## 3. Compute Tile 微架构

```
        ┌──────────────────────────────────────────────┐
 指令流 │ Sequencer（顺序单发射，1 指令/周期，停顿归因）│
        └──┬───────┬────────┬────────┬────────┬─────────┘
           ▼       ▼        ▼        ▼        ▼
        ┌────┐ ┌──────┐ ┌──────┐ ┌───────┐ ┌──────────┐
        │MXU │ │ VPU  │ │ DMA  │ │ join  │ │ LOCAL    │
        │流水│ │16lane│ │引擎  │ │计数器 │ │ SRAM     │
        │16³ │ │ f32  │ │1通道 │ │cnt[t] │ │256KB 2R1W│
        └────┘ └──────┘ └──────┘ └───────┘ └──────────┘
```

### 3.1 Sequencer
顺序、单发射：每周期取指/译码/派发 1 条。直线代码（编译器展开循环，无分支）。
长延迟单元（MXU、DMA）异步执行，由显式 `WAIT_MXU`/`DMA_FENCE` 暴露完成。
停顿按原因分类计数（acquire / dma / mxu / vpu / inject）。

### 3.2 MXU（矩阵单元）
- 原生 16×16×16；发射间隔 II=`mxu_issue_interval`（1），延迟 `mxu_latency`（4）
  周期后结果提交至 LOCAL SRAM。每次发射的操作数印记固定 **512 B（A）+ 512 B（B）**。
- **精度档**（同一 systolic 阵列，同 II/延迟；一次发射打包的 K 深度随精度变）：

  | 指令 | A / B | C | 每发射打包 K-block | 相对吞吐 | 元素宽 |
  |:--|:--|:--|:--:|:--:|:--:|
  | `MXU_F16F16` | f16 / f16 | f32 | 1（16 深） | 1× | 2 B |
  | `MXU_F32F16` | f32 / f16 | f32 | 1 | 1× | 2 B |
  | `MXU_I8I8` | int8 / int8 | int32 | 2（32 深） | **2×** | 1 B |
  | `MXU_I4I4` | int4 / int4 | int32 | 4（64 深） | **4×** | 0.5 B |

  低精度 MMA 是**时序模型**（§10）：更低精度每发射覆盖更多 K，故同一 GEMM 的 MXU
  发射数按 ½/¼ 缩放、DRAM 操作数字节同比缩小；数值不验（实测见 §16）。
- **在飞累加器转发**：对同一 C 块的连续累加（`acc=1`）从在飞流水读取最新值，
  不经 SRAM 往返（output-stationary 真实行为）。累加以旧 C 播种、k 升序——f16 下
  比特可复现（`blockops.h::mxu`）。这使 GEMM K 内层可**无 `WAIT_MXU` 连续累加**
  （双缓冲内层的前提，§3.7）。

### 3.3 VPU（向量单元，规范扩展）
16 f32 lane。原 MOBOL 只有 MXU；侧端 LLM 的 softmax/归一化/门控/RoPE 无法在
MXU 上执行，故 VPU 为**必需扩展**。逐拍计费（`vpu:` 延迟表）。算子见 §4。

### 3.4 DMA 引擎
- 每 tile 1 通道；描述符队列（深 `dma_queue_depth`=16）→ `dma_setup_cycles`(2)
  建立 → 每周期发射 `dma_chunks_per_cycle` 个 ≤64 B 子事务；在飞上限
  `dma_max_inflight_chunks`(64)。描述符**串行 FIFO** 处理（一次一个 active）。
- 2D strided（rows × row_bytes，独立 src/dst stride）。
- 推（本地→远端）发 `DATA_WRITE` flit，`WRITE_ACK` 回；拉（远端→本地）发
  `READ_REQ`，`READ_RESP` 携数据回并提交 LOCAL。自别名（src、dst 均本 tile）走
  纯 SRAM 快路径。
- **按发射序退休**：描述符即便物理完成较早（`READ_RESP` 乱序返回），也要等所有更早
  发射的描述符先退休；因此"未完成描述符集合"恒为**最近发射的 N 个**（`live_descs_`
  以 seq 有序）。这是 `DMA_FENCE keep=N` 部分排空语义的正确性基础（§3.7）。
- **完成计费**：`WRITE_ACK` 与 `READ_RESP` 均触发发起方回调（无论读写），故写 DMA
  的完成同样可被 fence/同步观察。

### 3.5 LOCAL SRAM 端口
2 读 + 1 写口（`local_sram.rd_ports/wr_ports`），每口 64 B/周期，逐拍在 DMA 读、
fabric 入口写、远程读服务间仲裁；争用计数。MXU/VPU 操作数走专用锁存通路
（`compute_uses_sram_ports=false`；置 true 则争用端口，用于敏感性研究）。

### 3.6 消费端口 DSE 旋钮 W
`vbond.flits_per_cycle`（tile↔bank 链路宽）、`local_sram.wr_ports`、
`dma.chunks_per_cycle` 联动放大 = W。W=1 时每 tile 64 B/周期消费；DSE 表明 W 与
垂直密度互补（见 REPORT.md §8.1）。

### 3.7 双/多缓冲 GEMM 内层（compute–memory overlap）
GEMM 输出块驻留映射下，K 内层原本 `load A[k]→load B[k]→DMA_FENCE→MXU→WAIT_MXU`
完全串行。**depth-D 多缓冲**：D 个操作数槽，循环始终把 D−1 个 K 步的操作数预取在飞
（`EDGC_DBUF_DEPTH`，默认 3）：

```
prologue: 预取前 D-1 步操作数
loop k:   预取第 k+(D-1) 步  →  DMA_FENCE keep=2·ahead  →  MXU(slot k%D, acc=k>0)
epilogue: WAIT_MXU
```

- `keep=2·ahead` 只排空当前步的 2 个操作数（A、B），预取的后续步留在飞 → **DMA 引擎
  与 MXU 重叠**；步间无 `WAIT_MXU`（靠在飞累加器转发链住，§3.2）。
- 观测计数器（per-tile）：`dma_busy_cycles`（DMA 引擎搬运拍数）、
  `mxu_dma_overlap_cycles`（MXU 与 DMA 同拍在跑）。overlap 比 = overlap/mxu_busy，
  DMA-BW 利用率 = dma_busy/总拍。
- **实测**（纯 GEMM M=N=128，K=512）：depth-2 overlap 57% / DMA-BW 66%；**depth-3
  达 overlap 89% / DMA-BW 87%**（DRAM 读延迟 ~40 拍需 ≥3 步在飞才盖住）。详见 §16。

### 3.8 停顿与利用率计数（模拟器观测面）
每 tile 逐拍归因：`busy`(发射)、`stall_acquire`(等 tag)、`stall_dma`(fence/队满/
sync ld-st)、`stall_mxu`(II 节流/WAIT_MXU)、`stall_vpu`、`stall_inject`(fabric 背压)；
以及 `mxu_busy`/`vpu_busy`/`dma_busy`/`mxu_dma_overlap`。芯片 MXU 占用 =
Σ`mxu_busy`/(16·总拍)。`trace_sim --json` 导出全部 per-tile 明细。

---

## 4. 指令集（Tile ISA）

编译器静态生成，文本 trace 由 `trace_loader.cpp` 解析。字段依 opcode 解释。

| 类 | 指令 | 语义 |
|:--|:--|:--|
| DMA | `DMA src dst rows row_bytes src_stride dst_stride` | 异步 2D 拷贝（任意 PGAS 段） |
| | `DMA_FENCE [keep=N]` | 阻塞至未完成 DMA 描述符 ≤ N（缺省 N=0 = 全量 fence，§3.4/§3.7） |
| MXU | `MXU_F16F16 / MXU_F32F16 a b d acc` | 16³ 矩阵乘（f16/f32 A），`acc` 累加至 f32 |
| | `MXU_I8I8 / MXU_I4I4 a b d acc` | 低精度 MMA，打包 2/4 K-block（2×/4× 吞吐），C int32（§3.2，时序模型） |
| | `WAIT_MXU` | 阻塞至 MXU 流水清空（结果已提交 SRAM） |
| VPU | `VPU_ADD_F32 / ADD_F16 / MUL_F32` | 逐元素 f32/f16 加、f32 乘 |
| | `VPU_CVT_F32_F16 / CVT_F16_F32` | 精度转换 |
| | `VPU_TRANS_F16` | 16×16 块转置 |
| | `VPU_SCALE_F32 a d scalar` | 标量缩放 |
| | `VPU_SILU_F32 / GELU_F32` | SiLU（x·σ(x)）/ tanh 近似 GELU |
| | `VPU_SOFTMAX_F32` | 行 softmax（scalar=缩放） |
| | `VPU_SOFTMAX_BLK a d scalar count aux rows row_bytes` | 跨 count 块行 softmax；`aux`=行 0 绝对位置（causal 掩码，<0 关）；`rows`=滑窗（0 关）；`row_bytes`=有效列 |
| | `VPU_RMSNORM_BLK / LAYERNORM_F32 a d scalar count` | 跨 count 块行归一化，scalar=eps |
| | `VPU_ROPE_F16 a d count aux` | 单头块 rotate-half RoPE；`count`=half；`aux`=cos/sin 表 LOCAL 偏移 |
| SHARED | `LOAD_SHARED / STORE_SHARED` | 近 bank 直接读写（远 bank trap） |
| NMC | `NMC src dst count tag arity[=NmcOp] scalar` | buffer-die 近存命令（§6.4）；完成回 release token |
| | `NMC …（arity=PREFETCH）` | bank 预取引擎：DRAM→本 bank 流式（§6.5） |
| 同步 | `RELEASE consumer tag` | 向 consumer 的 cnt[tag] 发布（自发布折叠为本地自增） |
| | `ACQUIRE tag arity` | 阻塞至 cnt[tag]≥arity，通过后 cnt-=arity |
| — | `NOP` / `HALT` | 空 / 停机 |

块布局（BLOCKED）：`count` 个连续 16×16 块并排为一个 16×(16·count) 逻辑行组，
元素 (i,j) 位于块 j/16 的局部索引 i·16+j%16。

---

## 5. NoC：双向环（base die）

16 节点单环，双向。每链路每方向每周期 1 flit（64 B）；每跳 `noc.hop_latency`(1)
周期；路由器输入缓冲深 `noc.buffer_depth`(4)，背压。

- **确定性路由**：最短路，平局取顺时针（`ring_direction`）。
- **2 虚拟通道**：VC0=请求（`DATA_WRITE`/`READ_REQ`），VC1=响应
  （`WRITE_ACK`/`READ_RESP`/`RELEASE_TOK`）。分离缓冲、响应优先，且响应在目标
  必可汇——**请求→响应协议级无死锁**（`fabric.cpp`）。
- **网关**：去往远组 SHARED 的流量沿环到目标组最近的 tile（网关），再垂直上行；
  响应在网关落下、经环回源。
- release token 作为 tile 定向 flit 传输，抵达即增 cnt[tag]，与 NoC 乱序无关。

---

## 6. Buffer die：共享 SRAM + 近存计算

### 6.1 SHARED bank
4 体，各 8 MB。bank 控制器逐拍轮询服务 4 条 tile 上行链路（宽度随 W），
per-bank 端口争用可开关（`shared_sram.port_contention`），访问计数常开。

### 6.2 融合驻留
生产者-消费者中间结果（如 GEMM1 的 C）驻留 SHARED，不回 DRAM。近访问 3 周期
vs 远 13 周期（微基准 T3）。

### 6.3 DRAM 直通
bank 控制器对 DRAM 段流量作透明转发（tile↔DRAM 经本组垂直列，bank 不介入数据）。

### 6.4 近存计算引擎 NMC（可选，paradigm C）
每 bank 侧一个 16-lane 引擎，命令驱动（`NMC_CMD` flit），完成发 release token：
- `REDUCE_F32`：固定序 f32 块归约（`count` 个 1 KB 块），比特等价于 tile-VPU 加链。
- `LN_F16`：融合 LayerNorm + f32→f16。
延迟 `nmc.*_cycles_per_block`。**设计要点**：归约须随 bank 分布（4 引擎并行），
集中单 bank 反而变慢（REPORT.md §8.3）。默认 `nmc.enable=false`。

### 6.5 权重流式预取引擎（可选）
每 bank 侧独立预取 DMA（与 NMC 并行）：命令式经**本组垂直列**直接从 DRAM 流式
拉权重进 bank SRAM，**全程不经 base die**；按层双缓冲、提前一层与计算重叠；零复制
切片分装（DRAM 字节守恒）。见 REPORT.md §8.5。

---

## 7. 3D 垂直互连（混合键合）

两级垂直键合，均 flit 级、每跳 `vbond.latency`(1) 周期：

1. **tile ↔ 本组 bank**：每 tile 一条私有键合，`vbond.flits_per_cycle`(=W) 个
   64 B flit/周期/方向。近 SHARED 访问即经此一跳。
2. **bank 列 ↔ DRAM**：每组一条密集键合列，密度 `vbond.dram_flits_per_cycle`
   个 flit/周期（关键 3D 旋钮）：1=512b, 2=1024b, 4=2048b, 8=4096b, 16=8192b
   每列每周期。这是"DRAM die 堆顶部、通过先进封装供超大带宽"的落点。

**DSE 拐点**：垂直密度 2048 bits/列/周期即饱和 base die 消费能力（每组 4 tile ×
64 B/周期）；再高需先加宽消费端口（W）或用 §6.5 预取绕过 tile 端口
（REPORT.md §3、§8.1）。远 bank 流量：环到网关 tile → 垂直上行。

---

## 8. DRAM die（顶层，Ramulator2 驱动）

- 器件模型：**Ramulator2 HBM3 级**（`config/ramulator_3d_dram.yaml`）：16 通道 ×
  128-bit、bank-group × bank 并行、32 B 原生访问粒度、OpenRow FR-FCFS 调度、
  REFab 刷新（自写 `AllBankNoRank`——Ramulator2 原生 AllBank 对无 rank 器件静默
  不刷新）。16 GB，核心峰值 ~256 GB/s。
- **时钟域**：逻辑 : DRAM 控制器比值取自 ramulator yaml 的 `MemorySystem.
  clock_ratio`（参考 1:2，500 MHz base / 1 GHz 控制器），是唯一真值源。控制器每
  逻辑周期推进 `ratio` 个 DRAM 周期，每 DRAM 周期最多 `dram.issue_width` 个并行
  通道发射。
- 一个 64 B fabric flit 在控制器处拆成 2×32 B 器件事务，全部完成才放行响应——
  链路满载计费与器件时序粒度两不误。
- **无 fallback**：缺 ramulator 配置直接报错（近似 DRAM 会使全部周期数失真）。
- 混合键合 DRAM（无 TSV/PHY serdes）真实行为可能更快；HBM3 时序为保守替身，
  流片前应替换为供应商 die 实测时序（结构支持任意 Ramulator2 器件）。

---

## 9. 同步：基于 tag 的 release/acquire + arity-k join

无锁、无缓存一致性、单向消息传递。三元组：`release(tag)` / `acquire(tag,arity)`
/ join 计数器 `cnt[tag]`（每消费者 tile 本地寄存器表，容量 `sync.max_live_tags`）。

- **release(tag)**：生产者声明"本 tile 此前对该 buffer 的写已到达消费者可观察
  位置"，绑定名字 tag。非完成信号而是**可读信号**。自发布折叠为本地 cnt 自增；
  远程发布经 NoC 传 `RELEASE_TOK`，抵达增 cnt。
- **acquire(tag,arity)**：阻塞至 `cnt[tag]≥arity`，通过后 `cnt-=arity`（代际
  自动复位）。
- **arity-k join**：k 个生产者分片（如 K 维归约）时消费者等 k 个 release。k 为
  编译期常数。硬件只计数比较，不监视地址。
- **正确性**：不漏项（计数达 k）、不早读（cnt<k 物理阻塞）、顺序无关（不关心
  token 到达序）、零地址监视。因果性由模拟器校验（每 acquire 通过时刻 ≥ 第 k 个
  匹配 release token 抵达时刻）。

PGAS 三正交支柱：**命名**（全局地址）+ **搬运**（DMA 描述符）+ **同步**（tag）。

### 9.1 依赖感知 barrier 放置（编译器策略）
全局 barrier（16 tile 全 release 到 tile0、tile0 arity-16 acquire、再广播回落）是
保守的相位串行化。编译器改为**只在真实数据依赖处插 barrier**：每个原语声明其读/写
的 DRAM buffer 集合，仅当与"上次 barrier 以来被读/写过的 buffer"发生 RAW/WAR/WAW
冲突时才插；相互独立的原语（三个 QKV 投影都只读同一 `xn`、MoE 各专家都只读 `hn`）
之间不插 barrier（读-读非冲突）。语义上永不去掉必要同步 → 正确性由构造保证。实测
省 barrier 17–33%、端到端 +4–12%（MoE 最高，专家独立），见 §16。

---

## 10. 数值规范

- I/O 与激活/权重：**f16**（IEEE binary16，RNE）。DRAM 中激活与权重、tile 间
  中间量均 f16。
- 累加：MXU 内部 **f32**，k 升序、以旧 C 播种连续累加 → 比特可复现。
- VPU 归一化/softmax/门控在 f32 内计算，写回按需转 f16。
- **超越函数**：softmax(exp)、GELU/SiLU(tanh/sigmoid)、norm(sqrt) 用 f32。模拟器
  用 C 库 `expf/tanhf/sqrtf`；编译器 golden/reference 经 ctypes 调**同一 libm 的
  f32 函数**并逐操作镜像运算序，保证三方（sim/golden/reference）比特一致。
  **流片注意**：真实硬件用定义好的 LUT/多项式而非 libm；流片前应把两侧统一换成
  RTL 采用的同一近似（工具链结构已就绪）。
- 归约顺序 `fixed`：固定树/线性合并序，牺牲部分调度自由换比特级确定性。
- **低精度（INT8/INT4）**：A/B int8/int4，累加 int32。当前实现为**纯时序模型**——
  只反映"每发射打包更多 K（2×/4× 吞吐）+ DRAM 操作数字节 ½/¼"的时序效果，不做真实
  整型运算、不验数值（§3.2/§16）。流片前若要功能正确，需补 int8/int4 定点数据通路与
  量化/反量化算子。f16 主链路的逐位确定性不受影响。

---

## 11. Host↔Device 接口：操作描述符

定长记录（数十字节），客户端写入共享队列、服务端读出执行。声明受依赖约束的纯
函数 `C = opcode(A,B) under {acc.dtype, reduce.order} subject to deps`：

| 字段 | 语义 |
|:--|:--|
| op_id | 实例唯一标识，被 deps 引用、作完成信号键 |
| opcode | 操作码（如 GEMM_F16），选数据通路与瓦片展开规则 |
| A/B/C.{base,shape,stride,dtype} | 操作数逻辑视图（base 高位含 die/bank/tile） |
| acc.dtype | 累加精度（如 f32） |
| tile.MNK | 瓦片尺寸（如 16,16,16） |
| reduce.order | K 维归约合并序（fixed = 比特确定） |
| deps | 前驱 op_id 列表（算子 DAG 的边） |

描述符是**请求非执行计划**：不含循环/搬运/瓦片指派——那些属瓦片调度。A/B 在
DRAM 或 SRAM 对描述符透明，仅 base 高位不同。descriptor.deps（粗粒度）与
release/acquire（细粒度数据就位）配合构成完整任务图调度。

---

## 12. 编译器（tools/edgc）

侧端 LLM 静态编译器：`ModelConfig`(metadata) → tile-ISA trace + DRAM 镜像 →
周期精准模拟器。支持 MHA/GQA/MQA、causal/滑窗、RoPE、RMS/LayerNorm、
SwiGLU/GELU、MoE（编译期静态路由）、投机解码。DSE 闭环调模拟器择优。双重比特级
校验（golden==reference 编译器对、golden==sim 模拟器对）。详见
`tools/edgc/REPORT_COMPILER.md`。结构常量与 DSE 搜索空间同样从统一 YAML 读取。

**代码生成优化（v1.1，均经模拟器实测，§16）**：
- 依赖感知 barrier 消除（§9.1，`EDGC_BARRIER=global` 复现旧全量 barrier 基线）。
- depth-D 多缓冲 GEMM 内层（§3.7，`EDGC_DBUF`/`EDGC_DBUF_DEPTH`）。
- 精度参数化 GEMM `prec∈{f16,i8,i4}`（§3.2），发 `MXU_I8I8/I4I4`、按精度调操作数字节。

**离线时序工具**（`tools/edgc/`，均纯时序）：`profile_mxu.py`（MXU/停顿画像）、
`bench_gemm.py`（Tier-3 纯 GEMM K/depth/精度扫描）、`decode_tier1.py`（Tier-1 真实
模型结构 decode 端到端）、`mxu_dse.py`（MXU 尺寸解析 DSE）、`bench_attention.py`
（head-parallel）。各自带 `REPORT_*.md` 与 `results/*.csv`。

---

## 13. 统一配置（单一真值源）

`config/mobol_arch.yaml` 管理**全部**架构参数，C++ 模拟器（`arch_config.cpp`）
与 Python 编译器（`edgc/arch.py`）读同一文件。层叠覆盖：
**arch YAML（基）→ trace 内 .config（每次编译的 DSE 旋钮）→ CLI 标志**。

分节：`structural`（拓扑，编译期校验）、`tile`/`vpu`（计算延迟）、
`local_sram`/`shared_sram`（端口）、`noc`、`vbond`（3D 密度）、`dma`、`dram`
（含 ramulator 路径与时钟比）、`nmc`、`sync`、`sim`、`compiler`（DSE 搜索空间）。
结构常量改动需重新编译（模拟器会校验并报错）；其余为运行时旋钮。

---

## 14. 周期时序模型（关键延迟，已知答案微基准锁定）

| 路径 | 周期 | 来源 |
|:--|:--|:--|
| 邻居 tile 64 B 推送端到端（含 ack） | 9 | 微基准 T1 |
| NoC 流带宽 | 1 flit/周期/链路 | T2 |
| 近 SHARED 拉取 | 3 | T3 |
| 远 SHARED 拉取（4 跳网关） | 13 | T3 |
| MXU 8 连发（II=1，延迟 4） | 8+4 | T4 |
| release token 飞行（邻居） | 3 | T5 |
| DRAM 冷读经 3D 列往返（HBM3@1:2） | 14 | T7 |
| 非法地址 / 远 bank 直访 | trap | T6 |

微基准（`test_cycle_micro.cpp`）逐拍锁定这些金值；任何时序变更必须是有意的并
更新金值。

---

## 15. 已验证性能包络（参考配置）

3D DRAM 密度 2048、500 MHz、baseline 调度：

| 负载 | 逻辑周期 | 墙钟 | 说明 |
|:--|:--|:--|:--|
| 双 GEMM（32³ f16） | 418 | 0.84 µs | 比特级正确，C 零 DRAM 往返 |
| Transformer 层 prefill（seq16,d64,h4,ff256） | 3.6k/层 | 7.2 µs/层 | 线性扩展 |
| 编译器 edge_gqa（2 层，GQA+RoPE+SwiGLU） | 11 055 | 22 µs | golden==sim 比特一致 |

架构演进（NMC + 端口加宽 W2 + 权重流式）叠加达 baseline 的 **−30%**
（REPORT.md §8）。侧端瓶颈画像：DRAM 供给（3D 键合）已充裕，瓶颈转向 base die
消费端口与行方向归约/同步串行化——对应的优化方向已量化验证。

---

## 16. 代码生成优化实测（f16 逐位一致，baseline 调度、密度 2048、500 MHz）

**16.1 依赖感知 barrier 消除（§9.1）——`EDGC_BARRIER=global` 逐位复现旧基线**

| 模型 | baseline 拍 | 消除后 | 加速 | barrier 数 | acquire 停顿占比 |
|:--|--:|--:|--:|--:|--:|
| edge_dense (MHA/LN/GELU) | 10 387 | 9 974 | 1.041× | 24→20 | 78% |
| edge_gqa (GQA4→2/RoPE/SwiGLU) | 11 055 | 10 603 | 1.043× | 30→22 | 75% |
| edge_mqa (MQA4→1) | 10 884 | 10 315 | 1.055× | 30→22 | 77% |
| edge_sliding (GQA+滑窗8) | 11 055 | 10 603 | 1.043× | 30→22 | 75% |
| **edge_moe (4 专家 top-2)** | 26 983 | 24 142 | **1.118×** | 62→42 | 69% |
| spec_target (GQA×4 层) | 22 316 | 21 109 | 1.057× | 60→44 | 76% |

关键：acquire 停顿降幅有限（3–11%），因其成因是**DMA 受限关键路径 + 半数 tile 空置**
（每模型仅 8/16 tile 跑 MXU），非 barrier 数——barrier 只是停顿被归因的位置。这正是
§3.7 双缓冲与 §18 head-parallel 的动机。

**16.2 多缓冲 GEMM 内层（§3.7）——叠加在 barrier 消除之上**

Tier-3 纯 GEMM（M=N=128，depth-3）稳态达标：

| K | 单缓冲拍 | 双缓冲拍 | overlap | DMA-BW | depth-2 vs depth-3(K=512) |
|--:|--:|--:|--:|--:|:--|
| 128 | 7 540 | 6 777 | 51% | 67% | — |
| 512 | 15 196 | 14 422 | 57% | 66% | overlap 57%/BW 66% → **89%/87%** |
| 1024 | 29 263 | 27 276 | 57% | 66% | depth≥4 收益饱和 |

端到端 edge 模型（默认 depth-3）再 +9.5–12.7%（edge_gqa 累计 11 055→9 358 = **1.18×**）。
成因：16³ MXU 每 K 步算术强度 ~4 MAC/B（载 1024 B / 4 拍 MXU），GEMM **访存受限**；多
缓冲把 DMA 引擎喂到 ~88% 占用，但计算量不足以完全盖住访存 → 上限由键合带宽定。

**16.3 低精度 MMA（§3.2/§10）——纯时序，M=N=128 depth-3**

| 精度 | K | cycles | MXU 发射 | DRAM 读 KB | vs f16 |
|:--|--:|--:|--:|--:|--:|
| f16 | 512 | 14 857 | 2 048 | 2 048 | 1.00× |
| i8 | 512 | 11 157 | 1 024 | 1 024 | 1.33× |
| i4 | 512 | 12 165 | 512 | 512 | 1.22× |
| f16 | 1024 | 27 104 | 4 096 | 4 096 | 1.00× |
| i8 | 1024 | 20 674 | 2 048 | 2 048 | 1.31× |
| i4 | 1024 | 17 039 | 1 024 | 1 024 | 1.59× |

MXU 发射数与 DRAM 字节严格按 ½/¼ 缩放。但端到端加速**亚线性**（1.3–1.6×，非 2×/4×）
且**非单调**（K=512 时 i4<i8）：输出块写回、MXU 排空、预取序段等固定开销不随精度缩小。
结论：**INT8 稳健，INT4 只在大 K 划算**。

---

## 17. Decode 阶段与 Tier-1 真实模型结构（纯时序，含 P1+P2）

真实网络结构（真实层数/hidden/heads/FFN），随机数据，shape-driven；每层 shape 相同，
故每个唯一 GEMM shape 精确仿真一次、组合出单层再 ×层数（层采样外推）。decode = 单 token
对 KV cache。

**17.1 端到端（batch=1，ctx=512，@500 MHz）**

| 模型 | 精度 | 层 | cyc/token | tok/s | MXU 利用率 | 活跃 tile | DRAM 读/token |
|:--|:--:|--:|--:|--:|--:|--:|--:|
| Llama-3.2-1B | f16 | 16 | 28.5 M | 17.5 | 3.45% | 16/16 | 4.0 GB |
| Llama-3.2-3B | f16 | 28 | 81.4 M | 6.1 | 3.49% | 16/16 | 11.6 GB |
| Phi-3.5-mini | i8 | 32 | 74.3 M | 6.7 | 2.44% | 16/16 | 7.5 GB |
| Mistral-7B-INT4 | i4 | 32 | 161.5 M | 3.1 | 1.08% | 16/16 | 7.1 GB |
| Mixtral-8x7B | f16 | 32 | 331.7 M | 1.5 | 3.75% | 16/16 | 51.0 GB |

**核心结论**：decode **极度访存受限**——N 维并行填满 16 个 tile，但 MXU 利用率仅 1–4%，
因为 (a) 单 token M=1，MXU 16×16 块只有 1/16 行是真数据；(b) 每 token 把整套权重从 DRAM
流一遍（4–51 GB）。INT4（Mistral）利用率最低但 DRAM 最省。

**17.2 Tier-2 维度扫描（Llama-3.2-1B）**

*batch（ctx=512）——batch≤16 免费提吞吐*：batch 1→8→16 **cycle 完全不变**，tok/s
17.5→140→**280**（16×）——因 decode 的 M-block 本有 16 行只填 1 行，batch 到 16 填满
空 padding。batch>16 转为线性（batch32 ≈ 287）。

*context（batch=1）——decode 对 context 不敏感*：ctx 128→512→2048，cyc/token
27.6M→28.5M→32.1M（仅 +16%）；注意力 ∝ L 但相对权重流（FFN 主导、与 L 无关）是小头。

---

## 18. MXU 尺寸解析 DSE（P3）与 head-parallel 注意力（P4-A）

**18.1 MXU 尺寸 vs. decode 利用率**（纸面：util = 真实 MAC / 发射 MAC 槽 = 加权）

| batch M | 8×8×8 | 16×16×16 | 32×32×32 |
|--:|--:|--:|--:|
| 1（单 token） | 12.5% | 6.2% | 3.1% |
| 16 | 100% | **100%** | 50% |
| 32 | 100% | 100% | 100% |

M=1 时 util = 1/S（M-padding 支配）；每种尺寸在 **batch = S** 处饱和。**推荐 16×16×16**：
在 batch=16（现实 decode 批量）满载，而 32³ 需 batch=32；8³ 达同等利用率需 ~8× 的
tile-pass（发射/同步开销翻倍）。M=1 的真正解药是 **batch（§17.2 已量化 16× 免费）**，
不是缩小 MXU。prefill（M 大）三尺寸皆近满，偏爱大 MXU（pass 最少）。

**18.2 head-parallel 注意力（decode，M=1，L=512）**

head h → tile (h mod 16)，各 head 独立、无 tile 间通信：

| 配置 | heads | 单 tile: 活跃/cyc | head-parallel: 活跃/cyc | 加速 | tile 利用率 |
|:--|--:|--:|--:|--:|--:|
| Llama-1B | 32 | 1 / 520 055 | **16** / 95 181 | 5.46× | 6.25% → **100%** |
| Llama-3B | 24 | 1 / 644 766 | **16** / 165 041 | 3.91× | 6.25% → **100%** |
| Mistral-7B | 32 | 1 / 859 630 | **16** / 195 518 | 4.40× | 6.25% → **100%** |
| edge (4h) | 4 | 1 / 34 501 | 4 / 9 099 | 3.79× | 6.25% → **25%** |

tile 利用率 ~1/16 → min(n_heads,16)/16。加速 3.8–5.5×（<tile 数）：填满阵列后瓶颈转到
**垂直键合带宽**（K/V 流），与 §17 "decode 访存受限" 一致。

---

## 19. 验证与复现

```bash
./run_all.sh                 # 架构：构建 + 6 项验证 + 全实验矩阵
cd tools/edgc && ./run_edgc.sh   # 编译器：5 模型比特级 + DSE + 投机解码
# v1.1 时序工具（纯时序）：
python3 tools/edgc/profile_mxu.py --sim build/src/trace_sim         # §16.1
python3 tools/edgc/bench_gemm.py  --sim build/src/trace_sim         # §16.2
python3 tools/edgc/bench_gemm.py  --sim build/src/trace_sim --int   # §16.3
python3 tools/edgc/decode_tier1.py --sim build/src/trace_sim        # §17
python3 tools/edgc/mxu_dse.py                                        # §18.1
python3 tools/edgc/bench_attention.py --sim build/src/trace_sim     # §18.2
```

设计原则回顾：**静态调度 + tag 同步 + PGAS 位比较寻址 → 比特级确定性**，这是流片
sign-off 的基石；**3D 堆叠 DRAM** 是把访存墙推开的关键；**VPU 与 buffer-die 近存
计算/预取**是让侧端 LLM 真正跑起来并高效的必要补全。

**v1.1 瓶颈画像（基于周期精准模拟器全量实测）**：侧端推理 **decode 阶段是访存受限**
——16 个 tile 都在动但 MXU 利用率仅 1–4%，因为单 token 的 M=1 让 MXU 块浪费 15/16 行、
且每 token 要流一遍整套权重。最有效的杠杆不是单点计算优化，而是 **batch 到 16**
（免费 16× 吞吐，填满 M-padding，§17.2/§18.1）+ **低精度权重流**（INT8/INT4，§16.3）。
compute-side 优化（barrier 消除 §16.1、多缓冲 §16.2、head-parallel §18.2）各自填满一
类空闲资源，直至**垂直键合带宽**成为最终天花板——这与 3D 堆叠"供给充裕、消费/搬运
是瓶颈"的整体画像一致。MXU 尺寸维持 **16×16×16**（§18.1：batch=16 满载的甜点）。
