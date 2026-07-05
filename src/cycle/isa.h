/// @file isa.h
/// @brief Tile instruction set for the cycle-accurate MOBOL simulator.
///
/// Each compute tile runs a static, straight-line program (no branches —
/// loops are unrolled by the schedule generator, matching the "compiler
/// statically schedules everything" philosophy of the architecture).
///
/// The sequencer is in-order, single-issue: fetch+decode+dispatch of one
/// instruction per cycle. Long-latency units (MXU, DMA) run asynchronously;
/// explicit WAIT/FENCE instructions expose their completion to the program,
/// exactly as the compiler would have to reason about them on real silicon.
#pragma once

#include "common/types.h"
#include <cstdint>
#include <string>
#include <vector>

namespace mobol::cycle {

enum class Op : uint8_t {
    NOP,

    // ── DMA (asynchronous; issue enqueues a descriptor) ──────
    // 2D strided copy: rows x row_bytes, independent src/dst strides.
    // src/dst are full 40-bit PGAS addresses (LOCAL/SHARED/DRAM).
    DMA,
    // Block until every previously issued DMA descriptor of this tile has
    // fully completed (all write-acks / read-data received). Spec §7 fence.
    DMA_FENCE,

    // ── MXU (pipelined: issue interval II, latency L) ────────
    // 16x16x16 matmul over LOCAL scratchpad offsets.
    //   MXU_F16F16: A f16, B f16 -> C f32
    //   MXU_F32F16: A f32, B f16 -> C f32
    // acc=true accumulates into C (in-flight forwarding models the MXU's
    // internal accumulator; C in SRAM is committed at pipeline exit).
    MXU_F16F16,
    MXU_F32F16,
    // Block until the MXU pipeline is empty (results committed to SRAM).
    WAIT_MXU,

    // ── VPU (synchronous vector unit, 16 f32 lanes) ──────────
    // All operate on 16x16 blocks at LOCAL offsets (a/b -> d).
    VPU_ADD_F32,      // d = a + b (elementwise f32)
    VPU_ADD_F16,      // d = f16(f32(a) + f32(b))
    VPU_CVT_F32_F16,  // d(f16 block) = convert(a(f32 block))
    VPU_CVT_F16_F32,  // d(f32 block) = convert(a(f16 block))
    VPU_TRANS_F16,    // d = transpose(a), f16 block
    VPU_SCALE_F32,    // d = a * scalar
    VPU_SOFTMAX_F32,  // d = rowwise softmax(a * scalar), fused
    VPU_GELU_F32,     // d = gelu(a), tanh approximation
    VPU_LAYERNORM_F32,// d = layernorm over rows spanning `count` adjacent
                      // 16x16 blocks (row width = 16*count), eps in scalar

    // ── Compiler-target extensions (edge-LLM ops; BLOCKED layout:
    //    `count` consecutive 16x16 blocks form one 16 x (16*count) row
    //    group, row i of the group = row i of every block) ──
    VPU_MUL_F32,      // d = a * b elementwise (f32 blocks)
    VPU_SILU_F32,     // d = a * sigmoid(a) elementwise
    VPU_RMSNORM_BLK,  // blocked rmsnorm over `count` f32 blocks, eps=scalar
    VPU_SOFTMAX_BLK,  // blocked row softmax over `count` f32 blocks:
                      //   scale=scalar; valid cols=row_bytes (0 => all);
                      //   aux = absolute position of row 0 (causal mask,
                      //   aux<0 => no mask); rows = sliding window (0=off)
    VPU_ROPE_F16,     // rotate-half RoPE on one 16-wide head block (f16):
                      //   a = head block (16x16 f16, cols = the 16 head
                      //   dims), d = rotated block. `count` = half = dims/2;
                      //   aux = LOCAL offset of cos(1KB f32) then sin(1KB),
                      //   where column k<half holds cos/sin(pos,k). Pairs
                      //   dim k with dim k+half:
                      //     out[k]      = x[k]*cos - x[k+half]*sin
                      //     out[k+half] = x[k+half]*cos + x[k]*sin

    // ── Direct load/store to the near SHARED bank (Q9) ───────
    // Synchronous: sequencer stalls for the vertical round trip.
    // Trap if the bank is not this tile's group bank.
    LOAD_SHARED,      // LOCAL[dst_off] <- SHARED[src_addr], size bytes
    STORE_SHARED,     // SHARED[dst_addr] <- LOCAL[src_off], size bytes

    // ── Near-memory compute on the buffer die (paradigm C) ───
    // Issue an NMC command flit to the bank owning `src` (must be the
    // near bank). Asynchronous: the bank engine executes and sends a
    // release token to (this tile, tag) on completion — pair with
    // ACQUIRE(tag, 1).
    NMC,

    // ── Synchronization (tag release/acquire, MOBOL spec §3) ─
    // RELEASE: notify consumer tile's join counter. Self-release is folded
    // to a local counter increment; remote release travels the NoC as a
    // token flit and increments on arrival.
    RELEASE,
    // ACQUIRE: block until cnt[tag] >= arity, then consume (cnt -= arity).
    ACQUIRE,

    HALT,
};

/// One decoded instruction. Fields are interpreted per opcode.
struct Instr {
    Op op = Op::NOP;

    // DMA / LOAD / STORE
    Addr src = 0;             ///< full PGAS address (DMA src, LOAD src)
    Addr dst = 0;             ///< full PGAS address (DMA dst, STORE dst)
    uint32_t rows = 1;        ///< 2D DMA: number of rows
    uint32_t row_bytes = 0;   ///< 2D DMA: contiguous bytes per row
    int64_t src_stride = 0;   ///< bytes between row starts at src
    int64_t dst_stride = 0;   ///< bytes between row starts at dst

    // MXU / VPU (LOCAL scratchpad byte offsets)
    uint32_t a_off = 0;
    uint32_t b_off = 0;
    uint32_t d_off = 0;       ///< destination block offset
    bool acc = false;         ///< MXU: accumulate into destination
    float scalar = 0.0f;      ///< VPU scale/softmax/layernorm parameter
    uint32_t count = 1;       ///< VPU_LAYERNORM: blocks per row group

    // Sync
    uint32_t tag = 0;         ///< tag index (scoped to consumer tile)
    TileId consumer = 0;      ///< RELEASE: tile whose counter to bump
    uint32_t arity = 0;       ///< ACQUIRE: producers to wait for

    int32_t aux = 0;          ///< op-specific extra (softmax row pos, rope table)

    std::string note;         ///< annotation for traces
};

/// A per-tile program plus a chip-wide container.
struct Program {
    std::vector<Instr> code[NUM_TILES];

    void add(TileId t, Instr i) { code[t].push_back(std::move(i)); }
    size_t total_instrs() const {
        size_t n = 0;
        for (const auto& c : code) n += c.size();
        return n;
    }
};

// ─── Instruction constructors (schedule-builder helpers) ─────

inline Instr mk_dma(Addr src, Addr dst, uint32_t rows, uint32_t row_bytes,
                    int64_t src_stride, int64_t dst_stride, std::string note = "") {
    Instr i; i.op = Op::DMA; i.src = src; i.dst = dst; i.rows = rows;
    i.row_bytes = row_bytes; i.src_stride = src_stride; i.dst_stride = dst_stride;
    i.note = std::move(note); return i;
}

inline Instr mk_dma_linear(Addr src, Addr dst, uint32_t bytes, std::string note = "") {
    return mk_dma(src, dst, 1, bytes, 0, 0, std::move(note));
}

inline Instr mk_dma_fence() { Instr i; i.op = Op::DMA_FENCE; return i; }

inline Instr mk_mxu(Op op, uint32_t a, uint32_t b, uint32_t d, bool acc,
                    std::string note = "") {
    Instr i; i.op = op; i.a_off = a; i.b_off = b; i.d_off = d; i.acc = acc;
    i.note = std::move(note); return i;
}

inline Instr mk_wait_mxu() { Instr i; i.op = Op::WAIT_MXU; return i; }

inline Instr mk_vpu(Op op, uint32_t a, uint32_t b, uint32_t d,
                    float scalar = 0.0f, uint32_t count = 1) {
    Instr i; i.op = op; i.a_off = a; i.b_off = b; i.d_off = d;
    i.scalar = scalar; i.count = count; return i;
}

inline Instr mk_vpu_aux(Op op, uint32_t a, uint32_t b, uint32_t d,
                        float scalar, uint32_t count, int32_t aux,
                        uint32_t rows = 0, uint32_t row_bytes = 0) {
    Instr i; i.op = op; i.a_off = a; i.b_off = b; i.d_off = d;
    i.scalar = scalar; i.count = count; i.aux = aux;
    i.rows = rows; i.row_bytes = row_bytes; return i;
}

/// NMC command: op over `count` blocks at SHARED `src`, result to SHARED
/// `dst`; completion token to (issuing tile, tag). scalar = LN epsilon.
inline Instr mk_nmc(uint32_t nmc_op, Addr src, Addr dst, uint32_t count,
                    uint32_t tag, float scalar = 0.0f, std::string note = "") {
    Instr i; i.op = Op::NMC; i.src = src; i.dst = dst; i.count = count;
    i.tag = tag; i.scalar = scalar; i.arity = nmc_op; // arity carries NmcOp
    i.note = std::move(note); return i;
}

/// Bank prefetch command (weight streaming): stream a 2D region from
/// DRAM `src` into SHARED `dst` (the destination bank's own SRAM) via
/// that bank's vertical column. Completion token to (this tile, tag).
inline Instr mk_nmc_prefetch(Addr src_dram, Addr dst_shared, uint32_t rows,
                             uint32_t row_bytes, int64_t src_stride,
                             int64_t dst_stride, uint32_t tag,
                             std::string note = "") {
    Instr i; i.op = Op::NMC; i.src = src_dram; i.dst = dst_shared;
    i.rows = rows; i.row_bytes = row_bytes;
    i.src_stride = src_stride; i.dst_stride = dst_stride;
    i.tag = tag; i.arity = 3; // NmcOp::PREFETCH
    i.note = std::move(note); return i;
}

inline Instr mk_release(TileId consumer, uint32_t tag, std::string note = "") {
    Instr i; i.op = Op::RELEASE; i.consumer = consumer; i.tag = tag;
    i.note = std::move(note); return i;
}

inline Instr mk_acquire(uint32_t tag, uint32_t arity, std::string note = "") {
    Instr i; i.op = Op::ACQUIRE; i.tag = tag; i.arity = arity;
    i.note = std::move(note); return i;
}

inline Instr mk_halt() { Instr i; i.op = Op::HALT; return i; }

} // namespace mobol::cycle
