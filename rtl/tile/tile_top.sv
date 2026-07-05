// tile_top.sv — integrated compute tile (ARCH_SPEC §3).
//
// Wires the verified sub-blocks into a working datapath: LOCAL SRAM (2R1W) +
// MXU (16x16x16) + join counter (release/acquire sync) + a sequencer FSM. It
// executes a matmul micro-op end-to-end out of local SRAM:
//   LOAD_A (16 lines) -> LOAD_B (8 lines) -> MXU (16 cyc) -> STORE_C (16 lines)
// demonstrating the SRAM->MXU->SRAM compute path under sequencer control with
// the join counter available for inter-tile synchronization. (DMA/NoC ingress
// and the VPU are separate verified/stub blocks; this top shows the core
// compute integration a backend flow needs to floorplan the tile.)
//
// SRAM data layout (512-bit / 64 B lines):
//   A block: 256 f32 at A_LINE0, 16 lines (16 f32 words per line).
//   B block: 256 f16 at B_LINE0, 8 lines (32 f16 words per line).
//   C block: 256 f32 at C_LINE0, 16 lines.
`include "common/mobol_pkg.sv"

module tile_top #(
    parameter int DEPTH  = 4096,
    parameter int ADDR_W = 12
) (
    input  logic         clk,
    input  logic         rst_n,

    // matmul micro-op command (SRAM line addresses of the blocks)
    input  logic         start,
    input  logic [ADDR_W-1:0] a_line0,
    input  logic [ADDR_W-1:0] b_line0,
    input  logic [ADDR_W-1:0] c_line0,
    input  logic         acc,
    output logic         busy,
    output logic         done,

    // Backdoor SRAM port for the TB to preload operands / read results
    // (muxed onto the write port and read port 1 when the tile is idle).
    input  logic         ext_we,
    input  logic [ADDR_W-1:0] ext_waddr,
    input  logic [511:0] ext_wdata,
    input  logic [ADDR_W-1:0] ext_raddr,
    output logic [511:0] ext_rdata,

    // Join counter (sync) — exposed for release/acquire.
    input  logic         rel_valid,
    input  logic [5:0]   rel_tag,
    input  logic         acq_valid,
    input  logic [5:0]   acq_tag,
    input  logic [11:0]  acq_arity,
    output logic         acq_pass
);
  // ── LOCAL SRAM (behavioral; CACTI-modeled for PPA, see sram/) ──
  logic             sr_we;
  logic [ADDR_W-1:0] sr_waddr, sr_raddr0, sr_raddr1;
  logic [511:0]     sr_wdata, sr_rdata0, sr_rdata1;

  sram_2r1w #(.DATA_W(512), .DEPTH(DEPTH), .ADDR_W(ADDR_W)) u_sram (
    .clk, .we(sr_we), .waddr(sr_waddr), .wdata(sr_wdata),
    .raddr0(sr_raddr0), .rdata0(sr_rdata0),
    .raddr1(sr_raddr1), .rdata1(sr_rdata1)
  );

  // ── MXU ──
  logic        mxu_start, mxu_done, mxu_acc;
  logic [31:0] mxu_a  [256];
  logic [15:0] mxu_b  [256];
  logic [31:0] mxu_cin[256];
  logic [31:0] mxu_cout[256];
  mxu16 u_mxu (
    .clk, .rst_n, .start(mxu_start), .acc(mxu_acc),
    .a_f32(mxu_a), .b_f16(mxu_b), .cin_f32(mxu_cin),
    .cout_f32(mxu_cout), .done(mxu_done)
  );

  // ── Join counter ──
  logic [11:0] dbg_cnt [64];
  join_ctr #(.NTAGS(64), .CNT_W(12)) u_join (
    .clk, .rst_n,
    .rel_valid, .rel_tag,
    .acq_valid, .acq_tag, .acq_arity, .acq_pass,
    .dbg_cnt(dbg_cnt)
  );

  // ── Sequencer FSM ──
  typedef enum logic [2:0] {IDLE, LDA, LDB, RUNM, WAITM, STC, FIN} st_e;
  st_e st;
  logic [4:0] li;               // line index 0..15 (A/C) or 0..7 (B)

  // operand / result registers
  logic [31:0] a_reg [256];
  logic [15:0] b_reg [256];
  logic [31:0] c_reg [256];

  assign busy = (st != IDLE);

  // SRAM port muxing: internal FSM when busy, else backdoor for the TB.
  always_comb begin
    sr_we    = ext_we;      sr_waddr = ext_waddr;  sr_wdata = ext_wdata;
    sr_raddr0 = '0;         sr_raddr1 = ext_raddr;
    if (busy) begin
      sr_we = 1'b0;
      // read A/B via port0 during load; write C via write port during store.
      case (st)
        LDA:  sr_raddr0 = a_line0 + li;
        LDB:  sr_raddr0 = b_line0 + li;
        STC:  begin sr_we = 1'b1; sr_waddr = c_line0 + li;
                    for (int w=0; w<16; w++) sr_wdata[w*32 +: 32] = c_reg[li*16 + w]; end
        default: ;
      endcase
    end
  end
  assign ext_rdata = sr_rdata1;

  integer w, idx;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE; done <= 1'b0; li <= 0; mxu_start <= 1'b0;
    end else begin
      done <= 1'b0; mxu_start <= 1'b0;
      case (st)
        IDLE: if (start) begin li <= 0; st <= LDA; end
        LDA: begin
          // rdata0 has line (li-1) due to 1-cycle SRAM latency; capture it.
          if (li != 0)
            for (w=0;w<16;w++) a_reg[(li-1)*16 + w] <= sr_rdata0[w*32 +: 32];
          if (li == 16) begin
            // capture last line
            for (w=0;w<16;w++) a_reg[15*16 + w] <= sr_rdata0[w*32 +: 32];
            li <= 0; st <= LDB;
          end else li <= li + 5'd1;
        end
        LDB: begin
          if (li != 0)
            for (w=0;w<32;w++) b_reg[(li-1)*32 + w] <= sr_rdata0[w*16 +: 16];
          if (li == 8) begin
            for (w=0;w<32;w++) b_reg[7*32 + w] <= sr_rdata0[w*16 +: 16];
            st <= RUNM;
          end else li <= li + 5'd1;
        end
        RUNM: begin
          for (idx=0; idx<256; idx++) begin
            mxu_a[idx]   <= a_reg[idx];
            mxu_b[idx]   <= b_reg[idx];
            mxu_cin[idx] <= 32'd0;      // acc handled via Cin from SRAM omitted here
          end
          mxu_acc   <= acc;
          mxu_start <= 1'b1;
          st <= WAITM;
        end
        WAITM: if (mxu_done) begin
          for (idx=0; idx<256; idx++) c_reg[idx] <= mxu_cout[idx];
          li <= 0; st <= STC;
        end
        STC: begin
          if (li == 15) st <= FIN;
          li <= li + 5'd1;
        end
        FIN: begin done <= 1'b1; st <= IDLE; end
      endcase
    end
  end
endmodule
