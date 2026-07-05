// ssp_bank.sv — one buffer-die SHARED scratchpad bank (ARCH_SPEC §6.1).
//
// 8 MB = 131072 x 512-bit lines, built from 32 mobol_sram_4096x512_2r1w
// macros (synth/fakeram/gen_sram_macros.py). Same port discipline as the
// tile LOCAL scratchpad (2 read + 1 write, 64 B lines, 1-cycle synchronous
// read, read-old-data on a same-address write) so the bank controller /
// NMC / prefetch engines attach exactly like tile logic attaches to
// sram_2r1w. The address here is a LINE address (byte addr >> 6).
//
// Composition rules that keep it cycle-equivalent to one flat 2R1W array:
//   - the write strobes exactly one sub-macro (upper address bits decode);
//   - every read port activates the sub-macro it addresses (ce_in);
//   - the read mux select is REGISTERED, because data shows up one cycle
//     after the address — total latency stays 1 cycle.
`include "common/mobol_pkg.sv"

module ssp_bank #(
    parameter int DATA_W      = 512,            // 64 B line
    parameter int DEPTH       = (1<<23)/64,     // 8 MB / 64 B = 131072 lines
    parameter int ADDR_W      = $clog2(DEPTH),  // 17
    parameter int MACRO_DEPTH = 4096,
    parameter int MACRO_AW    = $clog2(MACRO_DEPTH),
    parameter int NSUB        = DEPTH / MACRO_DEPTH,   // 32
    parameter int SUB_W       = $clog2(NSUB)           // 5
) (
    input  logic              clk,
    // write port
    input  logic              we,
    input  logic [ADDR_W-1:0] waddr,
    input  logic [DATA_W-1:0] wdata,
    // read port 0
    input  logic [ADDR_W-1:0] raddr0,
    output logic [DATA_W-1:0] rdata0,
    // read port 1
    input  logic [ADDR_W-1:0] raddr1,
    output logic [DATA_W-1:0] rdata1
);
  // Upper address bits select the sub-macro.
  wire [SUB_W-1:0] wsel  = waddr [ADDR_W-1:MACRO_AW];
  wire [SUB_W-1:0] r0sel = raddr0[ADDR_W-1:MACRO_AW];
  wire [SUB_W-1:0] r1sel = raddr1[ADDR_W-1:MACRO_AW];

  // Read data arrives one cycle after the address: delay the mux select.
  logic [SUB_W-1:0] r0sel_q, r1sel_q;
  always_ff @(posedge clk) begin
    r0sel_q <= r0sel;
    r1sel_q <= r1sel;
  end

  logic [DATA_W-1:0] rd0_sub [NSUB];
  logic [DATA_W-1:0] rd1_sub [NSUB];

  genvar g;
  generate
    for (g = 0; g < NSUB; g++) begin : g_sub
      wire w_hit  = we && (wsel == SUB_W'(g));
      wire r0_hit = (r0sel == SUB_W'(g));
      wire r1_hit = (r1sel == SUB_W'(g));

      // ce_in gates the clock inside the macro: enable it only when this
      // sub-macro is written or addressed by a read (power saving; an
      // un-addressed macro's stale rd*_out is never selected by the mux).
      mobol_sram_4096x512_2r1w u_macro (
        .clk       (clk),
        .ce_in     (w_hit | r0_hit | r1_hit),
        .we_in     (w_hit),
        .waddr_in  (waddr [MACRO_AW-1:0]),
        .wd_in     (wdata),
        .raddr0_in (raddr0[MACRO_AW-1:0]),
        .rd0_out   (rd0_sub[g]),
        .raddr1_in (raddr1[MACRO_AW-1:0]),
        .rd1_out   (rd1_sub[g])
      );
    end
  endgenerate

  assign rdata0 = rd0_sub[r0sel_q];
  assign rdata1 = rd1_sub[r1sel_q];
endmodule
