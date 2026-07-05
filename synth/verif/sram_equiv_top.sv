// sram_equiv_top.sv — equivalence harness: behavioral SRAM vs macro views.
//
// Two independent pairs, compared cycle-by-cycle under the same stimulus:
//   LSP pair: sram_2r1w (behavioral, 4096 x 512)  vs  the 2R1W fakeram macro
//             wired exactly as rtl/sram/sram_2r1w_macro.sv wires it.
//   SSP pair: sram_2r1w (behavioral, 131072 x 512) vs ssp_bank (32 macros).
// A mismatch output goes high the cycle any read port differs; the C++
// driver (tb_sram_equiv.cpp) randomizes ports with heavy address collisions
// to exercise read-during-write ordering.
module sram_equiv_top (
    input  logic         clk,

    // LSP stimulus
    input  logic         we,
    input  logic [11:0]  waddr,
    input  logic [511:0] wdata,
    input  logic [11:0]  raddr0,
    input  logic [11:0]  raddr1,
    output logic         lsp_mismatch,

    // SSP stimulus
    input  logic         swe,
    input  logic [16:0]  swaddr,
    input  logic [511:0] swdata,
    input  logic [16:0]  sraddr0,
    input  logic [16:0]  sraddr1,
    output logic         ssp_mismatch
);
  // ── LSP pair ──────────────────────────────────────────────
  logic [511:0] ref0, ref1, dut0, dut1;

  sram_2r1w #(.DATA_W(512), .DEPTH(4096)) u_lsp_ref (
    .clk, .we, .waddr, .wdata,
    .raddr0, .rdata0(ref0), .raddr1, .rdata1(ref1)
  );

  // same wiring as sram_2r1w_macro.sv (ce tied high)
  mobol_sram_4096x512_2r1w u_lsp_dut (
    .clk(clk), .ce_in(1'b1),
    .we_in(we), .waddr_in(waddr), .wd_in(wdata),
    .raddr0_in(raddr0), .rd0_out(dut0),
    .raddr1_in(raddr1), .rd1_out(dut1)
  );

  assign lsp_mismatch = (ref0 != dut0) || (ref1 != dut1);

  // ── SSP pair ──────────────────────────────────────────────
  logic [511:0] sref0, sref1, sdut0, sdut1;

  sram_2r1w #(.DATA_W(512), .DEPTH(131072)) u_ssp_ref (
    .clk, .we(swe), .waddr(swaddr), .wdata(swdata),
    .raddr0(sraddr0), .rdata0(sref0), .raddr1(sraddr1), .rdata1(sref1)
  );

  ssp_bank u_ssp_dut (
    .clk, .we(swe), .waddr(swaddr), .wdata(swdata),
    .raddr0(sraddr0), .rdata0(sdut0), .raddr1(sraddr1), .rdata1(sdut1)
  );

  assign ssp_mismatch = (sref0 != sdut0) || (sref1 != sdut1);
endmodule
