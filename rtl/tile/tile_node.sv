// tile_node.sv — a tile's fabric-facing datapath: LOCAL SRAM + DMA engine +
// NoC ingress, with injection/eject to the ring (ARCH_SPEC §3-5). Demonstrates
// the DMA+NoC integration: a DMA push reads local SRAM and streams DATA_WRITE
// flits to a remote tile, whose ingress writes them into its SRAM. Write-acks
// and the compute core (MXU/VPU/join, see tile_top) attach at the same SRAM/
// inject/eject boundary; kept lean here to verify the fabric path.
`include "common/mobol_pkg.sv"
`include "noc/noc_pkg.sv"
import noc_pkg::*;

module tile_node #(
    parameter int ADDR_W = 12,
    parameter logic [TID_W-1:0] MY_ID = '0
) (
    input  logic  clk,
    input  logic  rst_n,

    // DMA command (from the sequencer)
    input  logic              dma_start,
    input  logic [ADDR_W-1:0] dma_src_line,
    input  logic [TID_W-1:0]  dma_dst_tile,
    input  logic [17:0]       dma_dst_addr0,
    input  logic [15:0]       dma_nlines,
    output logic              dma_done,

    // backdoor SRAM access for the TB (load/inspect)
    input  logic              ext_we,
    input  logic [ADDR_W-1:0] ext_waddr,
    input  logic [511:0]      ext_wdata,
    input  logic [ADDR_W-1:0] ext_raddr,
    output logic [511:0]      ext_rdata,

    // NoC ports
    output flit_t             inj_req,
    input  logic              inj_req_ready,
    input  flit_t             eject_cw,
    input  flit_t             eject_ccw
);
  // ── LOCAL SRAM (behavioral; CACTI macro at synth) ──
  logic              sr_we;
  logic [ADDR_W-1:0] sr_waddr, sr_raddr0, sr_raddr1;
  logic [511:0]      sr_wdata, sr_rdata0, sr_rdata1;
  sram_2r1w #(.DATA_W(512), .DEPTH(1<<ADDR_W), .ADDR_W(ADDR_W)) u_sram (
    .clk, .we(sr_we), .waddr(sr_waddr), .wdata(sr_wdata),
    .raddr0(sr_raddr0), .rdata0(sr_rdata0),
    .raddr1(sr_raddr1), .rdata1(sr_rdata1)
  );

  // ── DMA engine (uses read port 0) ──
  logic [ADDR_W-1:0] dma_rd_line;
  logic              dma_busy;
  dma_engine #(.ADDR_W(ADDR_W), .MY_ID(MY_ID)) u_dma (
    .clk, .rst_n, .start(dma_start),
    .src_line(dma_src_line), .dst_tile(dma_dst_tile),
    .dst_addr0(dma_dst_addr0), .nlines(dma_nlines),
    .busy(dma_busy), .done(dma_done),
    .rd_line(dma_rd_line), .rd_data(sr_rdata0),
    .inj_req(inj_req), .inj_req_ready(inj_req_ready)
  );

  // ── NoC ingress: a DATA_WRITE flit from either direction writes its
  //    payload into local SRAM at flit.addr (line = addr>>6). One write/cycle;
  //    CW has priority (the two directions rarely eject the same cycle). ──
  flit_t ing;
  always_comb begin
    ing = '0;
    if (eject_cw.valid && eject_cw.kind == DATA_WRITE)       ing = eject_cw;
    else if (eject_ccw.valid && eject_ccw.kind == DATA_WRITE) ing = eject_ccw;
  end

  // ── SRAM port muxing ──
  //   write port : ingress DATA_WRITE, else TB backdoor.
  //   read port0 : DMA, else TB backdoor via port1.
  always_comb begin
    if (ing.valid) begin
      sr_we = 1'b1; sr_waddr = ing.addr[ADDR_W+5:6]; sr_wdata = ing.payload;
    end else begin
      sr_we = ext_we; sr_waddr = ext_waddr; sr_wdata = ext_wdata;
    end
    sr_raddr0 = dma_rd_line;
    sr_raddr1 = ext_raddr;
  end
  assign ext_rdata = sr_rdata1;
endmodule
