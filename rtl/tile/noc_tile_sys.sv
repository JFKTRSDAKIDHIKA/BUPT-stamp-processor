// noc_tile_sys.sv — 16 tile_nodes wired to the ring NoC (ARCH_SPEC §3-5).
// The DMA+NoC integration: any tile can push a LOCAL region to any other
// tile's LOCAL SRAM through the fabric. Per-tile scalar command/backdoor ports
// for the C++ TB.
`include "common/mobol_pkg.sv"
`include "noc/noc_pkg.sv"
import noc_pkg::*;

module noc_tile_sys #(
    parameter int ADDR_W = 12
) (
    input  logic clk,
    input  logic rst_n,

    // per-tile DMA command
    input  logic [NUM_TILES-1:0]       dma_start,
    input  logic [ADDR_W-1:0]          dma_src_line  [NUM_TILES],
    input  logic [TID_W-1:0]           dma_dst_tile  [NUM_TILES],
    input  logic [17:0]                dma_dst_addr0 [NUM_TILES],
    input  logic [15:0]                dma_nlines    [NUM_TILES],
    output logic [NUM_TILES-1:0]       dma_done,

    // per-tile backdoor SRAM
    input  logic [NUM_TILES-1:0]       ext_we,
    input  logic [ADDR_W-1:0]          ext_waddr [NUM_TILES],
    input  logic [511:0]               ext_wdata [NUM_TILES],
    input  logic [ADDR_W-1:0]          ext_raddr [NUM_TILES],
    output logic [511:0]               ext_rdata [NUM_TILES]
);
  flit_t inj_req  [NUM_TILES], inj_resp [NUM_TILES];
  logic  inj_req_ready [NUM_TILES], inj_resp_ready [NUM_TILES];
  flit_t eject_cw [NUM_TILES], eject_ccw [NUM_TILES];

  genvar t;
  generate
    for (t=0;t<NUM_TILES;t++) begin : g_tile
      assign inj_resp[t] = '0;                 // this integration uses VC0 only
      tile_node #(.ADDR_W(ADDR_W), .MY_ID(t[TID_W-1:0])) u_node (
        .clk, .rst_n,
        .dma_start(dma_start[t]), .dma_src_line(dma_src_line[t]),
        .dma_dst_tile(dma_dst_tile[t]), .dma_dst_addr0(dma_dst_addr0[t]),
        .dma_nlines(dma_nlines[t]), .dma_done(dma_done[t]),
        .ext_we(ext_we[t]), .ext_waddr(ext_waddr[t]), .ext_wdata(ext_wdata[t]),
        .ext_raddr(ext_raddr[t]), .ext_rdata(ext_rdata[t]),
        .inj_req(inj_req[t]), .inj_req_ready(inj_req_ready[t]),
        .eject_cw(eject_cw[t]), .eject_ccw(eject_ccw[t])
      );
    end
  endgenerate

  ring_noc u_noc (
    .clk, .rst_n,
    .inj_req, .inj_resp, .inj_req_ready, .inj_resp_ready,
    .eject_cw, .eject_ccw
  );
endmodule
