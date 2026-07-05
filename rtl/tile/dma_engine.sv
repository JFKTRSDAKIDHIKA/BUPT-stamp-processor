// dma_engine.sv — per-tile DMA engine (ARCH_SPEC §3.4), NoC push path.
//
// Linear LOCAL->remote push: reads `nlines` 64 B lines from local SRAM at
// src_line, emits one DATA_WRITE flit per line to dst_tile carrying the dst
// LOCAL byte address (+64 per line). Per-line FSM (READ->CAP->SEND) — clean
// and correct with a 1-cycle-latency SRAM. 2-cycle setup models descriptor
// decode (ARCH_SPEC §3.4). (2D-strided descriptors add row/stride counters as
// in prefetch_engine.)
`include "common/mobol_pkg.sv"
`include "noc/noc_pkg.sv"
import noc_pkg::*;

module dma_engine #(
    parameter int ADDR_W = 12,
    parameter logic [TID_W-1:0] MY_ID = '0
) (
    input  logic              clk,
    input  logic              rst_n,
    input  logic              start,
    input  logic [ADDR_W-1:0] src_line,
    input  logic [TID_W-1:0]  dst_tile,
    input  logic [17:0]       dst_addr0,
    input  logic [15:0]       nlines,
    output logic              busy,
    output logic              done,
    output logic [ADDR_W-1:0] rd_line,
    input  logic [511:0]      rd_data,
    output flit_t             inj_req,
    input  logic              inj_req_ready
);
  typedef enum logic [2:0] {IDLE, SETUP, READ, CAP, SEND, FIN} st_e;
  st_e st;
  logic [1:0]  setup_cnt;
  logic [15:0] sent;
  logic [ADDR_W-1:0] cur_line;
  logic [17:0]  cur_addr;
  logic [511:0] line_buf;

  assign busy    = (st != IDLE);
  assign rd_line = cur_line;

  always_comb begin
    inj_req = '0;
    if (st == SEND) begin
      inj_req.valid   = 1'b1;
      inj_req.kind    = DATA_WRITE;
      inj_req.dst     = dst_tile;
      inj_req.src     = MY_ID;
      inj_req.addr    = cur_addr;
      inj_req.payload = line_buf;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin st<=IDLE; done<=1'b0; sent<=0; setup_cnt<=0; end
    else begin
      done<=1'b0;
      case (st)
        IDLE: if (start) begin
          cur_line<=src_line; cur_addr<=dst_addr0; sent<=0;
          setup_cnt<=2'd2; st<=SETUP;
        end
        SETUP: if (setup_cnt==0) st<=READ; else setup_cnt<=setup_cnt-2'd1;
        READ:  st<=CAP;                        // rd_line=cur_line presented
        CAP:   begin line_buf<=rd_data; st<=SEND; end
        SEND:  if (inj_req_ready) begin        // flit accepted
          sent<=sent+16'd1;
          cur_addr<=cur_addr+18'd64;
          cur_line<=cur_line+1'b1;
          if (sent+16'd1==nlines) st<=FIN; else st<=READ;
        end
        FIN: begin done<=1'b1; st<=IDLE; end
        default: st<=IDLE;                     // fully-specified (no latch)
      endcase
    end
  end
endmodule
