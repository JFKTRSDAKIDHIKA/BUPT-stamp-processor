// ring_router.sv — one ring node, one direction (ARCH_SPEC §5).
//
// Per VC (0=req, 1=resp) a 1-flit input buffer. A node accepts an upstream
// flit only when its buffer is empty (in_ready = !buf.valid — a REGISTERED
// signal, so the ring-wide backpressure has no combinational loop). Each
// cycle a buffered flit ejects (if it terminates here) or forwards to the
// downstream buffer (if that is empty); the outgoing link, when not used by a
// forward, carries a locally-injected flit. Responses (VC1) take the eject
// port and the link ahead of requests (VC0) — deadlock-free.
//
// Throughput is one flit / 2 cycles per node (a 1-deep buffer bubbles); this
// is a functional model — the C++ cycle-accurate simulator carries the
// precise timing. The datapath (buffers, muxes, arbiters) is what backend PPA
// needs and is all present.
`include "noc/noc_pkg.sv"
import noc_pkg::*;

module ring_router #(
    parameter logic [TID_W-1:0] NODE_ID = '0
) (
    input  logic  clk,
    input  logic  rst_n,
    input  flit_t in_req,
    input  flit_t in_resp,
    output logic  in_req_ready,      // = !buf.valid (registered)
    output logic  in_resp_ready,
    output flit_t out_req,
    output flit_t out_resp,
    input  logic  out_req_ready,     // downstream !buf.valid
    input  logic  out_resp_ready,
    input  flit_t inj_req,
    input  flit_t inj_resp,
    output logic  inj_req_ready,
    output logic  inj_resp_ready,
    output flit_t eject,
    input  logic  eject_ready
);
  flit_t br_req, br_resp;

  wire req_here  = br_req.valid  && (br_req.dst  == NODE_ID);
  wire resp_here = br_resp.valid && (br_resp.dst == NODE_ID);
  wire req_thru  = br_req.valid  && !req_here;
  wire resp_thru = br_resp.valid && !resp_here;


  // Ejection: response first.
  logic ej_resp, ej_req;
  always_comb begin
    eject   = '0;
    ej_resp = resp_here && eject_ready;
    ej_req  = !ej_resp && req_here && eject_ready;
    if (ej_resp)      eject = br_resp;
    else if (ej_req)  eject = br_req;
    eject.valid = ej_resp | ej_req;
  end

  // Outgoing VC1 (resp): forward beats inject, unless injection is starving.
  logic fwd_resp, inj_resp_take;
  always_comb begin
    out_resp = '0; fwd_resp = 1'b0; inj_resp_take = 1'b0;
    if (resp_thru && out_resp_ready) begin
      out_resp = br_resp; out_resp.valid = 1'b1; fwd_resp = 1'b1;
    end else if (!resp_thru && inj_resp.valid && out_resp_ready) begin
      out_resp = inj_resp; out_resp.valid = 1'b1; inj_resp_take = 1'b1;
    end
  end

  // Outgoing VC0 (req): forward beats inject, unless injection is starving.
  logic fwd_req, inj_req_take;
  always_comb begin
    out_req = '0; fwd_req = 1'b0; inj_req_take = 1'b0;
    if (req_thru && out_req_ready) begin
      out_req = br_req; out_req.valid = 1'b1; fwd_req = 1'b1;
    end else if (!req_thru && inj_req.valid && out_req_ready) begin
      out_req = inj_req; out_req.valid = 1'b1; inj_req_take = 1'b1;
    end
  end

  assign inj_resp_ready = inj_resp_take;
  assign inj_req_ready  = inj_req_take;
  assign in_resp_ready  = !br_resp.valid;         // registered -> loop-free
  assign in_req_ready   = !br_req.valid;

  wire resp_drained = ej_resp | fwd_resp;
  wire req_drained  = ej_req  | fwd_req;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      br_req  <= '0;
      br_resp <= '0;
    end else begin
      if (!br_resp.valid)       br_resp <= in_resp;     // accept (may be null)
      else if (resp_drained)    br_resp <= '0;          // free
      if (!br_req.valid)        br_req  <= in_req;
      else if (req_drained)     br_req  <= '0;
    end
  end
endmodule
