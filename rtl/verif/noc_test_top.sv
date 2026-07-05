// noc_test_top.sv — scalar-port wrapper around ring_noc for the C++ TB.
`include "noc/noc_pkg.sv"
import noc_pkg::*;

module noc_test_top (
    input  logic clk,
    input  logic rst_n,
    // injection (per tile) as scalars
    input  logic [NUM_TILES-1:0]       inj_valid,
    input  logic [NUM_TILES-1:0]       inj_is_resp,
    input  logic [TID_W-1:0]           inj_dst  [NUM_TILES],
    input  logic [TID_W-1:0]           inj_src  [NUM_TILES],
    input  logic [31:0]                inj_id   [NUM_TILES],
    output logic [NUM_TILES-1:0]       inj_ready,
    // ejection (per tile) — the two ring directions merged in software; the
    // TB drains both. Expose CW in slot 0, CCW in slot 1 per tile.
    output logic [NUM_TILES-1:0]       ej_valid_cw,
    output logic [NUM_TILES-1:0]       ej_valid_ccw,
    output logic [TID_W-1:0]           ej_dst_cw  [NUM_TILES],
    output logic [31:0]                ej_id_cw   [NUM_TILES],
    output logic [TID_W-1:0]           ej_dst_ccw [NUM_TILES],
    output logic [31:0]                ej_id_ccw  [NUM_TILES]
);
  flit_t inj_req  [NUM_TILES], inj_resp [NUM_TILES];
  logic  inj_req_ready [NUM_TILES], inj_resp_ready [NUM_TILES];
  flit_t eject_cw [NUM_TILES], eject_ccw [NUM_TILES];

  genvar t;
  generate
    for (t = 0; t < NUM_TILES; t++) begin : g
      always_comb begin
        inj_req[t]  = '0;
        inj_resp[t] = '0;
        // encode id in the payload low 32 bits; kind READ_REQ/READ_RESP.
        if (inj_valid[t]) begin
          if (inj_is_resp[t]) begin
            inj_resp[t].valid   = 1'b1;
            inj_resp[t].kind    = READ_RESP;
            inj_resp[t].dst     = inj_dst[t];
            inj_resp[t].src     = inj_src[t];
            inj_resp[t].payload = {480'd0, inj_id[t]};
          end else begin
            inj_req[t].valid    = 1'b1;
            inj_req[t].kind     = READ_REQ;
            inj_req[t].dst      = inj_dst[t];
            inj_req[t].src      = inj_src[t];
            inj_req[t].payload  = {480'd0, inj_id[t]};
          end
        end
      end
      assign inj_ready[t] = inj_is_resp[t] ? inj_resp_ready[t] : inj_req_ready[t];
      assign ej_valid_cw[t]  = eject_cw[t].valid;
      assign ej_dst_cw[t]    = eject_cw[t].dst;
      assign ej_id_cw[t]     = eject_cw[t].payload[31:0];
      assign ej_valid_ccw[t] = eject_ccw[t].valid;
      assign ej_dst_ccw[t]   = eject_ccw[t].dst;
      assign ej_id_ccw[t]    = eject_ccw[t].payload[31:0];
    end
  endgenerate

  ring_noc u_noc (
    .clk, .rst_n,
    .inj_req, .inj_resp, .inj_req_ready, .inj_resp_ready,
    .eject_cw, .eject_ccw
  );
endmodule
