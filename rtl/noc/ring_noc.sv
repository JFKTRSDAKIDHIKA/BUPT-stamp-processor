// ring_noc.sv — 16-node bidirectional ring (ARCH_SPEC §5).
//
// Each node has two ring_routers (one per direction). A locally-injected flit
// enters the router for its shortest direction; ejected flits from either
// direction merge to the tile eject port. Neighbours are wired in a ring.
`include "noc/noc_pkg.sv"
import noc_pkg::*;

module ring_noc (
    input  logic clk,
    input  logic rst_n,

    // Per-tile injection (VC0 req / VC1 resp) and eject.
    input  flit_t inj_req  [NUM_TILES],
    input  flit_t inj_resp [NUM_TILES],
    output logic  inj_req_ready  [NUM_TILES],
    output logic  inj_resp_ready [NUM_TILES],
    // Per-direction eject to the tile (a packet arrives via one direction
    // only, so the two never carry the same flit). A real tile merges them
    // through a small FIFO; both are always-ready here.
    output flit_t eject_cw  [NUM_TILES],
    output flit_t eject_ccw [NUM_TILES]
);
  // Direction 0 = clockwise (node n -> n+1), 1 = counter-clockwise (n -> n-1).
  // Per node, per direction ring channels.
  flit_t cw_req  [NUM_TILES], cw_resp [NUM_TILES];   // node n output CW
  logic  cw_req_rdy [NUM_TILES], cw_resp_rdy [NUM_TILES];
  flit_t ccw_req [NUM_TILES], ccw_resp[NUM_TILES];   // node n output CCW
  logic  ccw_req_rdy[NUM_TILES], ccw_resp_rdy[NUM_TILES];

  // Per node injection split by direction, and per-direction eject.
  flit_t inj_req_cw [NUM_TILES], inj_req_ccw [NUM_TILES];
  flit_t inj_resp_cw[NUM_TILES], inj_resp_ccw[NUM_TILES];
  logic  ir_cw_rdy[NUM_TILES], ir_ccw_rdy[NUM_TILES];
  logic  irs_cw_rdy[NUM_TILES], irs_ccw_rdy[NUM_TILES];

  genvar n;
  generate
    for (n = 0; n < NUM_TILES; n++) begin : g_node
      localparam int NP1 = (n + 1) % NUM_TILES;
      localparam int NM1 = (n + NUM_TILES - 1) % NUM_TILES;

      // Route local injection to its shortest direction.
      always_comb begin
        inj_req_cw[n]  = '0; inj_req_ccw[n]  = '0;
        inj_resp_cw[n] = '0; inj_resp_ccw[n] = '0;
        if (inj_req[n].valid) begin
          if (ring_dir(n[TID_W-1:0], inj_req[n].dst) == 1'b0) inj_req_cw[n] = inj_req[n];
          else inj_req_ccw[n] = inj_req[n];
        end
        if (inj_resp[n].valid) begin
          if (ring_dir(n[TID_W-1:0], inj_resp[n].dst) == 1'b0) inj_resp_cw[n] = inj_resp[n];
          else inj_resp_ccw[n] = inj_resp[n];
        end
      end
      assign inj_req_ready[n]  = (inj_req[n].valid &&
                                  ring_dir(n[TID_W-1:0], inj_req[n].dst)==1'b0) ?
                                  ir_cw_rdy[n] : ir_ccw_rdy[n];
      assign inj_resp_ready[n] = (inj_resp[n].valid &&
                                  ring_dir(n[TID_W-1:0], inj_resp[n].dst)==1'b0) ?
                                  irs_cw_rdy[n] : irs_ccw_rdy[n];

      // CW router: input from node n-1's CW output, output to node n+1.
      ring_router #(.NODE_ID(n[TID_W-1:0])) u_cw (
        .clk, .rst_n,
        .in_req(cw_req[NM1]),   .in_resp(cw_resp[NM1]),
        .in_req_ready(cw_req_rdy[NM1]), .in_resp_ready(cw_resp_rdy[NM1]),
        .out_req(cw_req[n]),    .out_resp(cw_resp[n]),
        .out_req_ready(cw_req_rdy[n]),  .out_resp_ready(cw_resp_rdy[n]),
        .inj_req(inj_req_cw[n]), .inj_resp(inj_resp_cw[n]),
        .inj_req_ready(ir_cw_rdy[n]), .inj_resp_ready(irs_cw_rdy[n]),
        .eject(eject_cw[n]), .eject_ready(1'b1)
      );

      // CCW router: input from node n+1's CCW output, output to node n-1.
      ring_router #(.NODE_ID(n[TID_W-1:0])) u_ccw (
        .clk, .rst_n,
        .in_req(ccw_req[NP1]),  .in_resp(ccw_resp[NP1]),
        .in_req_ready(ccw_req_rdy[NP1]), .in_resp_ready(ccw_resp_rdy[NP1]),
        .out_req(ccw_req[n]),   .out_resp(ccw_resp[n]),
        .out_req_ready(ccw_req_rdy[n]),  .out_resp_ready(ccw_resp_rdy[n]),
        .inj_req(inj_req_ccw[n]), .inj_resp(inj_resp_ccw[n]),
        .inj_req_ready(ir_ccw_rdy[n]), .inj_resp_ready(irs_ccw_rdy[n]),
        .eject(eject_ccw[n]), .eject_ready(1'b1)
      );
    end
  endgenerate
endmodule
