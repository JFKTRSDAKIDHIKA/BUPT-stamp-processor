// vpu16.sv — 16-lane f32 vector unit (ARCH_SPEC §3.3).
//
// Processes a 256-element (16x16) block, 16 lanes over 16 cycles. Ops:
//   ADD/MUL/SCALE : elementwise f32, bit-exact (fp32_add/fp32_mul).
//   SILU          : x*sigmoid(x) = x/(1+exp(-x)) via the synthesizable fp_exp2
//                   + fp32_recip units (transcendental approximation).
//   RMSNORM       : two-pass — pass 1 computes per-row 1/sqrt(mean(x^2)+eps)
//                   with fp32_rsqrt; pass 2 scales x by it.
// Single-driver `res` register (synthesis-clean).
//
// SYNTHESIS NOTE: the transcendental units (fp_exp2/rsqrt/recip) are deep
// combinational chains here; for Fusion Compiler timing closure pipeline them
// (register between Newton/Horner stages). The lane structure and op decode —
// what floorplanning needs — are complete.
`include "common/mobol_pkg.sv"

module vpu16 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic [2:0]   op,          // 0=ADD 1=MUL 2=SCALE 3=SILU 4=RMSNORM
    input  logic [31:0]  scalar,      // SCALE factor / RMSNORM eps
    input  logic [31:0]  a_in [256],
    input  logic [31:0]  b_in [256],
    output logic         done,
    output logic [31:0]  y_out [256]
);
  localparam logic [2:0] OP_ADD=0, OP_MUL=1, OP_SCALE=2, OP_SILU=3, OP_RMS=4;
  localparam logic [31:0] LOG2E = 32'h3FB8AA3B;
  localparam logic [31:0] ONE   = 32'h3F800000;
  localparam logic [31:0] INV16 = 32'h3D800000;   // 1/16

  typedef enum logic [1:0] {IDLE, P1, P2, FIN} st_e;
  st_e st;
  logic [4:0] row;
  logic [31:0] res     [256];
  logic [31:0] inv_rms [16];

  // Current row inputs.
  logic [31:0] lane_a [16], lane_b [16];
  integer li;
  always_comb for (li=0;li<16;li++) begin
    lane_a[li] = a_in[row*16 + li];
    lane_b[li] = b_in[row*16 + li];
  end

  // Per-lane elementwise results + square (for RMS pass1).
  logic [31:0] lane_elt [16], lane_sq [16], lane_rms [16];
  genvar l;
  generate
    for (l=0;l<16;l++) begin : g_lane
      logic [31:0] add_y, mul_y, scale_y, sq;
      fp32_add u_add (.a(lane_a[l]), .b(lane_b[l]), .y(add_y));
      fp32_mul u_mul (.a(lane_a[l]), .b(lane_b[l]), .y(mul_y));
      fp32_mul u_scl (.a(lane_a[l]), .b(scalar),    .y(scale_y));
      fp32_mul u_sq  (.a(lane_a[l]), .b(lane_a[l]), .y(sq));
      // SILU
      logic [31:0] negx, e2arg, ex, denom, rinv, silu_y;
      assign negx = {~lane_a[l][31], lane_a[l][30:0]};
      fp32_mul   u_sc(.a(negx), .b(LOG2E), .y(e2arg));
      fp_exp2    u_e2(.x(e2arg), .y(ex));
      fp32_add   u_dn(.a(ONE), .b(ex), .y(denom));
      fp32_recip u_rc(.x(denom), .y(rinv));
      fp32_mul   u_si(.a(lane_a[l]), .b(rinv), .y(silu_y));
      // RMS pass2 scale
      fp32_mul   u_rm(.a(lane_a[l]), .b(inv_rms[row]), .y(lane_rms[l]));

      assign lane_sq[l] = sq;
      always_comb begin
        unique case (op)
          OP_ADD:   lane_elt[l] = add_y;
          OP_MUL:   lane_elt[l] = mul_y;
          OP_SCALE: lane_elt[l] = scale_y;
          OP_SILU:  lane_elt[l] = silu_y;
          default:  lane_elt[l] = add_y;
        endcase
      end
    end
  endgenerate

  // Fixed-order row reduction of the 16 squares -> rsum.
  logic [31:0] t [8], u [4], v [2], rsum;
  generate
    for (l=0;l<8;l++) fp32_add rs(.a(lane_sq[2*l]), .b(lane_sq[2*l+1]), .y(t[l]));
    for (l=0;l<4;l++) fp32_add ru(.a(t[2*l]), .b(t[2*l+1]), .y(u[l]));
    for (l=0;l<2;l++) fp32_add rv(.a(u[2*l]), .b(u[2*l+1]), .y(v[l]));
  endgenerate
  fp32_add rw(.a(v[0]), .b(v[1]), .y(rsum));
  logic [31:0] mean, meaneps, inv;
  fp32_mul mm(.a(rsum), .b(INV16), .y(mean));
  fp32_add me(.a(mean), .b(scalar), .y(meaneps));
  fp32_rsqrt rq(.x(meaneps), .y(inv));

  // Output-lane mux: RMS pass2 uses the scaled value, else the elementwise.
  logic [31:0] out_lane [16];
  always_comb for (li=0;li<16;li++)
    out_lane[li] = (op==OP_RMS) ? lane_rms[li] : lane_elt[li];

  integer o;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin st<=IDLE; done<=1'b0; row<=0; end
    else begin
      done<=1'b0;
      case (st)
        IDLE: if (start) begin row<=0; st<=(op==OP_RMS)?P1:P2; end
        P1: begin                                  // RMS pass1: per-row 1/rms
          inv_rms[row] <= inv;
          if (row==5'd15) begin row<=0; st<=P2; end
          else row<=row+5'd1;
        end
        P2: begin                                  // write outputs, one row/cyc
          for (o=0;o<16;o++) res[row*16+o] <= out_lane[o];
          if (row==5'd15) st<=FIN; else row<=row+5'd1;
        end
        FIN: begin done<=1'b1; st<=IDLE; end
      endcase
    end
  end

  genvar g;
  generate for (g=0;g<256;g++) assign y_out[g]=res[g]; endgenerate
endmodule
