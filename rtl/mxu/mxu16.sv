// mxu16.sv — 16x16x16 matrix unit, output-stationary, k-serial.
//
// C = [acc? Cin : 0] + A(16x16) * B(16x16), f32 accumulate, bit-exact with
// src/cycle/blockops.h :: mxu:
//   for (i,j): s = acc? Cin[i][j] : 0;
//              for k: s = fp32_add(s, fp32_mul(A[i][k], f32(B[k][j])));
// A arrives as f32 (the f16->f32 promotion is exact and done by the caller
// for MXU_F16F16); B is f16 and promoted here. One k-step/cycle => 16-cycle
// latency for the accumulate chain (ARCH_SPEC §3.2 models the pipelined
// commit separately in the performance simulator).
//
// Ports use unpacked arrays for clean per-element access; row-major (r*16+c).
`include "common/mobol_pkg.sv"

module mxu16 (
    input  logic         clk,
    input  logic         rst_n,
    input  logic         start,
    input  logic         acc,
    input  logic [31:0]  a_f32   [256],   // A promoted to f32
    input  logic [15:0]  b_f16   [256],   // B as f16
    input  logic [31:0]  cin_f32 [256],
    output logic [31:0]  cout_f32[256],
    output logic         done
);
  localparam int N = 16;

  logic [31:0] a_reg   [256];
  logic [15:0] b_reg   [256];
  logic [4:0]  k;
  logic        running;
  logic [31:0] s [256];

  // Current-k B row promoted to f32 (16 shared converters, indexed by k).
  logic [31:0] bcol_f32 [N];
  genvar gj;
  generate
    for (gj = 0; gj < N; gj++) begin : g_bconv
      f16_to_f32 u_bconv (.h(b_reg[k*N + gj]), .f(bcol_f32[gj]));
    end
  endgenerate

  // Per-PE MAC: prod = A[i][k]*B[k][j] ; add_out = s + prod.
  logic [31:0] prod    [256];
  logic [31:0] add_out [256];
  genvar gi2, gj2;
  generate
    for (gi2 = 0; gi2 < N; gi2++) begin : g_row
      for (gj2 = 0; gj2 < N; gj2++) begin : g_col
        fp32_mul u_mul (.a(a_reg[gi2*N + k]), .b(bcol_f32[gj2]),
                        .y(prod[gi2*N + gj2]));
        fp32_add u_add (.a(s[gi2*N + gj2]), .b(prod[gi2*N + gj2]),
                        .y(add_out[gi2*N + gj2]));
      end
    end
  endgenerate

  integer idx;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      running <= 1'b0; k <= 5'd0; done <= 1'b0;
    end else begin
      done <= 1'b0;
      if (start) begin
        for (idx = 0; idx < 256; idx++) begin
          a_reg[idx] <= a_f32[idx];
          b_reg[idx] <= b_f16[idx];
          s[idx]     <= acc ? cin_f32[idx] : 32'd0;
        end
        k <= 5'd0;
        running <= 1'b1;
      end else if (running) begin
        for (idx = 0; idx < 256; idx++) s[idx] <= add_out[idx];
        if (k == 5'd15) begin running <= 1'b0; done <= 1'b1; end
        k <= k + 5'd1;
      end
    end
  end

  genvar go;
  generate
    for (go = 0; go < 256; go++) begin : g_out
      assign cout_f32[go] = s[go];
    end
  endgenerate
endmodule
