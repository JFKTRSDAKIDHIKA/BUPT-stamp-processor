// fp32_recip.sv — 1/x for x > 0, synthesizable, combinational.
//
// Bit-trick seed + 4 Newton iterations (y = y*(2 - x*y)) over the bit-exact
// fp32_mul/fp32_add cores; deterministic and reproducible in software. Used
// by softmax (1/sum). x is assumed positive (sums of exp are > 0).
`include "common/mobol_pkg.sv"

module fp32_recip (
    input  logic [31:0] x,
    output logic [31:0] y
);
  localparam logic [31:0] MAGIC = 32'h7EF127EA;   // 1/x seed magic
  localparam logic [31:0] F2_0  = 32'h40000000;   // 2.0

  wire [31:0] seed = MAGIC - x;

  // y_{n+1} = y_n * (2 - x*y_n)
  wire [31:0] y0 = seed;
  wire [31:0] xy0; fp32_mul m0(.a(x), .b(y0), .y(xy0));
  wire [31:0] c0 = {~xy0[31], xy0[30:0]};
  wire [31:0] b0; fp32_add a0(.a(F2_0), .b(c0), .y(b0));
  wire [31:0] y1; fp32_mul n0(.a(y0), .b(b0), .y(y1));

  wire [31:0] xy1; fp32_mul m1(.a(x), .b(y1), .y(xy1));
  wire [31:0] c1 = {~xy1[31], xy1[30:0]};
  wire [31:0] b1; fp32_add a1(.a(F2_0), .b(c1), .y(b1));
  wire [31:0] y2; fp32_mul n1(.a(y1), .b(b1), .y(y2));

  wire [31:0] xy2; fp32_mul m2(.a(x), .b(y2), .y(xy2));
  wire [31:0] c2 = {~xy2[31], xy2[30:0]};
  wire [31:0] b2; fp32_add a2(.a(F2_0), .b(c2), .y(b2));
  wire [31:0] y3; fp32_mul n2(.a(y2), .b(b2), .y(y3));

  wire [31:0] xy3; fp32_mul m3(.a(x), .b(y3), .y(xy3));
  wire [31:0] c3 = {~xy3[31], xy3[30:0]};
  wire [31:0] b3; fp32_add a3(.a(F2_0), .b(c3), .y(b3));
  fp32_mul n3(.a(y3), .b(b3), .y(y));
endmodule
