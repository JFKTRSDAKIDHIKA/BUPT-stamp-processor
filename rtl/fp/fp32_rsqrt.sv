// fp32_rsqrt.sv — 1/sqrt(x) for x > 0, synthesizable, combinational.
//
// Fast-inverse-sqrt seed (bit trick) + 3 Newton-Raphson iterations, all
// arithmetic done with the bit-exact fp32_mul/fp32_add cores, so the result
// is DETERMINISTIC and reproducible by an identical software model (the
// shared-approximation path of ARCH_SPEC §10). Accuracy after 3 iterations is
// within a few ULP of the true 1/sqrt — functionally correct for RMSNorm /
// LayerNorm (which need 1/sqrt(var+eps)). A pipelined version is a backend
// step; here it is one long combinational chain for functional verification.
`include "common/mobol_pkg.sv"

module fp32_rsqrt (
    input  logic [31:0] x,      // assumed > 0 (var+eps)
    output logic [31:0] y
);
  localparam logic [31:0] MAGIC = 32'h5f3759df;
  localparam logic [31:0] F1_5  = 32'h3FC00000;   // 1.5
  localparam logic [31:0] F0_5  = 32'h3F000000;   // 0.5

  // Seed.
  wire [31:0] seed = MAGIC - (x >> 1);
  // half x (0.5 * x), computed once.
  wire [31:0] halfx; fp32_mul m_hx(.a(F0_5), .b(x), .y(halfx));

  // One Newton iteration: yn = y * (1.5 - halfx*y*y).
  function automatic void unused(); endfunction  // (silence lint)

  // iter 1
  wire [31:0] y0 = seed;
  wire [31:0] yy0;   fp32_mul m0a(.a(y0), .b(y0),   .y(yy0));
  wire [31:0] hxy0;  fp32_mul m0b(.a(halfx),.b(yy0),.y(hxy0));
  wire [31:0] t0 = {~hxy0[31], hxy0[30:0]};        // -halfx*y*y
  wire [31:0] br0;   fp32_add a0(.a(F1_5), .b(t0), .y(br0));
  wire [31:0] y1;    fp32_mul m0c(.a(y0), .b(br0), .y(y1));

  // iter 2
  wire [31:0] yy1;   fp32_mul m1a(.a(y1), .b(y1),   .y(yy1));
  wire [31:0] hxy1;  fp32_mul m1b(.a(halfx),.b(yy1),.y(hxy1));
  wire [31:0] t1 = {~hxy1[31], hxy1[30:0]};
  wire [31:0] br1;   fp32_add a1(.a(F1_5), .b(t1), .y(br1));
  wire [31:0] y2;    fp32_mul m1c(.a(y1), .b(br1), .y(y2));

  // iter 3
  wire [31:0] yy2;   fp32_mul m2a(.a(y2), .b(y2),   .y(yy2));
  wire [31:0] hxy2;  fp32_mul m2b(.a(halfx),.b(yy2),.y(hxy2));
  wire [31:0] t2 = {~hxy2[31], hxy2[30:0]};
  wire [31:0] br2;   fp32_add a2(.a(F1_5), .b(t2), .y(br2));
  fp32_mul m2c(.a(y2), .b(br2), .y(y));
endmodule
