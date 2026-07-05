// transcend_test_top.sv — expose rsqrt/recip/exp2 for the C++ tolerance TB.
`include "common/mobol_pkg.sv"

module transcend_test_top (
    input  logic [31:0] rsqrt_x, output logic [31:0] rsqrt_y,
    input  logic [31:0] recip_x, output logic [31:0] recip_y,
    input  logic [31:0] exp2_x,  output logic [31:0] exp2_y
);
  fp32_rsqrt u_rs (.x(rsqrt_x), .y(rsqrt_y));
  fp32_recip u_rc (.x(recip_x), .y(recip_y));
  fp_exp2    u_e2 (.x(exp2_x),  .y(exp2_y));
endmodule
