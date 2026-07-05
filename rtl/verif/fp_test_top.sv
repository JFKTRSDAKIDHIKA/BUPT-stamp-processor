// fp_test_top.sv — combinational wrapper exposing the four FP leaf ops so a
// C++ testbench can drive random vectors and compare bit-exactly.
`include "common/mobol_pkg.sv"

module fp_test_top (
    input  logic [15:0] h_in,       // f16 -> f32
    output logic [31:0] f_out,
    input  logic [31:0] f_in,       // f32 -> f16
    output logic [15:0] h_out,
    input  logic [31:0] mul_a,
    input  logic [31:0] mul_b,
    output logic [31:0] mul_y,
    input  logic [31:0] add_a,
    input  logic [31:0] add_b,
    output logic [31:0] add_y
);
  f16_to_f32 u_f2f (.h(h_in), .f(f_out));
  f32_to_f16 u_f2h (.f(f_in), .h(h_out));
  fp32_mul   u_mul (.a(mul_a), .b(mul_b), .y(mul_y));
  fp32_add   u_add (.a(add_a), .b(add_b), .y(add_y));
endmodule
