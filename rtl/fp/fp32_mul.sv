// fp32_mul.sv — IEEE-754 binary32 multiply, round-to-nearest-even.
// Combinational. Matches C `float * float` (no FMA contraction): normal
// operands, gradual underflow to subnormal result, zero/Inf/NaN specials.
// This is the product in MXU's `s += a*b` (two-rounding, product then sum).
`include "common/mobol_pkg.sv"

module fp32_mul (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] y
);
  logic       sa, sb, sy;
  logic [7:0] ea, eb;
  logic [22:0] ma, mb;

  assign sa = a[31]; assign ea = a[30:23]; assign ma = a[22:0];
  assign sb = b[31]; assign eb = b[30:23]; assign mb = b[22:0];

  always_comb begin
    logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan, a_sub, b_sub;
    logic [23:0] fa, fb;
    logic signed [11:0] exp_a, exp_b, exp_sum, out_exp;
    logic [47:0] prod;
    logic [23:0] man24;              // {hidden 1, 23 mantissa}
    logic g, r, s;
    logic [24:0] mrnd;

    sy     = sa ^ sb;
    a_zero = (ea == 0) && (ma == 0);   b_zero = (eb == 0) && (mb == 0);
    a_inf  = (ea == 8'hFF) && (ma==0);  b_inf  = (eb == 8'hFF) && (mb==0);
    a_nan  = (ea == 8'hFF) && (ma!=0);  b_nan  = (eb == 8'hFF) && (mb!=0);
    a_sub  = (ea == 0) && (ma != 0);    b_sub  = (eb == 0) && (mb != 0);

    if (a_nan || b_nan || (a_inf && b_zero) || (b_inf && a_zero)) begin
      y = {1'b0, 8'hFF, 23'h400000};                         // qNaN
    end else if (a_inf || b_inf) begin
      y = {sy, 8'hFF, 23'd0};                                // Inf
    end else if (a_zero || b_zero) begin
      y = {sy, 31'd0};                                       // signed zero
    end else begin
      fa = a_sub ? {1'b0, ma} : {1'b1, ma};
      fb = b_sub ? {1'b0, mb} : {1'b1, mb};
      exp_a = a_sub ? -12'sd126 : ($signed({4'b0, ea}) - 12'sd127);
      exp_b = b_sub ? -12'sd126 : ($signed({4'b0, eb}) - 12'sd127);
      exp_sum = exp_a + exp_b;
      prod = fa * fb;                                         // 48-bit

      // Normalize: leading 1 at bit47 (>=2) or bit46 (>=1).
      if (prod[47]) begin
        man24   = prod[47:24];
        g = prod[23]; r = prod[22]; s = |prod[21:0];
        out_exp = exp_sum + 12'sd1 + 12'sd127;                // rebias
      end else begin
        man24   = prod[46:23];
        g = prod[22]; r = prod[21]; s = |prod[20:0];
        out_exp = exp_sum + 12'sd127;
      end

      // Round-to-nearest-even on the 24-bit significand.
      mrnd = {1'b0, man24};
      if (g && (r || s || man24[0])) mrnd = mrnd + 25'd1;
      if (mrnd[24]) begin                                     // mantissa carry
        mrnd    = mrnd >> 1;
        out_exp = out_exp + 12'sd1;
      end

      if (out_exp >= 12'sd255) begin
        y = {sy, 8'hFF, 23'd0};                               // overflow -> Inf
      end else if (out_exp <= 12'sd0) begin
        // Gradual underflow to subnormal (shift right by 1-out_exp).
        logic [63:0] w;
        int unsigned sh;
        logic gg, ss;
        logic [24:0] sm;
        sh = 1 - out_exp;
        if (sh >= 25) begin
          y = {sy, 31'd0};                                    // -> 0
        end else begin
          w  = {39'd0, mrnd};
          sm = 25'(w >> sh);
          gg = (sh >= 1) ? logic'((w >> (sh-1)) & 64'd1) : 1'b0;
          ss = (sh >= 2) ? (|(w & ((64'd1 << (sh-1)) - 64'd1))) : 1'b0;
          if (gg && (ss || sm[0])) sm = sm + 25'd1;
          if (sm[23]) y = {sy, 8'd1, sm[22:0]};               // rounded to normal
          else        y = {sy, 8'd0, sm[22:0]};               // subnormal
        end
      end else begin
        y = {sy, out_exp[7:0], mrnd[22:0]};                   // normal
      end
    end
  end
endmodule
