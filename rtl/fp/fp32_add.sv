// fp32_add.sv — IEEE-754 binary32 add, round-to-nearest-even.
// Combinational. Matches C `float + float` on normal/zero/Inf/NaN operands.
//
// Method: exact wide fixed-point. Both significands are aligned into a 96-bit
// field (larger operand's significand LSB at bit 57), so the add/subtract is
// exact for any alignment up to 57 (far beyond the meaningful range); a
// far-sticky covers the astronomically-rare diff>57 case. The 24-bit result
// significand, guard and sticky are then extracted from the MSB of the exact
// result and rounded RNE. This avoids the subtle borrow-into-guard errors of
// an align-then-track-GRS scheme.
`include "common/mobol_pkg.sv"

module fp32_add (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] y
);
  localparam int P = 57;   // fL significand LSB position in the exact field

  function automatic int unsigned msb96(input logic [95:0] v);
    msb96 = 0;
    for (int i = 0; i < 96; i++) if (v[i]) msb96 = i;
  endfunction

  always_comb begin
    logic sa, sb;
    logic [7:0] ea, eb;
    logic [22:0] ma, mb;
    logic a_zero, b_zero, a_inf, b_inf, a_nan, b_nan;
    logic [23:0] fa, fb;
    logic signed [9:0] ea_e, eb_e;
    logic sL, sS, ry, same_sign;
    logic [23:0] fL, fS;
    logic signed [9:0] eL_e, eS_e;
    logic [8:0] eL_bias;              // fL biased exponent field value
    int unsigned diff;
    logic [95:0] AL, AS, res;
    logic far_sticky;
    int unsigned M;
    logic [24:0] sig;                // 24-bit significand + rounding headroom
    logic guard, sticky;
    logic [24:0] mrnd;
    logic signed [11:0] fexp;
    logic [63:0] w; int unsigned sh; logic gg, ss3; logic [24:0] sm;

    sa=a[31]; ea=a[30:23]; ma=a[22:0];
    sb=b[31]; eb=b[30:23]; mb=b[22:0];
    a_zero=(ea==0)&&(ma==0);   b_zero=(eb==0)&&(mb==0);
    a_inf=(ea==8'hFF)&&(ma==0); b_inf=(eb==8'hFF)&&(mb==0);
    a_nan=(ea==8'hFF)&&(ma!=0); b_nan=(eb==8'hFF)&&(mb!=0);
    y = 32'd0;

    if (a_nan || b_nan || (a_inf && b_inf && (sa!=sb)))
      y = {1'b0, 8'hFF, 23'h400000};
    else if (a_inf) y = {sa, 8'hFF, 23'd0};
    else if (b_inf) y = {sb, 8'hFF, 23'd0};
    else if (a_zero && b_zero) y = {(sa & sb), 31'd0};
    else if (a_zero) y = b;
    else if (b_zero) y = a;
    else begin
      fa = (ea==0)?{1'b0,ma}:{1'b1,ma};
      fb = (eb==0)?{1'b0,mb}:{1'b1,mb};
      // Effective (biased) exponent for magnitude ordering; subnormal -> 1.
      ea_e = (ea==0)?10'sd1:$signed({2'b0,ea});
      eb_e = (eb==0)?10'sd1:$signed({2'b0,eb});

      if (ea_e>eb_e || (ea_e==eb_e && fa>=fb)) begin
        sL=sa; fL=fa; eL_e=ea_e; sS=sb; fS=fb; eS_e=eb_e;
      end else begin
        sL=sb; fL=fb; eL_e=eb_e; sS=sa; fS=fa; eS_e=ea_e;
      end
      eL_bias = eL_e[8:0];
      same_sign = (sL==sS);
      ry = sL;

      diff = eL_e - eS_e;
      AL = {39'd0, fL, {P{1'b0}}};              // fL at [P+23 : P]
      if (diff <= P) begin
        AS = ({39'd0, fS, {P{1'b0}}}) >> diff;
        far_sticky = 1'b0;
      end else begin
        AS = 96'd0;
        far_sticky = |fS;
      end

      res = same_sign ? (AL + AS) : (AL - AS);

      if (res == 96'd0) begin
        y = {1'b0, 31'd0};                       // exact cancellation -> +0
      end else begin
        M = msb96(res);
        // 24-bit significand with leading 1 at the top, plus guard & sticky.
        sig    = 25'((res >> (M - 23)) & 96'h1FFFFFF);
        guard  = (M >= 24) ? res[M-24] : 1'b0;  // bit select == (res>>(M-24))&1
        sticky = far_sticky |
                 ((M >= 25) ? (|(res & ((96'd1 << (M - 24)) - 96'd1))) : 1'b0);
        // Result biased exponent: eL + (M - 80).
        fexp = $signed({3'b0, eL_bias}) + 12'sd0 + (12'sd0 + (M - 80));

        // Round-to-nearest-even.
        mrnd = {1'b0, sig[23:0]};
        if (guard && (sticky || sig[0])) mrnd = mrnd + 25'd1;
        if (mrnd[24]) begin mrnd = mrnd >> 1; fexp = fexp + 12'sd1; end

        if (fexp >= 12'sd255) y = {ry, 8'hFF, 23'd0};
        else if (fexp <= 12'sd0) begin
          sh = 1 - fexp;
          if (sh >= 25) y = {ry, 31'd0};
          else begin
            w  = {39'd0, mrnd};
            sm = 25'(w >> sh);
            gg = (sh>=1) ? w[sh-1] : 1'b0;  // bit select == (w>>(sh-1))&1
            ss3= (sh>=2)?(|(w&((64'd1<<(sh-1))-64'd1))):1'b0;
            if (gg && (ss3 || sm[0])) sm = sm + 25'd1;
            if (sm[23]) y = {ry, 8'd1, sm[22:0]};
            else        y = {ry, 8'd0, sm[22:0]};
          end
        end else y = {ry, fexp[7:0], mrnd[22:0]};
      end
    end
  end
endmodule
