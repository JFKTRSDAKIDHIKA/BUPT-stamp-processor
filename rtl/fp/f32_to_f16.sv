// f32_to_f16.sv — IEEE-754 binary32 -> binary16, round-to-nearest-even.
// Bit-exact mirror of src/common/f16.h :: f32_to_f16_bits.
`include "common/mobol_pkg.sv"

module f32_to_f16 (
    input  logic [31:0] f,
    output logic [15:0] h
);
  logic        sign;
  logic [7:0]  exp32;
  logic [22:0] man23;

  assign sign  = f[31];
  assign exp32 = f[30:23];
  assign man23 = f[22:0];

  always_comb begin
    logic signed [9:0] new_exp;      // exp32 - 127 + 15, can be negative
    logic [10:0] rman;               // 11 bits to catch mantissa overflow

    h = 16'd0;

    if (exp32 == 8'hFF) begin
      // Inf / NaN
      h = man23 != 0 ? {sign, 15'h7E00} : {sign, 15'h7C00};
    end else begin
      new_exp = $signed({2'b0, exp32}) - 10'sd127 + 10'sd15;

      if (new_exp >= 10'sd31) begin
        h = {sign, 15'h7C00};                    // overflow -> Inf
      end else if (new_exp >= 10'sd1) begin
        // Normal path.
        logic round_bit, sticky;
        rman    = {1'b0, man23[22:13]};          // top 10 bits
        round_bit = man23[12];
        sticky    = |man23[11:0];
        if (round_bit && (sticky || rman[0])) rman = rman + 11'd1;
        if (rman >= 11'h400) begin               // mantissa carry -> bump exp
          rman    = 11'd0;
          new_exp = new_exp + 10'sd1;
          if (new_exp >= 10'sd31) h = {sign, 15'h7C00};
          else h = {sign, new_exp[4:0], rman[9:0]};
        end else begin
          h = {sign, new_exp[4:0], rman[9:0]};
        end
      end else begin
        // Subnormal / underflow (new_exp <= 0).
        logic signed [9:0] shift;
        int unsigned total_shift;
        logic [23:0] full_man;
        logic [63:0] work;
        logic [10:0] sub_man;
        logic round_bit, sticky;

        shift = 10'sd1 - new_exp;                // >= 1
        if (shift >= 10'sd25) begin
          h = {sign, 15'd0};                     // too small -> +/- 0
        end else begin
          full_man    = {1'b1, man23};           // implicit leading 1
          total_shift = 13 + (shift - 1);        // in [13, 36]
          work        = {40'd0, full_man};
          sub_man     = 11'(work >> total_shift);
          round_bit   = (work >> (total_shift - 1)) & 64'd1;
          sticky      = (total_shift >= 2) ?
                        (|(work & ((64'd1 << (total_shift - 1)) - 64'd1))) : 1'b0;
          rman = sub_man;
          if (round_bit && (sticky || rman[0])) rman = rman + 11'd1;
          if (rman >= 11'h400) h = {sign, 5'd1, 10'd0};   // -> smallest normal
          else h = {sign, 5'd0, rman[9:0]};               // subnormal, exp=0
        end
      end
    end
  end
endmodule
