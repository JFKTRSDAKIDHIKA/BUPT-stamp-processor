// f16_to_f32.sv — IEEE-754 binary16 -> binary32, combinational.
// Bit-exact mirror of src/common/f16.h :: f16_to_f32_bits.
`include "common/mobol_pkg.sv"

module f16_to_f32 (
    input  logic [15:0] h,
    output logic [31:0] f
);
  logic        sign;
  logic [4:0]  exp5;
  logic [9:0]  man10;

  assign sign  = h[15];
  assign exp5  = h[14:10];
  assign man10 = h[9:0];

  // Leading-one position of a nonzero 10-bit mantissa (bit index 0..9).
  function automatic int unsigned lead_pos(input logic [9:0] m);
    lead_pos = 0;
    for (int i = 0; i < 10; i++) if (m[i]) lead_pos = i;
  endfunction

  always_comb begin
    if (exp5 == 5'd0) begin
      if (man10 == 10'd0) begin
        f = {sign, 31'd0};                       // +/- zero
      end else begin
        // Subnormal: normalize. Shift left by e = 10 - msb_pos so the leading
        // 1 reaches bit10; the low 10 bits are then the f32 fraction (top of
        // the 23-bit field). exponent = 127 - 15 + 1 - e = 113 - e.
        int unsigned p, e;
        logic [19:0] shifted;    // wide enough for man10 << e (e up to 10)
        logic [9:0]  frac;
        p = lead_pos(man10);
        e = 10 - p;
        shifted = {10'd0, man10} << e;            // leading 1 now at bit10
        frac    = shifted[9:0];
        f = {sign, 8'(113 - e), frac, 13'd0};     // 1 + 8 + 10 + 13 = 32
      end
    end else if (exp5 == 5'h1F) begin
      f = {sign, 8'hFF, {man10, 13'd0}};          // Inf / NaN
    end else begin
      // Normal: rebias 15 -> 127, mantissa left-extended.
      f = {sign, (8'(exp5) - 8'd15 + 8'd127), {man10, 13'd0}};
    end
  end
endmodule
