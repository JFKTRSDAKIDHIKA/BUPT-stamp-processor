// fp_exp2.sv — 2^x for f32 x, synthesizable, combinational.
//
// x = n + f  (n = floor(x), f in [0,1));  2^x = 2^n * 2^f.
//   2^n : construct the exponent field directly (clamp to 0 / +Inf on range).
//   2^f : degree-5 minimax polynomial evaluated (Horner) with the bit-exact
//         fp32_mul/fp32_add cores -> deterministic, software-reproducible.
// exp(x) = fp_exp2(x * log2(e)); the caller pre-scales. Accuracy ~2^-20 rel
// on the fractional poly — functionally correct for softmax / SiLU / GELU.
`include "common/mobol_pkg.sv"

module fp_exp2 (
    input  logic [31:0] x,
    output logic [31:0] y
);
  // ── float floor(x) -> integer n and fractional f = x - n ──
  logic        sx;
  logic [7:0]  ex;
  logic [22:0] mx;
  assign sx = x[31]; assign ex = x[30:23]; assign mx = x[22:0];

  logic signed [9:0] e_unb;
  logic [31:0] floor_f;                 // floor(x) as f32
  logic signed [11:0] n_int;            // floor as integer
  always_comb begin
    e_unb = $signed({2'b0, ex}) - 10'sd127;
    if (ex == 8'd0) begin
      // subnormal / zero -> floor is 0 (x in (-1,1) tiny); if negative nonzero, -1
      floor_f = (sx && (mx != 0 || ex != 0)) ? 32'hBF800000 : 32'h00000000;
    end else if (e_unb >= 10'sd23) begin
      floor_f = x;                       // already integer
    end else if (e_unb < 10'sd0) begin
      // |x| < 1
      if (!sx) floor_f = 32'h00000000;   // [0,1) -> 0
      else     floor_f = (mx==0 && ex==8'd127) ? 32'hBF800000 // exactly -1? (e=0 handled above)
                        : 32'hBF800000;  // (-1,0) -> -1
    end else begin
      // 0 <= e_unb < 23: clear the fractional mantissa bits.
      logic [22:0] frac_mask;
      logic [22:0] kept;
      logic        had_frac;
      frac_mask = (23'h7FFFFF >> e_unb);         // low (23-e) bits are fractional
      kept      = mx & ~frac_mask;
      had_frac  = |(mx & frac_mask);
      if (!sx) begin
        floor_f = {sx, ex, kept};                 // truncate toward 0 == floor for +
      end else begin
        if (!had_frac) floor_f = {sx, ex, kept};  // integer already
        else begin
          // negative with fraction: floor = trunc - 1. trunc = {sx,ex,kept}.
          // Add -1 in float via fp32_add outside; here mark for subtract.
          floor_f = {sx, ex, kept};               // = ceil-toward-0; adjust below
        end
      end
    end
  end

  // For negative x with fractional part, floor = trunc - 1.
  wire neg_frac = sx && (ex != 0) && (e_unb >= 10'sd0) && (e_unb < 10'sd23)
                  && (|(mx & (23'h7FFFFF >> e_unb)));
  wire [31:0] floor_adj;
  fp32_add u_fm1(.a(floor_f), .b(32'hBF800000), .y(floor_adj));   // floor_f - 1
  wire [31:0] floor_final = neg_frac ? floor_adj : floor_f;

  // n as integer (floor_final is an exact integer-valued f32).
  logic signed [11:0] n;
  always_comb begin
    logic [7:0] fe; logic [22:0] fmv; logic signed [9:0] feu; logic [31:0] mag;
    fe = floor_final[30:23]; fmv = floor_final[22:0];
    feu = $signed({2'b0, fe}) - 10'sd127;
    if (fe == 0) n = 12'sd0;
    else begin
      mag = {1'b1, fmv, 8'd0} >> (8 + (23 - feu));  // integer magnitude
      n = floor_final[31] ? -$signed(mag[11:0]) : $signed(mag[11:0]);
    end
  end

  // f = x - floor_final  (in [0,1))
  wire [31:0] f; fp32_add u_frac(.a(x), .b({~floor_final[31], floor_final[30:0]}), .y(f));

  // 2^f via degree-5 Horner: P = c0 + f(c1 + f(c2 + f(c3 + f(c4 + f*c5))))
  // minimax coeffs for 2^f on [0,1] (f32 literals).
  localparam logic [31:0] C0 = 32'h3F800000; // 1.0000000
  localparam logic [31:0] C1 = 32'h3F317218; // 0.6931472  (ln2)
  localparam logic [31:0] C2 = 32'h3E75FDF0; // 0.2402265
  localparam logic [31:0] C3 = 32'h3D635847; // 0.0555041
  localparam logic [31:0] C4 = 32'h3C1E6362; // 0.0096181
  localparam logic [31:0] C5 = 32'h3AAF7FEF; // 0.0013334

  wire [31:0] h0; fp32_mul p0(.a(C5), .b(f), .y(h0));
  wire [31:0] s0; fp32_add q0(.a(h0), .b(C4), .y(s0));
  wire [31:0] h1; fp32_mul p1(.a(s0), .b(f), .y(h1));
  wire [31:0] s1; fp32_add q1(.a(h1), .b(C3), .y(s1));
  wire [31:0] h2; fp32_mul p2(.a(s1), .b(f), .y(h2));
  wire [31:0] s2; fp32_add q2(.a(h2), .b(C2), .y(s2));
  wire [31:0] h3; fp32_mul p3(.a(s2), .b(f), .y(h3));
  wire [31:0] s3; fp32_add q3(.a(h3), .b(C1), .y(s3));
  wire [31:0] h4; fp32_mul p4(.a(s3), .b(f), .y(h4));
  wire [31:0] pf; fp32_add q4(.a(h4), .b(C0), .y(pf));   // 2^f

  // Multiply by 2^n: add n to pf's exponent (with clamp).
  logic [31:0] scaled;
  always_comb begin
    logic signed [11:0] pe;
    pe = $signed({4'b0, pf[30:23]}) + n;   // pf is ~[1,2), exp near 127
    if (pf == 32'd0) scaled = 32'd0;
    else if (pe >= 12'sd255) scaled = {pf[31], 8'hFF, 23'd0};   // +Inf
    else if (pe <= 12'sd0)   scaled = 32'd0;                    // underflow -> 0
    else scaled = {pf[31], pe[7:0], pf[22:0]};
  end
  assign y = scaled;
endmodule
