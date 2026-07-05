module f16_to_f32 (
	h,
	f
);
	reg _sv2v_0;
	input wire [15:0] h;
	output reg [31:0] f;
	wire sign;
	wire [4:0] exp5;
	wire [9:0] man10;
	assign sign = h[15];
	assign exp5 = h[14:10];
	assign man10 = h[9:0];
	function automatic [31:0] lead_pos;
		input reg [9:0] m;
		begin
			lead_pos = 0;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < 10; i = i + 1)
					if (m[i])
						lead_pos = i;
			end
		end
	endfunction
	function automatic [7:0] sv2v_cast_8;
		input reg [7:0] inp;
		sv2v_cast_8 = inp;
	endfunction
	always @(*) begin
		if (_sv2v_0)
			;
		if (exp5 == 5'd0) begin
			if (man10 == 10'd0)
				f = {sign, 31'd0};
			else begin : sv2v_autoblock_2
				reg [31:0] p;
				reg [31:0] e;
				reg [19:0] shifted;
				reg [9:0] frac;
				p = lead_pos(man10);
				e = 10 - p;
				shifted = {10'd0, man10} << e;
				frac = shifted[9:0];
				f = {sign, sv2v_cast_8(113 - e), frac, 13'd0};
			end
		end
		else if (exp5 == 5'h1f)
			f = {sign, 8'hff, man10, 13'd0};
		else
			f = {sign, sv2v_cast_8(exp5) - (8'd15 - 8'd127), man10, 13'd0};
	end
	initial _sv2v_0 = 0;
endmodule
module f32_to_f16 (
	f,
	h
);
	reg _sv2v_0;
	input wire [31:0] f;
	output reg [15:0] h;
	wire sign;
	wire [7:0] exp32;
	wire [22:0] man23;
	assign sign = f[31];
	assign exp32 = f[30:23];
	assign man23 = f[22:0];
	function automatic [10:0] sv2v_cast_11;
		input reg [10:0] inp;
		sv2v_cast_11 = inp;
	endfunction
	always @(*) begin : sv2v_autoblock_1
		reg signed [9:0] new_exp;
		reg [10:0] rman;
		if (_sv2v_0)
			;
		h = 16'd0;
		if (exp32 == 8'hff)
			h = (man23 != 0 ? {sign, 15'h7e00} : {sign, 15'h7c00});
		else begin
			new_exp = $signed({2'b00, exp32}) - (10'sd127 - 10'sd15);
			if (new_exp >= 10'sd31)
				h = {sign, 15'h7c00};
			else if (new_exp >= 10'sd1) begin : sv2v_autoblock_2
				reg round_bit;
				reg sticky;
				rman = {1'b0, man23[22:13]};
				round_bit = man23[12];
				sticky = |man23[11:0];
				if (round_bit && (sticky || rman[0]))
					rman = rman + 11'd1;
				if (rman >= 11'h400) begin
					rman = 11'd0;
					new_exp = new_exp + 10'sd1;
					if (new_exp >= 10'sd31)
						h = {sign, 15'h7c00};
					else
						h = {sign, new_exp[4:0], rman[9:0]};
				end
				else
					h = {sign, new_exp[4:0], rman[9:0]};
			end
			else begin : sv2v_autoblock_3
				reg signed [9:0] shift;
				reg [31:0] total_shift;
				reg [23:0] full_man;
				reg [63:0] work;
				reg [10:0] sub_man;
				reg round_bit;
				reg sticky;
				shift = 10'sd1 - new_exp;
				if (shift >= 10'sd25)
					h = {sign, 15'd0};
				else begin
					full_man = {1'b1, man23};
					total_shift = shift + 12;
					work = {40'd0, full_man};
					sub_man = sv2v_cast_11(work >> total_shift);
					round_bit = (work >> (total_shift - 1)) & 64'd1;
					sticky = (total_shift >= 2 ? |(work & ((64'd1 << (total_shift - 1)) - 64'd1)) : 1'b0);
					rman = sub_man;
					if (round_bit && (sticky || rman[0]))
						rman = rman + 11'd1;
					if (rman >= 11'h400)
						h = {sign, 15'h0400};
					else
						h = {sign, 5'd0, rman[9:0]};
				end
			end
		end
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_mul (
	a,
	b,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	input wire [31:0] b;
	output reg [31:0] y;
	wire sa;
	wire sb;
	reg sy;
	wire [7:0] ea;
	wire [7:0] eb;
	wire [22:0] ma;
	wire [22:0] mb;
	assign sa = a[31];
	assign ea = a[30:23];
	assign ma = a[22:0];
	assign sb = b[31];
	assign eb = b[30:23];
	assign mb = b[22:0];
	function automatic [24:0] sv2v_cast_25;
		input reg [24:0] inp;
		sv2v_cast_25 = inp;
	endfunction
	always @(*) begin : sv2v_autoblock_1
		reg a_zero;
		reg b_zero;
		reg a_inf;
		reg b_inf;
		reg a_nan;
		reg b_nan;
		reg a_sub;
		reg b_sub;
		reg [23:0] fa;
		reg [23:0] fb;
		reg signed [11:0] exp_a;
		reg signed [11:0] exp_b;
		reg signed [11:0] exp_sum;
		reg signed [11:0] out_exp;
		reg [47:0] prod;
		reg [23:0] man24;
		reg g;
		reg r;
		reg s;
		reg [24:0] mrnd;
		if (_sv2v_0)
			;
		sy = sa ^ sb;
		a_zero = (ea == 0) && (ma == 0);
		b_zero = (eb == 0) && (mb == 0);
		a_inf = (ea == 8'hff) && (ma == 0);
		b_inf = (eb == 8'hff) && (mb == 0);
		a_nan = (ea == 8'hff) && (ma != 0);
		b_nan = (eb == 8'hff) && (mb != 0);
		a_sub = (ea == 0) && (ma != 0);
		b_sub = (eb == 0) && (mb != 0);
		if (((a_nan || b_nan) || (a_inf && b_zero)) || (b_inf && a_zero))
			y = 32'h7fc00000;
		else if (a_inf || b_inf)
			y = {sy, 31'h7f800000};
		else if (a_zero || b_zero)
			y = {sy, 31'd0};
		else begin
			fa = (a_sub ? {1'b0, ma} : {1'b1, ma});
			fb = (b_sub ? {1'b0, mb} : {1'b1, mb});
			exp_a = (a_sub ? -12'sd126 : $signed({4'b0000, ea}) - 12'sd127);
			exp_b = (b_sub ? -12'sd126 : $signed({4'b0000, eb}) - 12'sd127);
			exp_sum = exp_a + exp_b;
			prod = fa * fb;
			if (prod[47]) begin
				man24 = prod[47:24];
				g = prod[23];
				r = prod[22];
				s = |prod[21:0];
				out_exp = exp_sum + (12'sd1 + 12'sd127);
			end
			else begin
				man24 = prod[46:23];
				g = prod[22];
				r = prod[21];
				s = |prod[20:0];
				out_exp = exp_sum + 12'sd127;
			end
			mrnd = {1'b0, man24};
			if (g && ((r || s) || man24[0]))
				mrnd = mrnd + 25'd1;
			if (mrnd[24]) begin
				mrnd = mrnd >> 1;
				out_exp = out_exp + 12'sd1;
			end
			if (out_exp >= 12'sd255)
				y = {sy, 31'h7f800000};
			else if (out_exp <= 12'sd0) begin : sv2v_autoblock_2
				reg [63:0] w;
				reg [31:0] sh;
				reg gg;
				reg ss;
				reg [24:0] sm;
				sh = 1 - out_exp;
				if (sh >= 25)
					y = {sy, 31'd0};
				else begin
					w = {39'd0, mrnd};
					sm = sv2v_cast_25(w >> sh);
					gg = (sh >= 1 ? w[sh - 1] : 1'b0);
					ss = (sh >= 2 ? |(w & ((64'd1 << (sh - 1)) - 64'd1)) : 1'b0);
					if (gg && (ss || sm[0]))
						sm = sm + 25'd1;
					if (sm[23])
						y = {sy, 8'd1, sm[22:0]};
					else
						y = {sy, 8'd0, sm[22:0]};
				end
			end
			else
				y = {sy, out_exp[7:0], mrnd[22:0]};
		end
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_add (
	a,
	b,
	y
);
	reg _sv2v_0;
	input wire [31:0] a;
	input wire [31:0] b;
	output reg [31:0] y;
	localparam signed [31:0] P = 57;
	function automatic [31:0] msb96;
		input reg [95:0] v;
		begin
			msb96 = 0;
			begin : sv2v_autoblock_1
				reg signed [31:0] i;
				for (i = 0; i < 96; i = i + 1)
					if (v[i])
						msb96 = i;
			end
		end
	endfunction
	function automatic [24:0] sv2v_cast_25;
		input reg [24:0] inp;
		sv2v_cast_25 = inp;
	endfunction
	always @(*) begin : sv2v_autoblock_2
		reg sa;
		reg sb;
		reg [7:0] ea;
		reg [7:0] eb;
		reg [22:0] ma;
		reg [22:0] mb;
		reg a_zero;
		reg b_zero;
		reg a_inf;
		reg b_inf;
		reg a_nan;
		reg b_nan;
		reg [23:0] fa;
		reg [23:0] fb;
		reg signed [9:0] ea_e;
		reg signed [9:0] eb_e;
		reg sL;
		reg sS;
		reg ry;
		reg same_sign;
		reg [23:0] fL;
		reg [23:0] fS;
		reg signed [9:0] eL_e;
		reg signed [9:0] eS_e;
		reg [8:0] eL_bias;
		reg [31:0] diff;
		reg [95:0] AL;
		reg [95:0] AS;
		reg [95:0] res;
		reg far_sticky;
		reg [31:0] M;
		reg [24:0] sig;
		reg guard;
		reg sticky;
		reg [24:0] mrnd;
		reg signed [11:0] fexp;
		reg [63:0] w;
		reg [31:0] sh;
		reg gg;
		reg ss3;
		reg [24:0] sm;
		if (_sv2v_0)
			;
		sa = a[31];
		ea = a[30:23];
		ma = a[22:0];
		sb = b[31];
		eb = b[30:23];
		mb = b[22:0];
		a_zero = (ea == 0) && (ma == 0);
		b_zero = (eb == 0) && (mb == 0);
		a_inf = (ea == 8'hff) && (ma == 0);
		b_inf = (eb == 8'hff) && (mb == 0);
		a_nan = (ea == 8'hff) && (ma != 0);
		b_nan = (eb == 8'hff) && (mb != 0);
		y = 32'd0;
		if ((a_nan || b_nan) || ((a_inf && b_inf) && (sa != sb)))
			y = 32'h7fc00000;
		else if (a_inf)
			y = {sa, 31'h7f800000};
		else if (b_inf)
			y = {sb, 31'h7f800000};
		else if (a_zero && b_zero)
			y = {sa & sb, 31'd0};
		else if (a_zero)
			y = b;
		else if (b_zero)
			y = a;
		else begin
			fa = (ea == 0 ? {1'b0, ma} : {1'b1, ma});
			fb = (eb == 0 ? {1'b0, mb} : {1'b1, mb});
			ea_e = (ea == 0 ? 10'sd1 : $signed({2'b00, ea}));
			eb_e = (eb == 0 ? 10'sd1 : $signed({2'b00, eb}));
			if ((ea_e > eb_e) || ((ea_e == eb_e) && (fa >= fb))) begin
				sL = sa;
				fL = fa;
				eL_e = ea_e;
				sS = sb;
				fS = fb;
				eS_e = eb_e;
			end
			else begin
				sL = sb;
				fL = fb;
				eL_e = eb_e;
				sS = sa;
				fS = fa;
				eS_e = ea_e;
			end
			eL_bias = eL_e[8:0];
			same_sign = sL == sS;
			ry = sL;
			diff = eL_e - eS_e;
			AL = {39'd0, fL, {P {1'b0}}};
			if (diff <= P) begin
				AS = {39'd0, fS, {P {1'b0}}} >> diff;
				far_sticky = 1'b0;
			end
			else begin
				AS = 96'd0;
				far_sticky = |fS;
			end
			res = (same_sign ? AL + AS : AL - AS);
			if (res == 96'd0)
				y = 32'h00000000;
			else begin
				M = msb96(res);
				sig = sv2v_cast_25((res >> (M - 23)) & 96'h000000000000000001ffffff);
				guard = (M >= 24 ? res[M - 24] : 1'b0);
				sticky = far_sticky | (M >= 25 ? |(res & ((96'd1 << (M - 24)) - 96'd1)) : 1'b0);
				fexp = ($signed({3'b000, eL_bias}) + 12'sd0) + (M + (12'sd0 - 80));
				mrnd = {1'b0, sig[23:0]};
				if (guard && (sticky || sig[0]))
					mrnd = mrnd + 25'd1;
				if (mrnd[24]) begin
					mrnd = mrnd >> 1;
					fexp = fexp + 12'sd1;
				end
				if (fexp >= 12'sd255)
					y = {ry, 31'h7f800000};
				else if (fexp <= 12'sd0) begin
					sh = 1 - fexp;
					if (sh >= 25)
						y = {ry, 31'd0};
					else begin
						w = {39'd0, mrnd};
						sm = sv2v_cast_25(w >> sh);
						gg = (sh >= 1 ? w[sh - 1] : 1'b0);
						ss3 = (sh >= 2 ? |(w & ((64'd1 << (sh - 1)) - 64'd1)) : 1'b0);
						if (gg && (ss3 || sm[0]))
							sm = sm + 25'd1;
						if (sm[23])
							y = {ry, 8'd1, sm[22:0]};
						else
							y = {ry, 8'd0, sm[22:0]};
					end
				end
				else
					y = {ry, fexp[7:0], mrnd[22:0]};
			end
		end
	end
	initial _sv2v_0 = 0;
endmodule
module fp32_rsqrt (
	x,
	y
);
	input wire [31:0] x;
	output wire [31:0] y;
	localparam [31:0] MAGIC = 32'h5f3759df;
	localparam [31:0] F1_5 = 32'h3fc00000;
	localparam [31:0] F0_5 = 32'h3f000000;
	wire [31:0] seed = MAGIC - (x >> 1);
	wire [31:0] halfx;
	fp32_mul m_hx(
		.a(F0_5),
		.b(x),
		.y(halfx)
	);
	task automatic unused;
		;
	endtask
	wire [31:0] y0 = seed;
	wire [31:0] yy0;
	fp32_mul m0a(
		.a(y0),
		.b(y0),
		.y(yy0)
	);
	wire [31:0] hxy0;
	fp32_mul m0b(
		.a(halfx),
		.b(yy0),
		.y(hxy0)
	);
	wire [31:0] t0 = {~hxy0[31], hxy0[30:0]};
	wire [31:0] br0;
	fp32_add a0(
		.a(F1_5),
		.b(t0),
		.y(br0)
	);
	wire [31:0] y1;
	fp32_mul m0c(
		.a(y0),
		.b(br0),
		.y(y1)
	);
	wire [31:0] yy1;
	fp32_mul m1a(
		.a(y1),
		.b(y1),
		.y(yy1)
	);
	wire [31:0] hxy1;
	fp32_mul m1b(
		.a(halfx),
		.b(yy1),
		.y(hxy1)
	);
	wire [31:0] t1 = {~hxy1[31], hxy1[30:0]};
	wire [31:0] br1;
	fp32_add a1(
		.a(F1_5),
		.b(t1),
		.y(br1)
	);
	wire [31:0] y2;
	fp32_mul m1c(
		.a(y1),
		.b(br1),
		.y(y2)
	);
	wire [31:0] yy2;
	fp32_mul m2a(
		.a(y2),
		.b(y2),
		.y(yy2)
	);
	wire [31:0] hxy2;
	fp32_mul m2b(
		.a(halfx),
		.b(yy2),
		.y(hxy2)
	);
	wire [31:0] t2 = {~hxy2[31], hxy2[30:0]};
	wire [31:0] br2;
	fp32_add a2(
		.a(F1_5),
		.b(t2),
		.y(br2)
	);
	fp32_mul m2c(
		.a(y2),
		.b(br2),
		.y(y)
	);
endmodule
module fp32_recip (
	x,
	y
);
	input wire [31:0] x;
	output wire [31:0] y;
	localparam [31:0] MAGIC = 32'h7ef127ea;
	localparam [31:0] F2_0 = 32'h40000000;
	wire [31:0] seed = MAGIC - x;
	wire [31:0] y0 = seed;
	wire [31:0] xy0;
	fp32_mul m0(
		.a(x),
		.b(y0),
		.y(xy0)
	);
	wire [31:0] c0 = {~xy0[31], xy0[30:0]};
	wire [31:0] b0;
	fp32_add a0(
		.a(F2_0),
		.b(c0),
		.y(b0)
	);
	wire [31:0] y1;
	fp32_mul n0(
		.a(y0),
		.b(b0),
		.y(y1)
	);
	wire [31:0] xy1;
	fp32_mul m1(
		.a(x),
		.b(y1),
		.y(xy1)
	);
	wire [31:0] c1 = {~xy1[31], xy1[30:0]};
	wire [31:0] b1;
	fp32_add a1(
		.a(F2_0),
		.b(c1),
		.y(b1)
	);
	wire [31:0] y2;
	fp32_mul n1(
		.a(y1),
		.b(b1),
		.y(y2)
	);
	wire [31:0] xy2;
	fp32_mul m2(
		.a(x),
		.b(y2),
		.y(xy2)
	);
	wire [31:0] c2 = {~xy2[31], xy2[30:0]};
	wire [31:0] b2;
	fp32_add a2(
		.a(F2_0),
		.b(c2),
		.y(b2)
	);
	wire [31:0] y3;
	fp32_mul n2(
		.a(y2),
		.b(b2),
		.y(y3)
	);
	wire [31:0] xy3;
	fp32_mul m3(
		.a(x),
		.b(y3),
		.y(xy3)
	);
	wire [31:0] c3 = {~xy3[31], xy3[30:0]};
	wire [31:0] b3;
	fp32_add a3(
		.a(F2_0),
		.b(c3),
		.y(b3)
	);
	fp32_mul n3(
		.a(y3),
		.b(b3),
		.y(y)
	);
endmodule
module fp_exp2 (
	x,
	y
);
	reg _sv2v_0;
	input wire [31:0] x;
	output wire [31:0] y;
	wire sx;
	wire [7:0] ex;
	wire [22:0] mx;
	assign sx = x[31];
	assign ex = x[30:23];
	assign mx = x[22:0];
	reg signed [9:0] e_unb;
	reg [31:0] floor_f;
	wire signed [11:0] n_int;
	always @(*) begin
		if (_sv2v_0)
			;
		e_unb = $signed({2'b00, ex}) - 10'sd127;
		if (ex == 8'd0)
			floor_f = (sx && ((mx != 0) || (ex != 0)) ? 32'hbf800000 : 32'h00000000);
		else if (e_unb >= 10'sd23)
			floor_f = x;
		else if (e_unb < 10'sd0) begin
			if (!sx)
				floor_f = 32'h00000000;
			else
				floor_f = ((mx == 0) && (ex == 8'd127) ? 32'hbf800000 : 32'hbf800000);
		end
		else begin : sv2v_autoblock_1
			reg [22:0] frac_mask;
			reg [22:0] kept;
			reg had_frac;
			frac_mask = 23'h7fffff >> e_unb;
			kept = mx & ~frac_mask;
			had_frac = |(mx & frac_mask);
			if (!sx)
				floor_f = {sx, ex, kept};
			else if (!had_frac)
				floor_f = {sx, ex, kept};
			else
				floor_f = {sx, ex, kept};
		end
	end
	wire neg_frac = (((sx && (ex != 0)) && (e_unb >= 10'sd0)) && (e_unb < 10'sd23)) && |(mx & (23'h7fffff >> e_unb));
	wire [31:0] floor_adj;
	fp32_add u_fm1(
		.a(floor_f),
		.b(32'hbf800000),
		.y(floor_adj)
	);
	wire [31:0] floor_final = (neg_frac ? floor_adj : floor_f);
	reg signed [11:0] n;
	always @(*) begin : sv2v_autoblock_2
		reg [7:0] fe;
		reg [22:0] fmv;
		reg signed [9:0] feu;
		reg [31:0] mag;
		if (_sv2v_0)
			;
		fe = floor_final[30:23];
		fmv = floor_final[22:0];
		feu = $signed({2'b00, fe}) - 10'sd127;
		if (fe == 0)
			n = 12'sd0;
		else begin
			mag = {1'b1, fmv, 8'd0} >> (8 + (23 - feu));
			n = (floor_final[31] ? -$signed(mag[11:0]) : $signed(mag[11:0]));
		end
	end
	wire [31:0] f;
	fp32_add u_frac(
		.a(x),
		.b({~floor_final[31], floor_final[30:0]}),
		.y(f)
	);
	localparam [31:0] C0 = 32'h3f800000;
	localparam [31:0] C1 = 32'h3f317218;
	localparam [31:0] C2 = 32'h3e75fdf0;
	localparam [31:0] C3 = 32'h3d635847;
	localparam [31:0] C4 = 32'h3c1e6362;
	localparam [31:0] C5 = 32'h3aaf7fef;
	wire [31:0] h0;
	fp32_mul p0(
		.a(C5),
		.b(f),
		.y(h0)
	);
	wire [31:0] s0;
	fp32_add q0(
		.a(h0),
		.b(C4),
		.y(s0)
	);
	wire [31:0] h1;
	fp32_mul p1(
		.a(s0),
		.b(f),
		.y(h1)
	);
	wire [31:0] s1;
	fp32_add q1(
		.a(h1),
		.b(C3),
		.y(s1)
	);
	wire [31:0] h2;
	fp32_mul p2(
		.a(s1),
		.b(f),
		.y(h2)
	);
	wire [31:0] s2;
	fp32_add q2(
		.a(h2),
		.b(C2),
		.y(s2)
	);
	wire [31:0] h3;
	fp32_mul p3(
		.a(s2),
		.b(f),
		.y(h3)
	);
	wire [31:0] s3;
	fp32_add q3(
		.a(h3),
		.b(C1),
		.y(s3)
	);
	wire [31:0] h4;
	fp32_mul p4(
		.a(s3),
		.b(f),
		.y(h4)
	);
	wire [31:0] pf;
	fp32_add q4(
		.a(h4),
		.b(C0),
		.y(pf)
	);
	reg [31:0] scaled;
	always @(*) begin : sv2v_autoblock_3
		reg signed [11:0] pe;
		if (_sv2v_0)
			;
		pe = $signed({4'b0000, pf[30:23]}) + n;
		if (pf == 32'd0)
			scaled = 32'd0;
		else if (pe >= 12'sd255)
			scaled = {pf[31], 31'h7f800000};
		else if (pe <= 12'sd0)
			scaled = 32'd0;
		else
			scaled = {pf[31], pe[7:0], pf[22:0]};
	end
	assign y = scaled;
	initial _sv2v_0 = 0;
endmodule
module vpu16 (
	clk,
	rst_n,
	start,
	op,
	scalar,
	a_in,
	b_in,
	done,
	y_out
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [2:0] op;
	input wire [31:0] scalar;
	input wire [8191:0] a_in;
	input wire [8191:0] b_in;
	output reg done;
	output wire [8191:0] y_out;
	localparam [2:0] OP_ADD = 0;
	localparam [2:0] OP_MUL = 1;
	localparam [2:0] OP_SCALE = 2;
	localparam [2:0] OP_SILU = 3;
	localparam [2:0] OP_RMS = 4;
	localparam [31:0] LOG2E = 32'h3fb8aa3b;
	localparam [31:0] ONE = 32'h3f800000;
	localparam [31:0] INV16 = 32'h3d800000;
	reg [1:0] st;
	reg [4:0] row;
	reg [31:0] res [0:255];
	reg [31:0] inv_rms [0:15];
	reg [31:0] lane_a [0:15];
	reg [31:0] lane_b [0:15];
	integer li;
	always @(*) begin
		if (_sv2v_0)
			;
		for (li = 0; li < 16; li = li + 1)
			begin
				lane_a[li] = a_in[(255 - ((row * 16) + li)) * 32+:32];
				lane_b[li] = b_in[(255 - ((row * 16) + li)) * 32+:32];
			end
	end
	reg [31:0] lane_elt [0:15];
	wire [31:0] lane_sq [0:15];
	wire [31:0] lane_rms [0:15];
	genvar _gv_l_1;
	generate
		for (_gv_l_1 = 0; _gv_l_1 < 16; _gv_l_1 = _gv_l_1 + 1) begin : g_lane
			localparam l = _gv_l_1;
			wire [31:0] add_y;
			wire [31:0] mul_y;
			wire [31:0] scale_y;
			wire [31:0] sq;
			fp32_add u_add(
				.a(lane_a[l]),
				.b(lane_b[l]),
				.y(add_y)
			);
			fp32_mul u_mul(
				.a(lane_a[l]),
				.b(lane_b[l]),
				.y(mul_y)
			);
			fp32_mul u_scl(
				.a(lane_a[l]),
				.b(scalar),
				.y(scale_y)
			);
			fp32_mul u_sq(
				.a(lane_a[l]),
				.b(lane_a[l]),
				.y(sq)
			);
			wire [31:0] negx;
			wire [31:0] e2arg;
			wire [31:0] ex;
			wire [31:0] denom;
			wire [31:0] rinv;
			wire [31:0] silu_y;
			assign negx = {~lane_a[l][31], lane_a[l][30:0]};
			fp32_mul u_sc(
				.a(negx),
				.b(LOG2E),
				.y(e2arg)
			);
			fp_exp2 u_e2(
				.x(e2arg),
				.y(ex)
			);
			fp32_add u_dn(
				.a(ONE),
				.b(ex),
				.y(denom)
			);
			fp32_recip u_rc(
				.x(denom),
				.y(rinv)
			);
			fp32_mul u_si(
				.a(lane_a[l]),
				.b(rinv),
				.y(silu_y)
			);
			fp32_mul u_rm(
				.a(lane_a[l]),
				.b(inv_rms[row]),
				.y(lane_rms[l])
			);
			assign lane_sq[l] = sq;
			always @(*) begin
				if (_sv2v_0)
					;
				(* full_case, parallel_case *)
				case (op)
					OP_ADD: lane_elt[l] = add_y;
					OP_MUL: lane_elt[l] = mul_y;
					OP_SCALE: lane_elt[l] = scale_y;
					OP_SILU: lane_elt[l] = silu_y;
					default: lane_elt[l] = add_y;
				endcase
			end
		end
	endgenerate
	wire [31:0] t [0:7];
	wire [31:0] u [0:3];
	wire [31:0] v [0:1];
	wire [31:0] rsum;
	generate
		for (_gv_l_1 = 0; _gv_l_1 < 8; _gv_l_1 = _gv_l_1 + 1) begin : genblk2
			localparam l = _gv_l_1;
			fp32_add rs(
				.a(lane_sq[2 * l]),
				.b(lane_sq[(2 * l) + 1]),
				.y(t[l])
			);
		end
		for (_gv_l_1 = 0; _gv_l_1 < 4; _gv_l_1 = _gv_l_1 + 1) begin : genblk3
			localparam l = _gv_l_1;
			fp32_add ru(
				.a(t[2 * l]),
				.b(t[(2 * l) + 1]),
				.y(u[l])
			);
		end
		for (_gv_l_1 = 0; _gv_l_1 < 2; _gv_l_1 = _gv_l_1 + 1) begin : genblk4
			localparam l = _gv_l_1;
			fp32_add rv(
				.a(u[2 * l]),
				.b(u[(2 * l) + 1]),
				.y(v[l])
			);
		end
	endgenerate
	fp32_add rw(
		.a(v[0]),
		.b(v[1]),
		.y(rsum)
	);
	wire [31:0] mean;
	wire [31:0] meaneps;
	wire [31:0] inv;
	fp32_mul mm(
		.a(rsum),
		.b(INV16),
		.y(mean)
	);
	fp32_add me(
		.a(mean),
		.b(scalar),
		.y(meaneps)
	);
	fp32_rsqrt rq(
		.x(meaneps),
		.y(inv)
	);
	reg [31:0] out_lane [0:15];
	always @(*) begin
		if (_sv2v_0)
			;
		for (li = 0; li < 16; li = li + 1)
			out_lane[li] = (op == OP_RMS ? lane_rms[li] : lane_elt[li]);
	end
	integer o;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			st <= 2'd0;
			done <= 1'b0;
			row <= 0;
		end
		else begin
			done <= 1'b0;
			case (st)
				2'd0:
					if (start) begin
						row <= 0;
						st <= (op == OP_RMS ? 2'd1 : 2'd2);
					end
				2'd1: begin
					inv_rms[row] <= inv;
					if (row == 5'd15) begin
						row <= 0;
						st <= 2'd2;
					end
					else
						row <= row + 5'd1;
				end
				2'd2: begin
					for (o = 0; o < 16; o = o + 1)
						res[(row * 16) + o] <= out_lane[o];
					if (row == 5'd15)
						st <= 2'd3;
					else
						row <= row + 5'd1;
				end
				2'd3: begin
					done <= 1'b1;
					st <= 2'd0;
				end
			endcase
		end
	genvar _gv_g_1;
	generate
		for (_gv_g_1 = 0; _gv_g_1 < 256; _gv_g_1 = _gv_g_1 + 1) begin : genblk5
			localparam g = _gv_g_1;
			assign y_out[(255 - g) * 32+:32] = res[g];
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule
