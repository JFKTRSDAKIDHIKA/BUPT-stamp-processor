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
module mxu16 (
	clk,
	rst_n,
	start,
	acc,
	a_f32,
	b_f16,
	cin_f32,
	cout_f32,
	done
);
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire acc;
	input wire [8191:0] a_f32;
	input wire [4095:0] b_f16;
	input wire [8191:0] cin_f32;
	output wire [8191:0] cout_f32;
	output reg done;
	localparam signed [31:0] N = 16;
	reg [31:0] a_reg [0:255];
	reg [15:0] b_reg [0:255];
	reg [4:0] k;
	reg running;
	reg [31:0] s [0:255];
	wire [31:0] bcol_f32 [0:15];
	genvar _gv_gj_1;
	generate
		for (_gv_gj_1 = 0; _gv_gj_1 < N; _gv_gj_1 = _gv_gj_1 + 1) begin : g_bconv
			localparam gj = _gv_gj_1;
			f16_to_f32 u_bconv(
				.h(b_reg[(k * N) + gj]),
				.f(bcol_f32[gj])
			);
		end
	endgenerate
	wire [31:0] prod [0:255];
	wire [31:0] add_out [0:255];
	genvar _gv_gi2_1;
	genvar _gv_gj2_1;
	generate
		for (_gv_gi2_1 = 0; _gv_gi2_1 < N; _gv_gi2_1 = _gv_gi2_1 + 1) begin : g_row
			localparam gi2 = _gv_gi2_1;
			for (_gv_gj2_1 = 0; _gv_gj2_1 < N; _gv_gj2_1 = _gv_gj2_1 + 1) begin : g_col
				localparam gj2 = _gv_gj2_1;
				fp32_mul u_mul(
					.a(a_reg[(gi2 * N) + k]),
					.b(bcol_f32[gj2]),
					.y(prod[(gi2 * N) + gj2])
				);
				fp32_add u_add(
					.a(s[(gi2 * N) + gj2]),
					.b(prod[(gi2 * N) + gj2]),
					.y(add_out[(gi2 * N) + gj2])
				);
			end
		end
	endgenerate
	integer idx;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			running <= 1'b0;
			k <= 5'd0;
			done <= 1'b0;
		end
		else begin
			done <= 1'b0;
			if (start) begin
				for (idx = 0; idx < 256; idx = idx + 1)
					begin
						a_reg[idx] <= a_f32[(255 - idx) * 32+:32];
						b_reg[idx] <= b_f16[(255 - idx) * 16+:16];
						s[idx] <= (acc ? cin_f32[(255 - idx) * 32+:32] : 32'd0);
					end
				k <= 5'd0;
				running <= 1'b1;
			end
			else if (running) begin
				for (idx = 0; idx < 256; idx = idx + 1)
					s[idx] <= add_out[idx];
				if (k == 5'd15) begin
					running <= 1'b0;
					done <= 1'b1;
				end
				k <= k + 5'd1;
			end
		end
	genvar _gv_go_1;
	generate
		for (_gv_go_1 = 0; _gv_go_1 < 256; _gv_go_1 = _gv_go_1 + 1) begin : g_out
			localparam go = _gv_go_1;
			assign cout_f32[(255 - go) * 32+:32] = s[go];
		end
	endgenerate
endmodule
module join_ctr (
	clk,
	rst_n,
	rel_valid,
	rel_tag,
	acq_valid,
	acq_tag,
	acq_arity,
	acq_pass,
	dbg_cnt
);
	reg _sv2v_0;
	parameter signed [31:0] NTAGS = 64;
	parameter signed [31:0] CNT_W = 12;
	parameter signed [31:0] TAG_W = $clog2(NTAGS);
	input wire clk;
	input wire rst_n;
	input wire rel_valid;
	input wire [TAG_W - 1:0] rel_tag;
	input wire acq_valid;
	input wire [TAG_W - 1:0] acq_tag;
	input wire [CNT_W - 1:0] acq_arity;
	output reg acq_pass;
	output wire [(NTAGS * CNT_W) - 1:0] dbg_cnt;
	reg [CNT_W - 1:0] cnt [0:NTAGS - 1];
	reg [CNT_W - 1:0] eff_cnt;
	always @(*) begin
		if (_sv2v_0)
			;
		eff_cnt = cnt[acq_tag];
		if (rel_valid && (rel_tag == acq_tag))
			eff_cnt = eff_cnt + 1'b1;
		acq_pass = acq_valid && (eff_cnt >= acq_arity);
	end
	integer t;
	always @(posedge clk or negedge rst_n)
		if (!rst_n)
			for (t = 0; t < NTAGS; t = t + 1)
				cnt[t] <= 1'sb0;
		else begin
			if (rel_valid)
				cnt[rel_tag] <= cnt[rel_tag] + 1'b1;
			if (acq_pass) begin
				if (rel_valid && (rel_tag == acq_tag))
					cnt[acq_tag] <= (cnt[acq_tag] + 1'b1) - acq_arity;
				else
					cnt[acq_tag] <= cnt[acq_tag] - acq_arity;
			end
		end
	genvar _gv_g_1;
	generate
		for (_gv_g_1 = 0; _gv_g_1 < NTAGS; _gv_g_1 = _gv_g_1 + 1) begin : genblk1
			localparam g = _gv_g_1;
			assign dbg_cnt[((NTAGS - 1) - g) * CNT_W+:CNT_W] = cnt[g];
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule
module sram_2r1w (
	clk,
	we,
	waddr,
	wdata,
	raddr0,
	rdata0,
	raddr1,
	rdata1
);
	parameter signed [31:0] DATA_W = 512;
	parameter signed [31:0] DEPTH = 4096;
	parameter signed [31:0] ADDR_W = $clog2(DEPTH);
	input wire clk;
	input wire we;
	input wire [ADDR_W - 1:0] waddr;
	input wire [DATA_W - 1:0] wdata;
	input wire [ADDR_W - 1:0] raddr0;
	output wire [DATA_W - 1:0] rdata0;
	input wire [ADDR_W - 1:0] raddr1;
	output wire [DATA_W - 1:0] rdata1;
	generate
		if ((DATA_W != 512) || (DEPTH != 4096)) begin : g_param_check
			unsupported_sram_2r1w_configuration u_error();
		end
	endgenerate
	mobol_sram_4096x512_2r1w u_macro(
		.clk(clk),
		.ce_in(1'b1),
		.we_in(we),
		.waddr_in(waddr),
		.wd_in(wdata),
		.raddr0_in(raddr0),
		.rd0_out(rdata0),
		.raddr1_in(raddr1),
		.rd1_out(rdata1)
	);
endmodule
module tile_top (
	clk,
	rst_n,
	start,
	a_line0,
	b_line0,
	c_line0,
	acc,
	busy,
	done,
	ext_we,
	ext_waddr,
	ext_wdata,
	ext_raddr,
	ext_rdata,
	rel_valid,
	rel_tag,
	acq_valid,
	acq_tag,
	acq_arity,
	acq_pass
);
	reg _sv2v_0;
	parameter signed [31:0] DEPTH = 4096;
	parameter signed [31:0] ADDR_W = 12;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [ADDR_W - 1:0] a_line0;
	input wire [ADDR_W - 1:0] b_line0;
	input wire [ADDR_W - 1:0] c_line0;
	input wire acc;
	output wire busy;
	output reg done;
	input wire ext_we;
	input wire [ADDR_W - 1:0] ext_waddr;
	input wire [511:0] ext_wdata;
	input wire [ADDR_W - 1:0] ext_raddr;
	output wire [511:0] ext_rdata;
	input wire rel_valid;
	input wire [5:0] rel_tag;
	input wire acq_valid;
	input wire [5:0] acq_tag;
	input wire [11:0] acq_arity;
	output wire acq_pass;
	reg sr_we;
	reg [ADDR_W - 1:0] sr_waddr;
	reg [ADDR_W - 1:0] sr_raddr0;
	reg [ADDR_W - 1:0] sr_raddr1;
	reg [511:0] sr_wdata;
	wire [511:0] sr_rdata0;
	wire [511:0] sr_rdata1;
	sram_2r1w #(
		.DATA_W(512),
		.DEPTH(DEPTH),
		.ADDR_W(ADDR_W)
	) u_sram(
		.clk(clk),
		.we(sr_we),
		.waddr(sr_waddr),
		.wdata(sr_wdata),
		.raddr0(sr_raddr0),
		.rdata0(sr_rdata0),
		.raddr1(sr_raddr1),
		.rdata1(sr_rdata1)
	);
	reg mxu_start;
	wire mxu_done;
	reg mxu_acc;
	reg [8191:0] mxu_a;
	reg [4095:0] mxu_b;
	reg [8191:0] mxu_cin;
	wire [8191:0] mxu_cout;
	mxu16 u_mxu(
		.clk(clk),
		.rst_n(rst_n),
		.start(mxu_start),
		.acc(mxu_acc),
		.a_f32(mxu_a),
		.b_f16(mxu_b),
		.cin_f32(mxu_cin),
		.cout_f32(mxu_cout),
		.done(mxu_done)
	);
	wire [767:0] dbg_cnt;
	join_ctr #(
		.NTAGS(64),
		.CNT_W(12)
	) u_join(
		.clk(clk),
		.rst_n(rst_n),
		.rel_valid(rel_valid),
		.rel_tag(rel_tag),
		.acq_valid(acq_valid),
		.acq_tag(acq_tag),
		.acq_arity(acq_arity),
		.acq_pass(acq_pass),
		.dbg_cnt(dbg_cnt)
	);
	reg [2:0] st;
	reg [4:0] li;
	reg [31:0] a_reg [0:255];
	reg [15:0] b_reg [0:255];
	reg [31:0] c_reg [0:255];
	assign busy = st != 3'd0;
	always @(*) begin
		if (_sv2v_0)
			;
		sr_we = ext_we;
		sr_waddr = ext_waddr;
		sr_wdata = ext_wdata;
		sr_raddr0 = 1'sb0;
		sr_raddr1 = ext_raddr;
		if (busy) begin
			sr_we = 1'b0;
			case (st)
				3'd1: sr_raddr0 = a_line0 + li;
				3'd2: sr_raddr0 = b_line0 + li;
				3'd5: begin
					sr_we = 1'b1;
					sr_waddr = c_line0 + li;
					begin : sv2v_autoblock_1
						reg signed [31:0] w;
						for (w = 0; w < 16; w = w + 1)
							sr_wdata[w * 32+:32] = c_reg[(li * 16) + w];
					end
				end
				default:
					;
			endcase
		end
	end
	assign ext_rdata = sr_rdata1;
	integer w;
	integer idx;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			st <= 3'd0;
			done <= 1'b0;
			li <= 0;
			mxu_start <= 1'b0;
		end
		else begin
			done <= 1'b0;
			mxu_start <= 1'b0;
			case (st)
				3'd0:
					if (start) begin
						li <= 0;
						st <= 3'd1;
					end
				3'd1: begin
					if (li != 0)
						for (w = 0; w < 16; w = w + 1)
							a_reg[((li - 1) * 16) + w] <= sr_rdata0[w * 32+:32];
					if (li == 16) begin
						for (w = 0; w < 16; w = w + 1)
							a_reg[240 + w] <= sr_rdata0[w * 32+:32];
						li <= 0;
						st <= 3'd2;
					end
					else
						li <= li + 5'd1;
				end
				3'd2: begin
					if (li != 0)
						for (w = 0; w < 32; w = w + 1)
							b_reg[((li - 1) * 32) + w] <= sr_rdata0[w * 16+:16];
					if (li == 8) begin
						for (w = 0; w < 32; w = w + 1)
							b_reg[224 + w] <= sr_rdata0[w * 16+:16];
						st <= 3'd3;
					end
					else
						li <= li + 5'd1;
				end
				3'd3: begin
					for (idx = 0; idx < 256; idx = idx + 1)
						begin
							mxu_a[(255 - idx) * 32+:32] <= a_reg[idx];
							mxu_b[(255 - idx) * 16+:16] <= b_reg[idx];
							mxu_cin[(255 - idx) * 32+:32] <= 32'd0;
						end
					mxu_acc <= acc;
					mxu_start <= 1'b1;
					st <= 3'd4;
				end
				3'd4:
					if (mxu_done) begin
						for (idx = 0; idx < 256; idx = idx + 1)
							c_reg[idx] <= mxu_cout[(255 - idx) * 32+:32];
						li <= 0;
						st <= 3'd5;
					end
				3'd5: begin
					if (li == 15)
						st <= 3'd6;
					li <= li + 5'd1;
				end
				3'd6: begin
					done <= 1'b1;
					st <= 3'd0;
				end
			endcase
		end
	initial _sv2v_0 = 0;
endmodule
