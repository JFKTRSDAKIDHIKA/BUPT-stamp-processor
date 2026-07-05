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
module nmc_engine (
	clk,
	rst_n,
	start,
	op_ln,
	count,
	ln_eps,
	blk_valid,
	blk_ready,
	blk_in,
	done,
	res_f32,
	res_f16
);
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire op_ln;
	input wire [7:0] count;
	input wire [31:0] ln_eps;
	input wire blk_valid;
	output wire blk_ready;
	input wire [8191:0] blk_in;
	output reg done;
	output wire [8191:0] res_f32;
	output wire [4095:0] res_f16;
	reg [1:0] st;
	reg [7:0] cnt;
	reg [31:0] acc [0:255];
	wire [31:0] sum [0:255];
	genvar _gv_i_1;
	generate
		for (_gv_i_1 = 0; _gv_i_1 < 256; _gv_i_1 = _gv_i_1 + 1) begin : g_add
			localparam i = _gv_i_1;
			fp32_add u_add(
				.a(acc[i]),
				.b(blk_in[(255 - i) * 32+:32]),
				.y(sum[i])
			);
		end
	endgenerate
	assign blk_ready = st == 2'd1;
	integer j;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			st <= 2'd0;
			done <= 1'b0;
			cnt <= 8'd0;
		end
		else begin
			done <= 1'b0;
			case (st)
				2'd0:
					if (start) begin
						cnt <= 8'd0;
						st <= 2'd1;
					end
				2'd1:
					if (blk_valid) begin
						if (cnt == 8'd0)
							for (j = 0; j < 256; j = j + 1)
								acc[j] <= blk_in[(255 - j) * 32+:32];
						else
							for (j = 0; j < 256; j = j + 1)
								acc[j] <= sum[j];
						if ((cnt + 8'd1) == count)
							st <= 2'd2;
						cnt <= cnt + 8'd1;
					end
				2'd2: begin
					done <= 1'b1;
					st <= 2'd0;
				end
			endcase
		end
	genvar _gv_go_1;
	generate
		for (_gv_go_1 = 0; _gv_go_1 < 256; _gv_go_1 = _gv_go_1 + 1) begin : g_out
			localparam go = _gv_go_1;
			assign res_f32[(255 - go) * 32+:32] = acc[go];
			assign res_f16[(255 - go) * 16+:16] = 16'd0;
		end
	endgenerate
endmodule
