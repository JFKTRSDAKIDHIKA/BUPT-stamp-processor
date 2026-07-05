module ssp_bank (
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
	parameter signed [31:0] DEPTH = 131072;
	parameter signed [31:0] ADDR_W = $clog2(DEPTH);
	parameter signed [31:0] MACRO_DEPTH = 4096;
	parameter signed [31:0] MACRO_AW = $clog2(MACRO_DEPTH);
	parameter signed [31:0] NSUB = DEPTH / MACRO_DEPTH;
	parameter signed [31:0] SUB_W = $clog2(NSUB);
	input wire clk;
	input wire we;
	input wire [ADDR_W - 1:0] waddr;
	input wire [DATA_W - 1:0] wdata;
	input wire [ADDR_W - 1:0] raddr0;
	output wire [DATA_W - 1:0] rdata0;
	input wire [ADDR_W - 1:0] raddr1;
	output wire [DATA_W - 1:0] rdata1;
	wire [SUB_W - 1:0] wsel = waddr[ADDR_W - 1:MACRO_AW];
	wire [SUB_W - 1:0] r0sel = raddr0[ADDR_W - 1:MACRO_AW];
	wire [SUB_W - 1:0] r1sel = raddr1[ADDR_W - 1:MACRO_AW];
	reg [SUB_W - 1:0] r0sel_q;
	reg [SUB_W - 1:0] r1sel_q;
	always @(posedge clk) begin
		r0sel_q <= r0sel;
		r1sel_q <= r1sel;
	end
	wire [DATA_W - 1:0] rd0_sub [0:NSUB - 1];
	wire [DATA_W - 1:0] rd1_sub [0:NSUB - 1];
	genvar _gv_g_1;
	function automatic signed [SUB_W - 1:0] sv2v_cast_B83CA_signed;
		input reg signed [SUB_W - 1:0] inp;
		sv2v_cast_B83CA_signed = inp;
	endfunction
	generate
		for (_gv_g_1 = 0; _gv_g_1 < NSUB; _gv_g_1 = _gv_g_1 + 1) begin : g_sub
			localparam g = _gv_g_1;
			wire w_hit = we && (wsel == sv2v_cast_B83CA_signed(g));
			wire r0_hit = r0sel == sv2v_cast_B83CA_signed(g);
			wire r1_hit = r1sel == sv2v_cast_B83CA_signed(g);
			mobol_sram_4096x512_2r1w u_macro(
				.clk(clk),
				.ce_in((w_hit | r0_hit) | r1_hit),
				.we_in(w_hit),
				.waddr_in(waddr[MACRO_AW - 1:0]),
				.wd_in(wdata),
				.raddr0_in(raddr0[MACRO_AW - 1:0]),
				.rd0_out(rd0_sub[g]),
				.raddr1_in(raddr1[MACRO_AW - 1:0]),
				.rd1_out(rd1_sub[g])
			);
		end
	endgenerate
	assign rdata0 = rd0_sub[r0sel_q];
	assign rdata1 = rd1_sub[r1sel_q];
endmodule
