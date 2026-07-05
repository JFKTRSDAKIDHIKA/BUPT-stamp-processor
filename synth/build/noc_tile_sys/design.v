module ring_router (
	clk,
	rst_n,
	in_req,
	in_resp,
	in_req_ready,
	in_resp_ready,
	out_req,
	out_resp,
	out_req_ready,
	out_resp_ready,
	inj_req,
	inj_resp,
	inj_req_ready,
	inj_resp_ready,
	eject,
	eject_ready
);
	reg _sv2v_0;
	localparam signed [31:0] noc_pkg_TID_W = 4;
	parameter [3:0] NODE_ID = 1'sb0;
	input wire clk;
	input wire rst_n;
	localparam signed [31:0] noc_pkg_PAYLOAD_B = 64;
	localparam signed [31:0] noc_pkg_PAYLOAD_W = 512;
	input wire [549:0] in_req;
	input wire [549:0] in_resp;
	output wire in_req_ready;
	output wire in_resp_ready;
	output reg [549:0] out_req;
	output reg [549:0] out_resp;
	input wire out_req_ready;
	input wire out_resp_ready;
	input wire [549:0] inj_req;
	input wire [549:0] inj_resp;
	output wire inj_req_ready;
	output wire inj_resp_ready;
	output reg [549:0] eject;
	input wire eject_ready;
	reg [549:0] br_req;
	reg [549:0] br_resp;
	wire req_here = br_req[549] && (br_req[545-:4] == NODE_ID);
	wire resp_here = br_resp[549] && (br_resp[545-:4] == NODE_ID);
	wire req_thru = br_req[549] && !req_here;
	wire resp_thru = br_resp[549] && !resp_here;
	reg ej_resp;
	reg ej_req;
	always @(*) begin
		if (_sv2v_0)
			;
		eject = 1'sb0;
		ej_resp = resp_here && eject_ready;
		ej_req = (!ej_resp && req_here) && eject_ready;
		if (ej_resp)
			eject = br_resp;
		else if (ej_req)
			eject = br_req;
		eject[549] = ej_resp | ej_req;
	end
	reg fwd_resp;
	reg inj_resp_take;
	always @(*) begin
		if (_sv2v_0)
			;
		out_resp = 1'sb0;
		fwd_resp = 1'b0;
		inj_resp_take = 1'b0;
		if (resp_thru && out_resp_ready) begin
			out_resp = br_resp;
			out_resp[549] = 1'b1;
			fwd_resp = 1'b1;
		end
		else if ((!resp_thru && inj_resp[549]) && out_resp_ready) begin
			out_resp = inj_resp;
			out_resp[549] = 1'b1;
			inj_resp_take = 1'b1;
		end
	end
	reg fwd_req;
	reg inj_req_take;
	always @(*) begin
		if (_sv2v_0)
			;
		out_req = 1'sb0;
		fwd_req = 1'b0;
		inj_req_take = 1'b0;
		if (req_thru && out_req_ready) begin
			out_req = br_req;
			out_req[549] = 1'b1;
			fwd_req = 1'b1;
		end
		else if ((!req_thru && inj_req[549]) && out_req_ready) begin
			out_req = inj_req;
			out_req[549] = 1'b1;
			inj_req_take = 1'b1;
		end
	end
	assign inj_resp_ready = inj_resp_take;
	assign inj_req_ready = inj_req_take;
	assign in_resp_ready = !br_resp[549];
	assign in_req_ready = !br_req[549];
	wire resp_drained = ej_resp | fwd_resp;
	wire req_drained = ej_req | fwd_req;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			br_req <= 1'sb0;
			br_resp <= 1'sb0;
		end
		else begin
			if (!br_resp[549])
				br_resp <= in_resp;
			else if (resp_drained)
				br_resp <= 1'sb0;
			if (!br_req[549])
				br_req <= in_req;
			else if (req_drained)
				br_req <= 1'sb0;
		end
	initial _sv2v_0 = 0;
endmodule
module ring_noc (
	clk,
	rst_n,
	inj_req,
	inj_resp,
	inj_req_ready,
	inj_resp_ready,
	eject_cw,
	eject_ccw
);
	reg _sv2v_0;
	input wire clk;
	input wire rst_n;
	localparam signed [31:0] noc_pkg_NUM_TILES = 16;
	localparam signed [31:0] noc_pkg_PAYLOAD_B = 64;
	localparam signed [31:0] noc_pkg_PAYLOAD_W = 512;
	localparam signed [31:0] noc_pkg_TID_W = 4;
	input wire [8799:0] inj_req;
	input wire [8799:0] inj_resp;
	output wire [0:15] inj_req_ready;
	output wire [0:15] inj_resp_ready;
	output wire [8799:0] eject_cw;
	output wire [8799:0] eject_ccw;
	wire [549:0] cw_req [0:15];
	wire [549:0] cw_resp [0:15];
	wire cw_req_rdy [0:15];
	wire cw_resp_rdy [0:15];
	wire [549:0] ccw_req [0:15];
	wire [549:0] ccw_resp [0:15];
	wire ccw_req_rdy [0:15];
	wire ccw_resp_rdy [0:15];
	reg [549:0] inj_req_cw [0:15];
	reg [549:0] inj_req_ccw [0:15];
	reg [549:0] inj_resp_cw [0:15];
	reg [549:0] inj_resp_ccw [0:15];
	wire ir_cw_rdy [0:15];
	wire ir_ccw_rdy [0:15];
	wire irs_cw_rdy [0:15];
	wire irs_ccw_rdy [0:15];
	genvar _gv_n_1;
	function automatic signed [31:0] sv2v_cast_32_signed;
		input reg signed [31:0] inp;
		sv2v_cast_32_signed = inp;
	endfunction
	function automatic noc_pkg_ring_dir;
		input reg [3:0] from;
		input reg [3:0] to;
		reg signed [31:0] cw;
		begin
			cw = ((sv2v_cast_32_signed(to) - sv2v_cast_32_signed(from)) + noc_pkg_NUM_TILES) % noc_pkg_NUM_TILES;
			noc_pkg_ring_dir = (cw <= (noc_pkg_NUM_TILES - cw) ? 1'b0 : 1'b1);
		end
	endfunction
	generate
		for (_gv_n_1 = 0; _gv_n_1 < noc_pkg_NUM_TILES; _gv_n_1 = _gv_n_1 + 1) begin : g_node
			localparam n = _gv_n_1;
			localparam signed [31:0] NP1 = (n + 1) % noc_pkg_NUM_TILES;
			localparam signed [31:0] NM1 = ((n + noc_pkg_NUM_TILES) - 1) % noc_pkg_NUM_TILES;
			always @(*) begin
				if (_sv2v_0)
					;
				inj_req_cw[n] = 1'sb0;
				inj_req_ccw[n] = 1'sb0;
				inj_resp_cw[n] = 1'sb0;
				inj_resp_ccw[n] = 1'sb0;
				if (inj_req[((15 - n) * 550) + 549]) begin
					if (noc_pkg_ring_dir(n[3:0], inj_req[((15 - n) * 550) + 545-:4]) == 1'b0)
						inj_req_cw[n] = inj_req[(15 - n) * 550+:550];
					else
						inj_req_ccw[n] = inj_req[(15 - n) * 550+:550];
				end
				if (inj_resp[((15 - n) * 550) + 549]) begin
					if (noc_pkg_ring_dir(n[3:0], inj_resp[((15 - n) * 550) + 545-:4]) == 1'b0)
						inj_resp_cw[n] = inj_resp[(15 - n) * 550+:550];
					else
						inj_resp_ccw[n] = inj_resp[(15 - n) * 550+:550];
				end
			end
			assign inj_req_ready[n] = (inj_req[((15 - n) * 550) + 549] && (noc_pkg_ring_dir(n[3:0], inj_req[((15 - n) * 550) + 545-:4]) == 1'b0) ? ir_cw_rdy[n] : ir_ccw_rdy[n]);
			assign inj_resp_ready[n] = (inj_resp[((15 - n) * 550) + 549] && (noc_pkg_ring_dir(n[3:0], inj_resp[((15 - n) * 550) + 545-:4]) == 1'b0) ? irs_cw_rdy[n] : irs_ccw_rdy[n]);
			ring_router #(.NODE_ID(n[3:0])) u_cw(
				.clk(clk),
				.rst_n(rst_n),
				.in_req(cw_req[NM1]),
				.in_resp(cw_resp[NM1]),
				.in_req_ready(cw_req_rdy[NM1]),
				.in_resp_ready(cw_resp_rdy[NM1]),
				.out_req(cw_req[n]),
				.out_resp(cw_resp[n]),
				.out_req_ready(cw_req_rdy[n]),
				.out_resp_ready(cw_resp_rdy[n]),
				.inj_req(inj_req_cw[n]),
				.inj_resp(inj_resp_cw[n]),
				.inj_req_ready(ir_cw_rdy[n]),
				.inj_resp_ready(irs_cw_rdy[n]),
				.eject(eject_cw[(15 - n) * 550+:550]),
				.eject_ready(1'b1)
			);
			ring_router #(.NODE_ID(n[3:0])) u_ccw(
				.clk(clk),
				.rst_n(rst_n),
				.in_req(ccw_req[NP1]),
				.in_resp(ccw_resp[NP1]),
				.in_req_ready(ccw_req_rdy[NP1]),
				.in_resp_ready(ccw_resp_rdy[NP1]),
				.out_req(ccw_req[n]),
				.out_resp(ccw_resp[n]),
				.out_req_ready(ccw_req_rdy[n]),
				.out_resp_ready(ccw_resp_rdy[n]),
				.inj_req(inj_req_ccw[n]),
				.inj_resp(inj_resp_ccw[n]),
				.inj_req_ready(ir_ccw_rdy[n]),
				.inj_resp_ready(irs_ccw_rdy[n]),
				.eject(eject_ccw[(15 - n) * 550+:550]),
				.eject_ready(1'b1)
			);
		end
	endgenerate
	initial _sv2v_0 = 0;
endmodule
module dma_engine (
	clk,
	rst_n,
	start,
	src_line,
	dst_tile,
	dst_addr0,
	nlines,
	busy,
	done,
	rd_line,
	rd_data,
	inj_req,
	inj_req_ready
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_W = 12;
	localparam signed [31:0] noc_pkg_TID_W = 4;
	parameter [3:0] MY_ID = 1'sb0;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [ADDR_W - 1:0] src_line;
	input wire [3:0] dst_tile;
	input wire [17:0] dst_addr0;
	input wire [15:0] nlines;
	output wire busy;
	output reg done;
	output wire [ADDR_W - 1:0] rd_line;
	input wire [511:0] rd_data;
	localparam signed [31:0] noc_pkg_PAYLOAD_B = 64;
	localparam signed [31:0] noc_pkg_PAYLOAD_W = 512;
	output reg [549:0] inj_req;
	input wire inj_req_ready;
	reg [2:0] st;
	reg [1:0] setup_cnt;
	reg [15:0] sent;
	reg [ADDR_W - 1:0] cur_line;
	reg [17:0] cur_addr;
	reg [511:0] line_buf;
	assign busy = st != 3'd0;
	assign rd_line = cur_line;
	always @(*) begin
		if (_sv2v_0)
			;
		inj_req = 1'sb0;
		if (st == 3'd4) begin
			inj_req[549] = 1'b1;
			inj_req[548-:3] = 3'd0;
			inj_req[545-:4] = dst_tile;
			inj_req[541-:4] = MY_ID;
			inj_req[529-:18] = cur_addr;
			inj_req[511-:noc_pkg_PAYLOAD_W] = line_buf;
		end
	end
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			st <= 3'd0;
			done <= 1'b0;
			sent <= 0;
			setup_cnt <= 0;
		end
		else begin
			done <= 1'b0;
			case (st)
				3'd0:
					if (start) begin
						cur_line <= src_line;
						cur_addr <= dst_addr0;
						sent <= 0;
						setup_cnt <= 2'd2;
						st <= 3'd1;
					end
				3'd1:
					if (setup_cnt == 0)
						st <= 3'd2;
					else
						setup_cnt <= setup_cnt - 2'd1;
				3'd2: st <= 3'd3;
				3'd3: begin
					line_buf <= rd_data;
					st <= 3'd4;
				end
				3'd4:
					if (inj_req_ready) begin
						sent <= sent + 16'd1;
						cur_addr <= cur_addr + 18'd64;
						cur_line <= cur_line + 1'b1;
						if ((sent + 16'd1) == nlines)
							st <= 3'd5;
						else
							st <= 3'd2;
					end
				3'd5: begin
					done <= 1'b1;
					st <= 3'd0;
				end
				default: st <= 3'd0;
			endcase
		end
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
module tile_node (
	clk,
	rst_n,
	dma_start,
	dma_src_line,
	dma_dst_tile,
	dma_dst_addr0,
	dma_nlines,
	dma_done,
	ext_we,
	ext_waddr,
	ext_wdata,
	ext_raddr,
	ext_rdata,
	inj_req,
	inj_req_ready,
	eject_cw,
	eject_ccw
);
	reg _sv2v_0;
	parameter signed [31:0] ADDR_W = 12;
	localparam signed [31:0] noc_pkg_TID_W = 4;
	parameter [3:0] MY_ID = 1'sb0;
	input wire clk;
	input wire rst_n;
	input wire dma_start;
	input wire [ADDR_W - 1:0] dma_src_line;
	input wire [3:0] dma_dst_tile;
	input wire [17:0] dma_dst_addr0;
	input wire [15:0] dma_nlines;
	output wire dma_done;
	input wire ext_we;
	input wire [ADDR_W - 1:0] ext_waddr;
	input wire [511:0] ext_wdata;
	input wire [ADDR_W - 1:0] ext_raddr;
	output wire [511:0] ext_rdata;
	localparam signed [31:0] noc_pkg_PAYLOAD_B = 64;
	localparam signed [31:0] noc_pkg_PAYLOAD_W = 512;
	output wire [549:0] inj_req;
	input wire inj_req_ready;
	input wire [549:0] eject_cw;
	input wire [549:0] eject_ccw;
	reg sr_we;
	reg [ADDR_W - 1:0] sr_waddr;
	reg [ADDR_W - 1:0] sr_raddr0;
	reg [ADDR_W - 1:0] sr_raddr1;
	reg [511:0] sr_wdata;
	wire [511:0] sr_rdata0;
	wire [511:0] sr_rdata1;
	sram_2r1w #(
		.DATA_W(512),
		.DEPTH(1 << ADDR_W),
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
	wire [ADDR_W - 1:0] dma_rd_line;
	wire dma_busy;
	dma_engine #(
		.ADDR_W(ADDR_W),
		.MY_ID(MY_ID)
	) u_dma(
		.clk(clk),
		.rst_n(rst_n),
		.start(dma_start),
		.src_line(dma_src_line),
		.dst_tile(dma_dst_tile),
		.dst_addr0(dma_dst_addr0),
		.nlines(dma_nlines),
		.busy(dma_busy),
		.done(dma_done),
		.rd_line(dma_rd_line),
		.rd_data(sr_rdata0),
		.inj_req(inj_req),
		.inj_req_ready(inj_req_ready)
	);
	reg [549:0] ing;
	always @(*) begin
		if (_sv2v_0)
			;
		ing = 1'sb0;
		if (eject_cw[549] && (eject_cw[548-:3] == 3'd0))
			ing = eject_cw;
		else if (eject_ccw[549] && (eject_ccw[548-:3] == 3'd0))
			ing = eject_ccw;
	end
	always @(*) begin
		if (_sv2v_0)
			;
		if (ing[549]) begin
			sr_we = 1'b1;
			sr_waddr = ing[512 + (ADDR_W + 5):518];
			sr_wdata = ing[511-:noc_pkg_PAYLOAD_W];
		end
		else begin
			sr_we = ext_we;
			sr_waddr = ext_waddr;
			sr_wdata = ext_wdata;
		end
		sr_raddr0 = dma_rd_line;
		sr_raddr1 = ext_raddr;
	end
	assign ext_rdata = sr_rdata1;
	initial _sv2v_0 = 0;
endmodule
module noc_tile_sys (
	clk,
	rst_n,
	dma_start,
	dma_src_line,
	dma_dst_tile,
	dma_dst_addr0,
	dma_nlines,
	dma_done,
	ext_we,
	ext_waddr,
	ext_wdata,
	ext_raddr,
	ext_rdata
);
	parameter signed [31:0] ADDR_W = 12;
	input wire clk;
	input wire rst_n;
	localparam signed [31:0] noc_pkg_NUM_TILES = 16;
	input wire [15:0] dma_start;
	input wire [(noc_pkg_NUM_TILES * ADDR_W) - 1:0] dma_src_line;
	localparam signed [31:0] noc_pkg_TID_W = 4;
	input wire [(noc_pkg_NUM_TILES * noc_pkg_TID_W) - 1:0] dma_dst_tile;
	input wire [287:0] dma_dst_addr0;
	input wire [255:0] dma_nlines;
	output wire [15:0] dma_done;
	input wire [15:0] ext_we;
	input wire [(noc_pkg_NUM_TILES * ADDR_W) - 1:0] ext_waddr;
	input wire [8191:0] ext_wdata;
	input wire [(noc_pkg_NUM_TILES * ADDR_W) - 1:0] ext_raddr;
	output wire [8191:0] ext_rdata;
	localparam signed [31:0] noc_pkg_PAYLOAD_B = 64;
	localparam signed [31:0] noc_pkg_PAYLOAD_W = 512;
	wire [8799:0] inj_req;
	wire [8799:0] inj_resp;
	wire [0:15] inj_req_ready;
	wire [0:15] inj_resp_ready;
	wire [8799:0] eject_cw;
	wire [8799:0] eject_ccw;
	genvar _gv_t_1;
	generate
		for (_gv_t_1 = 0; _gv_t_1 < noc_pkg_NUM_TILES; _gv_t_1 = _gv_t_1 + 1) begin : g_tile
			localparam t = _gv_t_1;
			assign inj_resp[(15 - t) * 550+:550] = 1'sb0;
			tile_node #(
				.ADDR_W(ADDR_W),
				.MY_ID(t[3:0])
			) u_node(
				.clk(clk),
				.rst_n(rst_n),
				.dma_start(dma_start[t]),
				.dma_src_line(dma_src_line[(15 - t) * ADDR_W+:ADDR_W]),
				.dma_dst_tile(dma_dst_tile[(15 - t) * noc_pkg_TID_W+:noc_pkg_TID_W]),
				.dma_dst_addr0(dma_dst_addr0[(15 - t) * 18+:18]),
				.dma_nlines(dma_nlines[(15 - t) * 16+:16]),
				.dma_done(dma_done[t]),
				.ext_we(ext_we[t]),
				.ext_waddr(ext_waddr[(15 - t) * ADDR_W+:ADDR_W]),
				.ext_wdata(ext_wdata[(15 - t) * 512+:512]),
				.ext_raddr(ext_raddr[(15 - t) * ADDR_W+:ADDR_W]),
				.ext_rdata(ext_rdata[(15 - t) * 512+:512]),
				.inj_req(inj_req[(15 - t) * 550+:550]),
				.inj_req_ready(inj_req_ready[t]),
				.eject_cw(eject_cw[(15 - t) * 550+:550]),
				.eject_ccw(eject_ccw[(15 - t) * 550+:550])
			);
		end
	endgenerate
	ring_noc u_noc(
		.clk(clk),
		.rst_n(rst_n),
		.inj_req(inj_req),
		.inj_resp(inj_resp),
		.inj_req_ready(inj_req_ready),
		.inj_resp_ready(inj_resp_ready),
		.eject_cw(eject_cw),
		.eject_ccw(eject_ccw)
	);
endmodule
