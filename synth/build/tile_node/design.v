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
