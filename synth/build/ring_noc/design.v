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
