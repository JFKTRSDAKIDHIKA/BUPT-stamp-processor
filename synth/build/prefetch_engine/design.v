module prefetch_engine (
	clk,
	rst_n,
	start,
	src_base,
	dst_base,
	rows,
	row_bytes,
	src_stride,
	dst_stride,
	busy,
	done,
	rd_req,
	rd_addr,
	rd_gnt,
	rd_data_valid,
	rd_data,
	wr_en,
	wr_addr,
	wr_data
);
	parameter signed [31:0] BEAT_B = 64;
	parameter signed [31:0] BEAT_W = BEAT_B * 8;
	parameter signed [31:0] ADDR_W = 40;
	input wire clk;
	input wire rst_n;
	input wire start;
	input wire [ADDR_W - 1:0] src_base;
	input wire [ADDR_W - 1:0] dst_base;
	input wire [15:0] rows;
	input wire [15:0] row_bytes;
	input wire signed [31:0] src_stride;
	input wire signed [31:0] dst_stride;
	output wire busy;
	output reg done;
	output wire rd_req;
	output wire [ADDR_W - 1:0] rd_addr;
	input wire rd_gnt;
	input wire rd_data_valid;
	input wire [BEAT_W - 1:0] rd_data;
	output wire wr_en;
	output wire [ADDR_W - 1:0] wr_addr;
	output wire [BEAT_W - 1:0] wr_data;
	reg [1:0] st;
	reg [15:0] row;
	reg [15:0] off;
	reg [15:0] wrow;
	reg [15:0] woff;
	reg [15:0] issued;
	reg [15:0] done_beats;
	reg [15:0] total_beats;
	function automatic signed [ADDR_W - 1:0] sv2v_cast_23040_signed;
		input reg signed [ADDR_W - 1:0] inp;
		sv2v_cast_23040_signed = inp;
	endfunction
	wire [ADDR_W - 1:0] cur_src = (src_base + sv2v_cast_23040_signed($signed({16'd0, row}) * src_stride)) + {24'd0, off};
	wire [ADDR_W - 1:0] cur_dst = (dst_base + sv2v_cast_23040_signed($signed({16'd0, wrow}) * dst_stride)) + {24'd0, woff};
	wire [15:0] beats_per_row = row_bytes >> $clog2(BEAT_B);
	assign busy = st != 2'd0;
	assign rd_req = (st == 2'd1) && (issued < total_beats);
	assign rd_addr = cur_src;
	assign wr_en = rd_data_valid && (done_beats < total_beats);
	assign wr_addr = cur_dst;
	assign wr_data = rd_data;
	always @(posedge clk or negedge rst_n)
		if (!rst_n) begin
			st <= 2'd0;
			done <= 1'b0;
			row <= 0;
			off <= 0;
			wrow <= 0;
			woff <= 0;
			issued <= 0;
			done_beats <= 0;
			total_beats <= 0;
		end
		else begin
			done <= 1'b0;
			case (st)
				2'd0:
					if (start) begin
						row <= 0;
						off <= 0;
						wrow <= 0;
						woff <= 0;
						issued <= 0;
						done_beats <= 0;
						total_beats <= rows * (row_bytes >> $clog2(BEAT_B));
						st <= 2'd1;
					end
				2'd1: begin
					if (rd_req && rd_gnt) begin
						issued <= issued + 16'd1;
						if ((off + BEAT_B[15:0]) >= row_bytes) begin
							off <= 0;
							row <= row + 16'd1;
						end
						else
							off <= off + BEAT_B[15:0];
					end
					if (rd_data_valid && (done_beats < total_beats)) begin
						done_beats <= done_beats + 16'd1;
						if ((woff + BEAT_B[15:0]) >= row_bytes) begin
							woff <= 0;
							wrow <= wrow + 16'd1;
						end
						else
							woff <= woff + BEAT_B[15:0];
					end
					if (((done_beats + (rd_data_valid ? 16'd1 : 16'd0)) == total_beats) && (total_beats != 0))
						st <= 2'd2;
				end
				2'd2: begin
					done <= 1'b1;
					st <= 2'd0;
				end
			endcase
		end
endmodule
