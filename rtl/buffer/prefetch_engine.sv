// prefetch_engine.sv — buffer-die weight prefetch DMA (ARCH_SPEC §6.5).
//
// Command-driven 2D strided copy from DRAM into this bank's SRAM, entirely on
// the buffer die (never touches the base die). Streams `rows` rows of
// `row_bytes` bytes each, with independent src/dst strides, in 64-byte
// (512-bit) beats. Issues read requests up the vertical column and writes the
// returned data into bank SRAM. On completion asserts `done` (the wrapper
// turns this into a RELEASE_TOK). Double-buffering across layers is a matter
// of the caller choosing dst_base parity; the engine itself is one transfer.
//
// Interfaces are simple valid/ready to keep it synthesizable and verifiable:
//   rd_*  : read request to DRAM (addr), response (data) assumed in order.
//   wr_*  : write to bank SRAM.
`include "common/mobol_pkg.sv"

module prefetch_engine #(
    parameter int BEAT_B = 64,
    parameter int BEAT_W = BEAT_B*8,          // 512
    parameter int ADDR_W = 40
) (
    input  logic               clk,
    input  logic               rst_n,

    // command
    input  logic               start,
    input  logic [ADDR_W-1:0]  src_base,      // DRAM byte address
    input  logic [ADDR_W-1:0]  dst_base,      // bank SRAM byte address
    input  logic [15:0]        rows,
    input  logic [15:0]        row_bytes,     // multiple of BEAT_B
    input  logic signed [31:0] src_stride,    // bytes between row starts
    input  logic signed [31:0] dst_stride,
    output logic               busy,
    output logic               done,

    // DRAM read channel (request addr, in-order data response)
    output logic               rd_req,
    output logic [ADDR_W-1:0]  rd_addr,
    input  logic               rd_gnt,        // request accepted this cycle
    input  logic               rd_data_valid,
    input  logic [BEAT_W-1:0]  rd_data,

    // bank SRAM write channel
    output logic               wr_en,
    output logic [ADDR_W-1:0]  wr_addr,
    output logic [BEAT_W-1:0]  wr_data
);
  typedef enum logic [1:0] {IDLE, RUN, DRAIN} state_e;
  state_e st;

  logic [15:0] row, off;               // current row, byte offset in row
  logic [15:0] wrow, woff;             // write-side position (data returns in order)
  logic [15:0] issued, done_beats, total_beats;

  // Address generators.
  wire [ADDR_W-1:0] cur_src = src_base +
        ADDR_W'($signed({16'd0, row}) * src_stride) + {24'd0, off};
  wire [ADDR_W-1:0] cur_dst = dst_base +
        ADDR_W'($signed({16'd0, wrow}) * dst_stride) + {24'd0, woff};

  wire [15:0] beats_per_row = row_bytes >> $clog2(BEAT_B);

  assign busy   = (st != IDLE);
  assign rd_req = (st == RUN) && (issued < total_beats);
  assign rd_addr = cur_src;
  assign wr_en   = rd_data_valid && (done_beats < total_beats);
  assign wr_addr = cur_dst;
  assign wr_data = rd_data;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE; done <= 1'b0;
      row<=0; off<=0; wrow<=0; woff<=0; issued<=0; done_beats<=0; total_beats<=0;
    end else begin
      done <= 1'b0;
      case (st)
        IDLE: if (start) begin
          row<=0; off<=0; wrow<=0; woff<=0; issued<=0; done_beats<=0;
          total_beats <= rows * (row_bytes >> $clog2(BEAT_B));
          st <= RUN;
        end
        RUN: begin
          // Issue a read when granted.
          if (rd_req && rd_gnt) begin
            issued <= issued + 16'd1;
            if (off + BEAT_B[15:0] >= row_bytes) begin off<=0; row<=row+16'd1; end
            else off <= off + BEAT_B[15:0];
          end
          // Consume returned data (in order) and write to SRAM.
          if (rd_data_valid && (done_beats < total_beats)) begin
            done_beats <= done_beats + 16'd1;
            if (woff + BEAT_B[15:0] >= row_bytes) begin woff<=0; wrow<=wrow+16'd1; end
            else woff <= woff + BEAT_B[15:0];
          end
          if ((done_beats + (rd_data_valid?16'd1:16'd0)) == total_beats &&
              total_beats != 0)
            st <= DRAIN;
        end
        DRAIN: begin
          done <= 1'b1;
          st   <= IDLE;
        end
      endcase
    end
  end
endmodule
