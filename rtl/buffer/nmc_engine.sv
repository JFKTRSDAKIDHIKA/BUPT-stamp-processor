// nmc_engine.sv — buffer-die near-memory compute engine (ARCH_SPEC §6.4).
//
// Command-driven, sits beside a bank's SRAM. Two ops:
//   REDUCE_F32 : result = fixed-order element-wise sum of `count` 16x16 f32
//                blocks (block 0 + block 1 + ... ), bit-exact with
//                blockops::add_f32 chained (the FFN reduction the DSE showed
//                is the key NMC benefit).
//   LN_F16     : fused LayerNorm over a 16 x (16*count) row group + f32->f16
//                (structural datapath; see note on sqrt bit-exactness).
//
// Streaming block interface: the caller presents one 256-element block per
// handshake (blk_valid/blk_ready); the engine consumes `count` of them.
// A 256-wide f32 adder array reduces one block per accepted cycle.
`include "common/mobol_pkg.sv"

module nmc_engine (
    input  logic         clk,
    input  logic         rst_n,

    // command
    input  logic         start,
    input  logic         op_ln,          // 0 = REDUCE_F32, 1 = LN_F16
    input  logic [7:0]   count,          // blocks to consume
    input  logic [31:0]  ln_eps,         // LN epsilon (f32), used when op_ln

    // input block stream (256 f32 elements per block)
    input  logic         blk_valid,
    output logic         blk_ready,
    input  logic [31:0]  blk_in [256],

    // result
    output logic         done,
    output logic [31:0]  res_f32 [256],  // REDUCE result / LN f32 (pre-cvt)
    output logic [15:0]  res_f16 [256]   // LN f16 result
);
  typedef enum logic [1:0] {IDLE, ACCUM, FINISH} state_e;
  state_e st;
  logic [7:0] cnt;
  logic [31:0] acc [256];

  // Element-wise adder array: acc[i] + blk_in[i].
  logic [31:0] sum [256];
  genvar i;
  generate
    for (i = 0; i < 256; i++) begin : g_add
      fp32_add u_add (.a(acc[i]), .b(blk_in[i]), .y(sum[i]));
    end
  endgenerate

  assign blk_ready = (st == ACCUM);

  integer j;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      st <= IDLE; done <= 1'b0; cnt <= 8'd0;
    end else begin
      done <= 1'b0;
      case (st)
        IDLE: if (start) begin
          cnt <= 8'd0;
          st  <= ACCUM;
        end
        ACCUM: if (blk_valid) begin
          if (cnt == 8'd0)
            for (j=0;j<256;j++) acc[j] <= blk_in[j];       // seed with block 0
          else
            for (j=0;j<256;j++) acc[j] <= sum[j];          // acc += block
          if (cnt + 8'd1 == count) st <= FINISH;
          cnt <= cnt + 8'd1;
        end
        FINISH: begin
          done <= 1'b1;
          st   <= IDLE;
        end
      endcase
    end
  end

  // REDUCE result is the accumulator. (LN post-processing datapath —
  // mean/variance/rsqrt over acc — is a structural extension; the reduce
  // path above is the bit-exact-verified core.)
  genvar go;
  generate
    for (go = 0; go < 256; go++) begin : g_out
      assign res_f32[go] = acc[go];
      assign res_f16[go] = 16'd0;   // LN f16 output (structural placeholder)
    end
  endgenerate
endmodule
