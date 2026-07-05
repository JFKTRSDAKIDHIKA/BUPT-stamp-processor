// sram_2r1w.sv — behavioral 2-read / 1-write SRAM (LOCAL scratchpad model).
//
// ARCH_SPEC §3.5: per-tile LOCAL SRAM, 2 read + 1 write ports, 64 B/port.
// This is a BEHAVIORAL model for functional Verilator simulation. Physical
// timing/area/energy are modeled separately with CACTI; the intended CACTI
// invocation for one tile's LOCAL is documented in sram/cacti/local_256kB.cfg
// (256 KB, 512-bit = 64 B line, 2 read + 1 write port, target tech node).
// A synthesizable design would instantiate a foundry macro with the matching
// geometry; the read latency the memory controller assumes (1 cycle here)
// must be reconciled with CACTI's reported access time at the target clock.
`ifndef SRAM_2R1W_SV
`define SRAM_2R1W_SV

module sram_2r1w #(
    parameter int DATA_W = 512,           // 64 B line
    parameter int DEPTH  = (1<<18)/64,    // 256 KB / 64 B = 4096 lines
    parameter int ADDR_W = $clog2(DEPTH)
) (
    input  logic              clk,
    // write port
    input  logic              we,
    input  logic [ADDR_W-1:0] waddr,
    input  logic [DATA_W-1:0] wdata,
    // read port 0
    input  logic [ADDR_W-1:0] raddr0,
    output logic [DATA_W-1:0] rdata0,
    // read port 1
    input  logic [ADDR_W-1:0] raddr1,
    output logic [DATA_W-1:0] rdata1
);
  logic [DATA_W-1:0] mem [DEPTH];

  // Synchronous write, synchronous read (1-cycle latency), read-after-write
  // to the same address returns the OLD value this cycle (write-first would
  // change the memory controller's hazard assumptions).
  always_ff @(posedge clk) begin
    if (we) mem[waddr] <= wdata;
    rdata0 <= mem[raddr0];
    rdata1 <= mem[raddr1];
  end
endmodule

`endif
