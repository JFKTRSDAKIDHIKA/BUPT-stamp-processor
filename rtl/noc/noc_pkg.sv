// noc_pkg.sv — flit format and NoC parameters (ARCH_SPEC §5).
`ifndef NOC_PKG_SV
`define NOC_PKG_SV

package noc_pkg;
  localparam int NUM_TILES = 16;
  localparam int TID_W     = 4;                 // $clog2(16)
  localparam int PAYLOAD_B = 64;                // flit payload bytes
  localparam int PAYLOAD_W = PAYLOAD_B*8;       // 512 bits

  // Flit kinds (ARCH_SPEC §5). VC0 = request, VC1 = response.
  typedef enum logic [2:0] {
    DATA_WRITE  = 3'd0,   // -> memory target
    WRITE_ACK   = 3'd1,   // -> requester
    READ_REQ    = 3'd2,   // -> memory target
    READ_RESP   = 3'd3,   // -> requester (payload)
    RELEASE_TOK = 3'd4    // -> target tile (sync)
  } flit_kind_e;

  // A tile-directed ring flit. (Memory-target routing / vertical links are
  // handled at the tile/bank boundary; on the ring every flit carries its
  // final destination tile.)
  typedef struct packed {
    logic                  valid;
    flit_kind_e            kind;
    logic [TID_W-1:0]      dst;      // destination tile
    logic [TID_W-1:0]      src;      // originating tile
    logic [7:0]            tag;      // RELEASE_TOK tag / dma seq
    logic [17:0]           addr;     // DATA_WRITE: dst LOCAL byte address
    logic [PAYLOAD_W-1:0]  payload;  // one 64 B line
  } flit_t;

  function automatic logic is_resp(flit_kind_e k);
    is_resp = (k == WRITE_ACK) || (k == READ_RESP) || (k == RELEASE_TOK);
  endfunction

  // Shortest-direction on the bidirectional ring: 0 = clockwise (+1),
  // 1 = counter-clockwise. Tie -> clockwise.
  function automatic logic ring_dir(logic [TID_W-1:0] from, logic [TID_W-1:0] to);
    int cw;
    cw = (int'(to) - int'(from) + NUM_TILES) % NUM_TILES;
    ring_dir = (cw <= NUM_TILES - cw) ? 1'b0 : 1'b1;
  endfunction
endpackage

`endif
