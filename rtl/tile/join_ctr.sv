// join_ctr.sv — tag-based release/acquire join counter table (ARCH_SPEC §9).
//
// Per consumer tile: a small table cnt[tag]. Increment on a local release or a
// remote RELEASE_TOK arriving from the NoC. acquire(tag, arity) passes when
// cnt[tag] >= arity and then consumes arity (generation auto-reset). This is
// the only hardware carrier of dependency correctness (no cache coherence).
//
// One release and one acquire may be presented per cycle. If both target the
// same tag, the release is applied first (a token that arrives the same cycle
// an acquire is offered can satisfy it) — matching the simulator's ordering.
`include "common/mobol_pkg.sv"

module join_ctr #(
    parameter int NTAGS = 64,
    parameter int CNT_W = 12,               // max in-flight per tag
    parameter int TAG_W = $clog2(NTAGS)
) (
    input  logic              clk,
    input  logic              rst_n,

    // release: local (from this tile's RELEASE) or remote (NoC token).
    input  logic              rel_valid,
    input  logic [TAG_W-1:0]  rel_tag,

    // acquire request (combinational pass indication + registered consume).
    input  logic              acq_valid,
    input  logic [TAG_W-1:0]  acq_tag,
    input  logic [CNT_W-1:0]  acq_arity,
    output logic              acq_pass,     // 1 if this acquire can pass now

    // debug/observability
    output logic [CNT_W-1:0]  dbg_cnt [NTAGS]
);
  logic [CNT_W-1:0] cnt [NTAGS];

  // Effective count seen by an acquire this cycle includes a same-cycle
  // release to the same tag.
  logic [CNT_W-1:0] eff_cnt;
  always_comb begin
    eff_cnt = cnt[acq_tag];
    if (rel_valid && (rel_tag == acq_tag)) eff_cnt = eff_cnt + 1'b1;
    acq_pass = acq_valid && (eff_cnt >= acq_arity);
  end

  integer t;
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (t=0;t<NTAGS;t++) cnt[t] <= '0;
    end else begin
      // Apply release, then a passing acquire's consume. Same-tag case handled
      // by computing the net delta.
      if (rel_valid) cnt[rel_tag] <= cnt[rel_tag] + 1'b1;
      if (acq_pass) begin
        if (rel_valid && (rel_tag == acq_tag))
          cnt[acq_tag] <= cnt[acq_tag] + 1'b1 - acq_arity;   // release + consume
        else
          cnt[acq_tag] <= cnt[acq_tag] - acq_arity;
      end
    end
  end

  genvar g;
  generate for (g=0;g<NTAGS;g++) assign dbg_cnt[g] = cnt[g]; endgenerate
endmodule
