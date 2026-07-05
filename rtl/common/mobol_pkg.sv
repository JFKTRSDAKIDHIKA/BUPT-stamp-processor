// mobol_pkg.sv — architecture parameters for the MOBOL RTL.
//
// These mirror config/mobol_arch.yaml (the single source of truth for the
// simulator/compiler). For RTL, structural constants are elaboration-time
// parameters; a future step can generate this package from the YAML so the
// RTL, C++ simulator and Python compiler share one parameter set.
//
// Numeric formats follow ARCH_SPEC §10 and src/common/f16.h exactly.
`ifndef MOBOL_PKG_SV
`define MOBOL_PKG_SV

package mobol_pkg;

  // ── Structural (ARCH_SPEC §1) ──────────────────────────────
  localparam int NUM_TILES       = 16;
  localparam int NUM_BANKS       = 4;
  localparam int TILES_PER_GROUP = 4;

  // ── MXU native tile (ARCH_SPEC §3.2) ───────────────────────
  localparam int MXU_M = 16;
  localparam int MXU_N = 16;
  localparam int MXU_K = 16;

  // ── Numeric widths ─────────────────────────────────────────
  localparam int F16_W = 16;
  localparam int F32_W = 32;

  // f16 field extents (IEEE-754 binary16)
  localparam int F16_EXP_W = 5;
  localparam int F16_MAN_W = 10;
  localparam int F16_BIAS  = 15;

  // f32 field extents (IEEE-754 binary32)
  localparam int F32_EXP_W = 8;
  localparam int F32_MAN_W = 23;
  localparam int F32_BIAS  = 127;

  // ── LOCAL scratchpad (ARCH_SPEC §3.5) ──────────────────────
  localparam int LOCAL_BYTES = 1 << 18;   // 256 KB / tile
  localparam int SRAM_PORT_W = 64;         // bytes / port access

endpackage : mobol_pkg

`endif
