#!/bin/bash
# Validation gate (§3.3) for one structural variant build.
#   validate_variant.sh <variant_dir>
# where <variant_dir> = dse_bo/builds/tN_gG_sMB_topo (from gen_variant.py).
#
# Gate 1: ctest suite compiled at the variant's own structural constants
#         (address widths, functional, timing, cycle-accurate micro KATs;
#         the 16-tile llm/dual_gemm KATs auto-register only on baseline).
# Gate 2: f16 main-path bit-exactness: edgc spec_target compiled FOR this
#         variant must satisfy golden == reference == sim.
#
# Prints PASS or INVALID as the last line. Exit 0 iff PASS.
set -u
VDIR="$(cd "$1" && pwd)"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BDIR="$VDIR/build"
YAML="$VDIR/mobol_arch.yaml"
LOG="$VDIR/validate.log"

fail() { echo "gate failed: $1" >> "$LOG"; echo "INVALID"; exit 1; }

: > "$LOG"
[ -x "$BDIR/src/trace_sim" ] || fail "trace_sim missing"

# ── Gate 1: ctest ──
(cd "$BDIR" && ctest --output-on-failure) >> "$LOG" 2>&1 \
  || fail "ctest"

# ── Gate 2: golden == reference == sim on spec_target ──
(cd "$ROOT/tools/edgc" && \
 MOBOL_ARCH_YAML="$YAML" python3 edgc_cc.py --model spec_target \
   --out "$VDIR/spec_check" --check --sim "$BDIR/src/trace_sim") \
   >> "$LOG" 2>&1 || fail "edgc_cc"
grep -q "ref=PASS sim=PASS" "$LOG" || fail "bit-exact compare"

echo "PASS"
