#!/bin/bash
# End-to-end edge-LLM compiler pipeline: compile each model to a trace,
# validate golden==reference (compiler correct) and golden==simulator
# (simulator executes the ISA exactly), all bit-exact, then DSE + spec-decode.
set -e
cd "$(dirname "$0")"
ROOT=../..
SIM=$ROOT/build/src/trace_sim
OUT=/tmp/edgc_out
mkdir -p $OUT
# Ramulator/schedules come from the unified arch YAML (config/mobol_arch.yaml).

if [ ! -x "$SIM" ]; then
  echo "building trace_sim..."; cmake --build $ROOT/build --target trace_sim -j10 >/dev/null
fi

echo "════ Edge-LLM compiler: bit-exact validation (golden==ref AND golden==sim) ════"
for m in edge_dense edge_gqa edge_mqa edge_sliding edge_moe; do
  python3 edgc_cc.py --model $m --out $OUT --check --sim $SIM
done

echo
echo "════ DSE: schedule search on edge_gqa ════"
python3 edgc_cc.py --model edge_gqa --out $OUT --dse --sim $SIM

echo
echo "════ Speculative decoding cost model ════"
python3 spec_decode.py --sim $SIM --out $OUT --gamma 8

echo
echo "All edge-LLM compiler checks passed."
