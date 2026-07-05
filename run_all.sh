#!/bin/bash
# MOBOL cycle-accurate simulator: build, verify, and run the full
# experiment matrix. DRAM die: 3D-stacked HBM3-class, 16 channels,
# hybrid-bonded columns (Ramulator2, logic:DRAM = 1:2).
set -e
cd "$(dirname "$0")"
R=config/ramulator_3d_dram.yaml

echo "════ 1. Build ════"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build build -j10 > /dev/null
echo "OK"

echo
echo "════ 2. Verification suite (functional + timing + cycle-accurate) ════"
(cd build && ctest --output-on-failure)

echo
echo "════ 3. Dual GEMM (E=(A×B)×D, 32×32 f16, fusion via SHARED) ════"
./build/src/mobol_sim --workload dual_gemm --ramulator $R

echo
echo "════ 4. LLM decoder prefill scaling (seq=16, d_model=64, h=4, ffn=256) ════"
for L in 1 2 4 8; do
  echo "--- $L layer(s) ---"
  ./build/src/mobol_sim --workload llm --layers $L --ramulator $R \
    | grep -E "Total logic|MXU ops|NoC|3D|DRAM \(|llm:"
done

echo
echo "════ 5. DSE: vertical hybrid-bond density (bits/column/cycle) ════"
for D in 512 1024 2048 4096 8192; do
  c=$(./build/src/mobol_sim --workload llm --layers 2 --ramulator $R --dram-density $D \
      | grep "Total logic" | awk '{print $5}')
  echo "  density ${D} bits: ${c} logic cycles"
done
echo "  (DDR4 planar baseline for contrast:)"
./build/src/mobol_sim --workload llm --layers 2 --ramulator config/ramulator_mobol.yaml --dram-txn 64 \
  | grep "Total logic" | sed 's/^/  /'

echo
echo "════ 6. Base-die consumption-port DSE (W = tile-link/wr-ports/dma-rate) ════"
for W in 1 2 4; do for D in 2048 4096 8192; do
  c=$(./build/src/mobol_sim --workload llm --layers 2 --ramulator $R \
      --tile-link $W --wr-ports $W --rd-ports $((W*2)) --dma-rate $W --dram-density $D \
      | grep "Total logic" | awk '{print $5}')
  echo "  W=$W density=$D: $c logic cycles"
done; done

echo
echo "════ 7. Three architecture paradigms (llm 2/8 layers, W=1 and W=2) ════"
for A in baseline nobuffer nmc; do for W in 1 2; do for L in 2 8; do
  c=$(./build/src/mobol_sim --workload llm --layers $L --arch $A --ramulator $R \
      --tile-link $W --wr-ports $W --rd-ports $((W*2)) --dma-rate $W \
      | grep "Total logic" | awk '{print $5}')
  echo "  arch=$A W=$W layers=$L: $c logic cycles"
done; done; done

echo
echo "════ 8. Weight streaming (bank DMA prefetch, double-buffered) ════"
for A in baseline nmc; do for S in "" "--stream-weights"; do for W in 1 2; do
  c=$(./build/src/mobol_sim --workload llm --layers 2 --arch $A $S --ramulator $R \
      --tile-link $W --wr-ports $W --rd-ports $((W*2)) --dma-rate $W \
      | grep "Total logic" | awk '{print $5}')
  st=off; [ -n "$S" ] && st=on
  echo "  arch=$A stream=$st W=$W: $c logic cycles"
done; done; done

echo
echo "════ 8b. Sensitivity: SHARED port contention modeling ON ════"
./build/src/mobol_sim --workload llm --layers 2 --ramulator $R --shared-contention \
  | grep -E "Total logic|SHARED|llm:"

echo
echo "════ 9. Full 2-layer LLM report (all paradigms in test 5 above) ════"
./build/src/mobol_sim --workload llm --layers 2 --ramulator $R

echo
echo "All simulations complete."
