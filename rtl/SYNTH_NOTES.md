# MOBOL RTL — Synthesis notes for Fusion Compiler

The RTL under `rtl/` is synthesizable SystemVerilog and lints clean
(`make lint`: no latches, no combinational loops, no width truncations,
fully-specified case defaults, async-reset flops). This note lists what a
Fusion Compiler (or Design Compiler) flow needs to know before elaboration.

## 1. Memory macros (do NOT synthesize the behavioral SRAM)

`sram/sram_2r1w.sv` models a 2-read/1-write SRAM as a `logic [] mem [...]`
array. Synthesized as-is it becomes a huge flop bank. Replace it with a
foundry-compiled macro:

- Geometry: 256 KB/tile, 512-bit (64 B) line, 4096 lines, **2 read + 1 write**
  ports. CACTI reference config: `sram/cacti/local_256kB.cfg`.
- Reconcile the macro's access time with the 1-cycle synchronous read the
  controllers assume at 500 MHz (ARCH_SPEC §8). If the macro needs 2 cycles,
  add a pipeline stage in the readers (DMA `CAP` state, tile `LDA/LDB`).
- Buffer-die SHARED banks (8 MB × 4) are separate larger macros; same 2R1W
  interface pattern.

## 2. Pipeline the transcendental units for timing

`fp32_rsqrt` (3 Newton iters), `fp32_recip` (4 iters) and `fp_exp2` (float
floor + degree-5 Horner) are single long combinational chains of the
`fp32_mul/fp32_add` cores — correct but far too deep for a fast clock. For
timing closure:

- Register between Newton iterations (rsqrt/recip) and between Horner stages
  (exp2). ~4–6 pipeline stages each hits a comfortable f32 clock.
- The VPU FSM (`vpu16.sv`) already sequences per-row; extend its latency
  accounting to cover the pipelined units (add wait states equal to the unit
  depth before consuming results).
- Alternatively swap these for a LUT + linear-interpolation unit (smaller,
  fixed 1–2 cycle latency) — then adopt the SAME LUT in the C++ simulator /
  Python golden so the whole toolchain stays bit-exact (ARCH_SPEC §10).

## 3. Also pipeline the MXU MAC and FP add/mul if targeting a high clock

- `mxu16` does one `fp32_mul + fp32_add` per PE per cycle (256 PEs). The
  mul→add path is combinational within a cycle; at a high clock, register the
  product (add a pipeline stage, adjust the k-serial accumulate accordingly).
- `fp32_add` uses a 96-bit exact fixed-point add + a 96-bit leading-one
  detector; this is the critical path in the FP core. A 2-stage split
  (align/add | normalize/round) closes timing at typical nodes.

## 4. Clocking / reset

- Single clock domain (`clk`) for the base die logic (500 MHz target). The
  DRAM-die clock domain (1 GHz controller) is external; the vertical-link
  crossing needs a CDC (async FIFO) at the bank↔DRAM boundary — not in this
  RTL (the DRAM controller is modeled behaviorally / by Ramulator2 in the perf
  simulator). Add standard CDC FIFOs at that interface for the real chip.
- All sequential logic uses `always_ff @(posedge clk or negedge rst_n)` with
  an async active-low reset — map `rst_n` to the reset tree.

## 5. Top-level ports use SV arrays / packed structs

`noc_tile_sys` / `tile_top` expose unpacked-array and packed-struct ports for
verification convenience. Fusion Compiler flattens these during elaboration;
if your flow prefers flat vectors, wrap the DUT or use `-flatten`. The flit
struct (`noc/noc_pkg.sv`) is a packed struct (synthesizes to a flat bus).

## 6. Suggested constraints (SDC starting point)

```tcl
create_clock -name clk -period 2.0 [get_ports clk]      # 500 MHz
set_input_delay  0.4 -clock clk [remove_from_collection [all_inputs] clk]
set_output_delay 0.4 -clock clk [all_outputs]
set_false_path -from [get_ports rst_n]
# After pipelining the transcendental/MXU units, tighten toward the target.
```

## 7. What is verified (functional sign-off inputs)

`make all` runs 10 Verilator testbenches; all pass. FP core / MXU / NMC / VPU
elementwise are bit-exact vs the C++ golden the cycle-accurate simulator uses;
NoC delivery, join counter, prefetch, DMA-through-NoC are directed-correct;
transcendentals and SILU/RMSNorm are tolerance-checked vs libm. Use these as
the functional reference when bringing up gate-level simulation.
