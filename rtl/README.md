# MOBOL RTL — synthesizable Verilog for backend (PPA) evaluation

RTL implementation of the MOBOL accelerator's key modules (ARCH_SPEC.md),
verified functionally with Verilator against the C++ golden the cycle-accurate
simulator uses. SRAM is a behavioral model; physical macros/CACTI parameters
are documented per array (see `sram/`).

## Verified modules

| Module | File | Verification | Status |
|:--|:--|:--|:--|
| f16↔f32, fp32 mul/add | `fp/` | bit-exact vs f16.h + C float (`tb_fp`) | ✅ 1.17M checks, 0 fail |
| MXU 16×16×16 | `mxu/mxu16.sv` | bit-exact vs blockops::mxu (`tb_mxu`) | ✅ 300 cases, 0 fail |
| Ring NoC (2 VC, 16 node) | `noc/` | all packets delivered (`tb_router`) | ✅ 6000/6000, 0 mis |
| NMC reduce engine | `buffer/nmc_engine.sv` | bit-exact vs blockops (`tb_nmc`) | ✅ 200 cases, 0 fail |
| Join counter (release/acquire) | `tile/join_ctr.sv` | directed (`tb_join`) | ✅ PASSED |
| Weight prefetch engine | `buffer/prefetch_engine.sv` | 2D copy correctness (`tb_prefetch`) | ✅ 50 cases, 0 fail |
| Transcendentals rsqrt/recip/exp2 | `fp/fp32_rsqrt,recip,fp_exp2.sv` | tolerance vs libm (`tb_transcend`) | ✅ 200k pts, 0 over-tol |
| VPU 16-lane (add/mul/scale/SILU/RMSNorm) | `vpu/vpu16.sv` | elementwise bit-exact + SILU/RMS tol (`tb_vpu`) | ✅ 0 fail |
| Compute tile (SRAM+MXU+join+seq) | `tile/tile_top.sv` | matmul from SRAM, bit-exact (`tb_tile`) | ✅ 40 cases, 0 fail |
| DMA + NoC system (16 tile_node + ring) | `tile/noc_tile_sys.sv` | tile→tile DMA thru fabric (`tb_noc_tile`) | ✅ 16 xfers, 0 fail |

Every testbench passes: `make all` → 10/10 green. `make lint` is clean (no
latches, no combinational loops, no width errors) — synthesis-ready.

## Transcendental units (VPU)

`fp32_rsqrt / fp32_recip` (Newton) and `fp_exp2` (float-floor + degree-5
minimax) are built entirely from the bit-exact `fp32_mul/fp32_add` cores, so
each is DETERMINISTIC (reproducible by an identical software model — the
shared-approximation path of ARCH_SPEC §10). Accuracy: rsqrt/recip ~f32-exact,
exp2 rel err 5.8e-5. exp(x)=exp2(x·log2e), sigmoid/SiLU via exp2+recip,
1/sqrt(var+eps) via rsqrt. To make the whole toolchain bit-exact *including*
these ops, swap the simulator/golden softmax/GELU/norm to call these same
approximations (currently libm) — a defined, self-contained follow-on.

## DMA + NoC integration

`tile_node` = LOCAL SRAM + `dma_engine` (LOCAL→remote push, emits DATA_WRITE
flits) + NoC ingress (eject → SRAM write). `noc_tile_sys` wires 16 of them to
`ring_noc`; `tb_noc_tile` verifies any tile can DMA-push a region into any
other tile's SRAM through the fabric. NOTE: a single ring can starve injection
under fully-synchronized same-direction saturation (a known ring QoS limit);
realistic (compiler-scheduled) traffic does not hit it. Adding injection
credits/fairness for adversarial QoS is a backend option.

## Synthesis (Fusion Compiler) notes — see `SYNTH_NOTES.md`

- **SRAM**: behavioral 2R1W (`sram/sram_2r1w.sv`) synthesizes to flops — swap
  for a foundry macro (geometry in `sram/cacti/local_256kB.cfg`) at synth.
- **Transcendental units** are deep combinational chains (13+ FP ops); pipeline
  them (register between Newton/Horner stages) for timing closure.
- `tile_top` matmul micro-op runs `acc=0`; accumulate-from-SRAM (Cin) is a TODO.

## Build & run

```bash
cd rtl && make all          # build + run every testbench
make fp mxu router          # individual
```

Requires Verilator (5.x). The C reference is compiled `-ffp-contract=off` so C
float ops are two-rounding, matching the whole toolchain's golden.

## Numeric spec note

`tb_fp` exposed a latent C undefined-behavior bug in `src/common/f16.h`
(deep-underflow `shift >= 32` masked mod 32) — fixed there so f16.h is now
correct-IEEE and consistent with the RTL and the Python golden. The whole
C++/compiler toolchain re-validated bit-exact after the fix.
