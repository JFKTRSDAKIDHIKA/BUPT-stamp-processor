# tile_top synthesis file list (paths relative to repo root).
# NOTE: uses sram_2r1w_macro.sv (macro-backed), NOT sram/sram_2r1w.sv
# (behavioral) — the two define the same module and must never be mixed.
rtl/common/mobol_pkg.sv
rtl/fp/f16_to_f32.sv
rtl/fp/f32_to_f16.sv
rtl/fp/fp32_mul.sv
rtl/fp/fp32_add.sv
rtl/mxu/mxu16.sv
rtl/tile/join_ctr.sv
rtl/sram/sram_2r1w_macro.sv
rtl/tile/tile_top.sv
