import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgc import ModelConfig, gen_weights
from edgc import numerics as N
from edgc.compile import Compiler, Sched
from edgc.golden import Golden
import edgc.model as M

cfg = ModelConfig(name="dbg", d_model=64, n_layers=1, n_heads=4, n_kv_heads=4,
                  d_head=16, ffn_hidden=128, seq_len=16, norm="rmsnorm",
                  pos="none", ffn="gelu", attn_mask="full")
w = gen_weights(cfg)

# Monkeypatch: capture the DRAM offset of xn (first norm out) by adding a dump.
comp = Compiler(cfg, w, Sched(), "config/ramulator_3d_dram.yaml")
orig_layer = comp._layer
stages = {}
def layer_hook(l, x_in, x_out, routes):
    # replicate first steps with dumps
    return orig_layer(l, x_in, x_out, routes)
# Instead, directly instrument _norm to record first output base.
orig_norm = comp._norm
first = {"done": False}
def norm_hook(x_base, out_base, S, dm):
    orig_norm(x_base, out_base, S, dm)
    if not first["done"]:
        comp.b.add_dump("xn0", out_base, S * dm * 2)
        stages["xn0_base"] = out_base
        first["done"] = True
comp._norm = norm_hook
tb = comp.compile()
tb.write_trace("/tmp/dbg.trace"); tb.write_mem("/tmp/dbg.mem")

g = Golden("/tmp/dbg.trace", "/tmp/dbg.mem")
g.run()
xn0 = N.bytes_to_f16(g.dump("xn0"))

# reference: rmsnorm of x0 rows
x = [N.f16_bits_to_f32(b) for b in w.x0]
ref = []
for i in range(cfg.seq_len):
    row = x[i*cfg.d_model:(i+1)*cfg.d_model]
    ref.extend(N.rmsnorm(row, cfg.norm_eps))
ref16 = N.to_f16_array(ref)
diffs = sum(1 for a,b in zip(xn0, ref16) if a!=b)
print("norm stage diffs:", diffs, "/", len(ref16))
if diffs:
    for i in range(len(ref16)):
        if xn0[i] != ref16[i]:
            print(f"  first diff idx {i}: sim={N.f16_bits_to_f32(xn0[i]):.5f} ref={N.f16_bits_to_f32(ref16[i]):.5f}")
            break

d=[i for i in range(len(ref16)) if xn0[i]!=ref16[i]]
print('rows with diffs:', sorted(set(i//64 for i in d)))
print('cols with diffs:', sorted(set(i%64 for i in d)))
