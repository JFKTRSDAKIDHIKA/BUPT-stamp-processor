#!/usr/bin/env python3
"""edgc driver: compile a model, validate bit-exactness, optionally DSE.

  edgc_cc.py --model MODEL [--out DIR] [--sched LABEL] [--dse]
             [--sim PATH] [--ramulator CFG] [--check]

MODEL is a key in models.py (edge_dense, edge_gqa, edge_swiglu_rope,
edge_moe, edge_sliding, ...). Emits <out>/<model>.trace + .mem, runs the
golden interpreter, and (with --check) compares golden vs the reference
forward. With --sim, runs the C++ simulator and compares golden vs sim.
With --dse, sweeps schedules and reports the best by cycle count.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgc import ModelConfig, gen_weights, reference_forward, Compiler, Sched
from edgc.golden import Golden
from edgc import numerics as N
from edgc.arch import load_arch, resolve_ramulator
import models as MZOO


def scheds_from_arch(arch):
    """Build Sched objects from the arch YAML's compiler.dse list (single
    source of truth for the DSE search space)."""
    out = []
    for d in arch.dse:
        out.append(Sched(
            label=d.get("label", "sched"),
            tile_link=d.get("tile_link", 1),
            wr_ports=d.get("wr_ports", 1),
            rd_ports=d.get("rd_ports", 2),
            dma_rate=d.get("dma_rate", 1),
            dram_density=d.get("dram_density", 2048),
            nmc=d.get("nmc", False)))
    return out or [Sched()]


def build(cfg, out_dir, sched, ramulator):
    w = gen_weights(cfg)
    comp = Compiler(cfg, w, sched, ramulator)
    tb = comp.compile()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, cfg.name)
    tb.write_trace(base + ".trace")
    tb.write_mem(base + ".mem")
    return w, comp, base


def check_golden_vs_ref(cfg, w, comp, base):
    g = Golden(base + ".trace", base + ".mem")
    g.run()
    out = g.dump("out")
    ref = reference_forward(cfg, w)
    ref_bytes = N.f16_bytes(ref)
    if out == ref_bytes:
        return True, 0, 0.0
    # summarize
    a = N.bytes_to_f16(out)
    diffs = sum(1 for x, y in zip(a, ref) if x != y)
    maxerr = max((abs(N.f16_bits_to_f32(x) - N.f16_bits_to_f32(y))
                  for x, y in zip(a, ref)), default=0.0)
    return False, diffs, maxerr


def run_sim(sim_path, base, ramulator, out_dir):
    r = subprocess.run([sim_path, base + ".trace", base + ".mem",
                        "--ramulator", ramulator, "--out-dir", out_dir, "--json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"trace_sim failed ({r.returncode})")
    return json.loads(r.stdout.strip().splitlines()[-1])


def check_golden_vs_sim(base, out_dir):
    g = Golden(base + ".trace", base + ".mem")
    g.run()
    gold = g.dump("out")
    with open(os.path.join(out_dir, "out.bin"), "rb") as f:
        sim = f.read()
    return gold == sim, len(gold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="build/edgc_out")
    ap.add_argument("--sched", default="")
    ap.add_argument("--arch", default="", help="unified architecture YAML")
    ap.add_argument("--ramulator", default="")
    ap.add_argument("--sim", default="")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--dse", action="store_true")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    # Single source of truth: schedules and the default ramulator come from
    # the unified architecture YAML.
    arch = load_arch(args.arch)
    all_scheds = scheds_from_arch(arch)
    if not args.ramulator:
        args.ramulator = resolve_ramulator(arch)
    default_sched = args.sched or arch.default_sched

    if args.model not in MZOO.MODELS:
        sys.exit(f"unknown model '{args.model}'; have: {', '.join(MZOO.MODELS)}")
    cfg = MZOO.MODELS[args.model]()

    scheds = all_scheds if args.dse else [
        next((s for s in all_scheds if s.label == default_sched), all_scheds[0])]

    results = []
    for sched in scheds:
        w, comp, base = build(cfg, args.out, sched, args.ramulator)
        rec = {"model": cfg.name, "sched": sched.label,
               "instrs": sum(len(t.code) for t in comp.b.tiles)}
        if args.check:
            ok, diffs, maxerr = check_golden_vs_ref(cfg, w, comp, base)
            rec["golden_vs_ref"] = "PASS" if ok else f"FAIL({diffs},{maxerr:.4g})"
        if args.sim:
            stats = run_sim(args.sim, base, args.ramulator, args.out)
            ok, n = check_golden_vs_sim(base, args.out)
            rec["golden_vs_sim"] = "PASS" if ok else "FAIL"
            rec["cycles"] = stats["total_cycles"]
            rec["mxu_ops"] = stats["mxu_ops"]
            rec["dram_rd_bytes"] = stats["dram_read_bytes"]
            rec["causality"] = stats["causality_ok"]
        results.append(rec)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for rec in results:
            line = f"[{rec['model']}/{rec['sched']}] instrs={rec['instrs']}"
            if "golden_vs_ref" in rec:
                line += f" ref={rec['golden_vs_ref']}"
            if "golden_vs_sim" in rec:
                line += f" sim={rec['golden_vs_sim']} cycles={rec['cycles']}"
            print(line)
        if args.dse and any("cycles" in r for r in results):
            best = min((r for r in results if "cycles" in r), key=lambda r: r["cycles"])
            print(f"\nDSE winner: {best['sched']} @ {best['cycles']} cycles")


if __name__ == "__main__":
    main()
