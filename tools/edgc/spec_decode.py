#!/usr/bin/env python3
"""Speculative decoding driver.

Speculative decoding runs a cheap DRAFT model to propose gamma tokens, then
the expensive TARGET model verifies all gamma in ONE parallel forward pass
(instead of gamma sequential steps). The accepted prefix length is
data-dependent — the compiler resolves it statically from the actual logits.

This driver:
  1. compiles + simulates the draft model (proposal cost),
  2. compiles + simulates the target model over the gamma positions
     (one parallel verify pass),
  3. computes the accepted length from the reference (argmax agreement),
  4. reports the effective speedup vs plain sequential target decoding.

Both models are validated bit-exact (golden==sim) before their cycle counts
are used, so the cost numbers rest on a verified execution.
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgc import ModelConfig, gen_weights, reference_forward
from edgc.compile import Compiler, Sched
from edgc.golden import Golden
from edgc import numerics as N
from edgc.arch import load_arch, resolve_ramulator
import models as MZOO


def compile_run(cfg, sched, ramulator, sim, out_dir):
    w = gen_weights(cfg)
    comp = Compiler(cfg, w, sched, ramulator)
    tb = comp.compile()
    base = os.path.join(out_dir, cfg.name)
    tb.write_trace(base + ".trace")
    tb.write_mem(base + ".mem")
    # bit-exact validation
    g = Golden(base + ".trace", base + ".mem")
    g.run()
    gold = g.dump("out")
    ref = N.f16_bytes(reference_forward(cfg, w))
    assert gold == ref, f"{cfg.name}: golden != reference"
    r = subprocess.run([sim, base + ".trace", base + ".mem", "--ramulator",
                        ramulator, "--out-dir", out_dir, "--json"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError("sim failed")
    with open(os.path.join(out_dir, "out.bin"), "rb") as f:
        assert f.read() == gold, f"{cfg.name}: sim != golden"
    stats = json.loads(r.stdout.strip().splitlines()[-1])
    return w, N.bytes_to_f16(gold), stats


def argmax_tokens(out16, S, dm):
    """Greedy next-token per position via a trivial LM head (argmax over the
    d_model activation — a stand-in head; the acceptance logic is identical
    for a real vocab projection)."""
    toks = []
    for i in range(S):
        row = out16[i * dm:(i + 1) * dm]
        best, bj = -1e30, 0
        for j, b in enumerate(row):
            v = N.f16_bits_to_f32(b)
            if v > best:
                best, bj = v, j
        toks.append(bj)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="spec_draft")
    ap.add_argument("--target", default="spec_target")
    ap.add_argument("--gamma", type=int, default=8)
    ap.add_argument("--out", default="build/edgc_out")
    ap.add_argument("--arch", default="")
    ap.add_argument("--ramulator", default="")
    ap.add_argument("--sched", default="W2")
    ap.add_argument("--sim", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    arch = load_arch(args.arch)
    if not args.ramulator:
        args.ramulator = resolve_ramulator(arch)
    # schedule from the arch YAML's DSE list (default W2)
    sd = next((d for d in arch.dse if d.get("label") == args.sched), None) or {}
    sched = Sched(label=args.sched, tile_link=sd.get("tile_link", 2),
                  wr_ports=sd.get("wr_ports", 2), rd_ports=sd.get("rd_ports", 4),
                  dma_rate=sd.get("dma_rate", 2),
                  dram_density=sd.get("dram_density", 2048),
                  nmc=sd.get("nmc", False))

    draft_cfg = MZOO.MODELS[args.draft]()
    target_cfg = MZOO.MODELS[args.target]()

    dw, dout, dstats = compile_run(draft_cfg, sched, args.ramulator, args.sim, args.out)
    tw, tout, tstats = compile_run(target_cfg, sched, args.ramulator, args.sim, args.out)

    dm = target_cfg.d_model
    S = target_cfg.seq_len
    draft_tokens = argmax_tokens(dout, S, dm)
    target_tokens = argmax_tokens(tout, S, dm)

    # Acceptance: longest prefix (over the gamma proposed positions) where the
    # target's argmax agrees with the draft's proposal. (Bit-exact greedy
    # acceptance; a sampling scheme would use the probability ratio.)
    gamma = min(args.gamma, S - 1)
    accepted = 0
    for i in range(gamma):
        if draft_tokens[i] == target_tokens[i]:
            accepted += 1
        else:
            break
    accepted_plus_one = accepted + 1  # target always contributes one token

    draft_c = dstats["total_cycles"]
    target_c = tstats["total_cycles"]
    spec_cost = draft_c + target_c   # draft proposal + one parallel verify

    print("═══ Speculative decoding (cycle-accurate cost model) ═══")
    print(f"draft  ({args.draft}, {draft_cfg.n_layers}L): "
          f"{draft_c} cycles, bit-exact ✓")
    print(f"target ({args.target}, {target_cfg.n_layers}L): {target_c} cycles "
          f"— ONE parallel pass verifies all {gamma} proposed positions, bit-exact ✓")
    print(f"measured acceptance for this (untrained) draft/target pair: "
          f"{accepted}/{gamma}")
    print()
    print(f"Per accepted step: draft + target = {spec_cost} cycles, yielding "
          f"(alpha+1) tokens.")
    print(f"Sequential target decode = {target_c} cycles/token.")
    print("Cycles/token vs accepted length alpha (lower is better):")
    print(f"  {'alpha':>5}  {'tokens':>6}  {'cyc/token':>10}  {'speedup':>8}")
    best_break = None
    for a in range(0, gamma + 1):
        toks = a + 1
        cpt = spec_cost / toks
        sp = target_c / cpt
        if sp >= 1.0 and best_break is None:
            best_break = a
        mark = "  <- break-even" if a == best_break else ""
        print(f"  {a:>5}  {toks:>6}  {cpt:>10.0f}  {sp:>7.2f}x{mark}")
    print()
    if best_break is not None:
        print(f"→ Speculative decoding wins once the draft's accepted length "
              f">= {best_break} of {gamma}. A distilled edge draft typically "
              f"accepts 4-6/8, i.e. {target_c / (spec_cost/6):.1f}x throughput "
              f"at alpha=5. The architecture's parallel verify (target processes "
              f"all {gamma} positions in one {target_c}-cycle pass) is what makes "
              f"this possible; the draft cost is only "
              f"{100*draft_c/spec_cost:.0f}% of the step.")


if __name__ == "__main__":
    main()
