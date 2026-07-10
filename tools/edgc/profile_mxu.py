#!/usr/bin/env python3
"""MXU utilization profiler for the base-die compute-tile array.

Compiles each edge-LLM workload with edgc (baseline schedule from the arch
YAML), runs the cycle-accurate simulator, and reports chip-wide and per-tile
MXU occupancy plus the stall attribution that explains the gap to 100%.

  python3 profile_mxu.py --sim ../../build/src/trace_sim [--models a,b,c]
"""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edgc import gen_weights, Compiler, Sched
from edgc.arch import load_arch, resolve_ramulator
import models as MZOO

MXU_LAT = 4  # mxu_latency (cycles a single op occupies the pipe)


def compile_model(name, out_dir, sched, ramulator):
    cfg = MZOO.MODELS[name]()
    w = gen_weights(cfg)
    comp = Compiler(cfg, w, sched, ramulator)
    tb = comp.compile()
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, cfg.name)
    tb.write_trace(base + ".trace")
    tb.write_mem(base + ".mem")
    return cfg, base


def run_sim(sim, base, ramulator, out_dir):
    r = subprocess.run([sim, base + ".trace", base + ".mem",
                        "--ramulator", ramulator, "--out-dir", out_dir,
                        "--json"], capture_output=True, text=True)
    if r.returncode not in (0, 3):
        sys.stderr.write(r.stdout + r.stderr)
        raise RuntimeError(f"sim failed ({r.returncode}) for {base}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", default="../../build/src/trace_sim")
    ap.add_argument("--out", default="/tmp/edgc_prof")
    ap.add_argument("--arch", default="")
    ap.add_argument("--models", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    arch = load_arch(args.arch)
    ramulator = resolve_ramulator(arch)
    d = next(x for x in arch.dse if x["label"] == arch.default_sched)
    sched = Sched(label=d["label"], tile_link=d.get("tile_link", 1),
                  wr_ports=d.get("wr_ports", 1), rd_ports=d.get("rd_ports", 2),
                  dma_rate=d.get("dma_rate", 1),
                  dram_density=d.get("dram_density", 2048),
                  nmc=d.get("nmc", False))

    models = (args.models.split(",") if args.models else
              [m for m in MZOO.MODELS if m != "spec_draft"])

    rows = []
    for name in models:
        cfg, base = compile_model(name, args.out, sched, ramulator)
        s = run_sim(args.sim, base, ramulator, args.out)
        tiles = s["tiles"]
        active = [t for t in tiles if t["instrs"] > 0]
        cyc = s["total_cycles"]
        # Per-tile MXU occupancy over the whole run.
        per_tile_occ = [100.0 * t["mxu_busy"] / cyc for t in tiles]
        busy_tiles = [t for t in tiles if t["mxu_busy"] > 0]
        # Ideal cycles if the MXU pipe never stalled: total MXU ops * latency,
        # spread over the tiles that actually run MXU work.
        row = {
            "model": name,
            "cycles": cyc,
            "instrs": s["instrs"],
            "mxu_ops": s["mxu_ops"],
            "chip_mxu_util_pct": 100.0 * s["mxu_util"],
            "active_tiles": len(active),
            "mxu_tiles": len(busy_tiles),
            "mxu_util_over_active_pct": (100.0 * s["mxu_util"] * 16 / len(active)
                                         if active else 0.0),
            "per_tile_occ_min": min(per_tile_occ),
            "per_tile_occ_max": max(per_tile_occ),
            "noc_max_link_util_pct": 100.0 * s["noc_max_link_util"],
            "dram_rd_kb": s["dram_read_bytes"] / 1024.0,
            "dram_avg_lat": s["dram_avg_read_latency"],
        }
        # Aggregate stall attribution across active tiles (cycle-sums).
        for k in ("busy", "stall_acquire", "stall_dma", "stall_mxu",
                  "stall_vpu", "stall_inject", "mxu_busy", "vpu_busy"):
            row["sum_" + k] = sum(t[k] for t in tiles)
        rows.append(row)

    print(json.dumps(rows, indent=2))
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
