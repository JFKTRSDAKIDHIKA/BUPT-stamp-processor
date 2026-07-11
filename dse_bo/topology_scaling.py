#!/usr/bin/env python3
"""Phase 3: ring vs mesh vs torus at 8 / 16 / 32 tiles.

Fixed (tiles_per_group=4, shared_mb=8, W=1|4, density=2048); sweeps the
9 (num_tiles x topology) structural points, runs the three §4 workloads on
each, and collects the NoC-side metrics that decide whether topology is a
bottleneck at all:
    avg / max link utilization, flit-hops, eject-blocked cycles,
    end-to-end cycles, NoC diameter.

Writes dse_bo/results/topology_scaling.csv.
W=1 is the deployment point; W=4 (wide consumption ports) removes the
memory wall as far as possible to give the NoC its best chance to matter.
"""
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EDGC = os.path.join(ROOT, "tools", "edgc")
sys.path.insert(0, HERE)
from gen_variant import build_variant  # noqa: E402
from evaluate import _validate, DECODE_RUNS, CTX  # noqa: E402


def noc_diameter(nt, topo):
    if topo == "ring":
        return nt // 2
    w = {8: 4, 16: 4, 32: 8}[nt]   # grid shape (types.h NOC_GRID_W)
    h = nt // w
    if topo == "mesh":
        return (w - 1) + (h - 1)
    return w // 2 + h // 2


def run_decode(vdir, model, batch, W, out_tag):
    yaml_path = os.path.join(vdir, "mobol_arch.yaml")
    sim = os.path.join(vdir, "build", "src", "trace_sim")
    od = os.path.join(vdir, "runs", out_tag)
    env = dict(os.environ, MOBOL_ARCH_YAML=yaml_path)
    r = subprocess.run(
        ["python3", "decode_tier1.py", "--sim", sim, "--arch", yaml_path,
         "--models", model, "--batch", str(batch), "--ctx", str(CTX),
         "--W", str(W), "--dram-density", "2048", "--kv", "auto",
         "--out", od, "--json", os.path.join(od, "result.json")],
        cwd=EDGC, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"decode failed: {r.stdout[-800:]}{r.stderr[-800:]}")
    row = json.load(open(os.path.join(od, "result.json")))[0]
    # NoC metrics: aggregate over the per-shape sim JSONs of this run by
    # re-simulating? decode_tier1 already ran each unique shape; pull the
    # heaviest shape's NoC stats via a directed rerun of the largest GEMM.
    return row


def run_shape_noc(vdir, M, K, N, prec, W, tag):
    """Run one GEMM shape and return the trace_sim JSON (NoC metrics)."""
    yaml_path = os.path.join(vdir, "mobol_arch.yaml")
    sim = os.path.join(vdir, "build", "src", "trace_sim")
    od = os.path.join(vdir, "runs", tag)
    env = dict(os.environ, MOBOL_ARCH_YAML=yaml_path)
    code = (
        "import sys, json; sys.path.insert(0, '.');\n"
        "import bench_gemm as bg\n"
        "from edgc import Sched\n"
        f"sched = Sched(label='W{W}', tile_link={W}, wr_ports={W}, "
        f"rd_ports={2*W}, dma_rate={W}, dram_density=2048)\n"
        f"base = bg.build_gemm({M}, {K}, {N}, {od!r}, sched, "
        f"{os.path.join(ROOT, 'config/ramulator_3d_dram.yaml')!r}, "
        f"dbuf=True, prec={prec!r})\n"
        f"s = bg.run({sim!r}, base, "
        f"{os.path.join(ROOT, 'config/ramulator_3d_dram.yaml')!r}, {od!r})\n"
        "print(json.dumps(s))\n")
    r = subprocess.run(["python3", "-c", code], cwd=EDGC, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"shape sim failed: {r.stdout[-500:]}{r.stderr[-500:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main():
    rows = []
    for nt in (8, 16, 32):
        for topo in ("ring", "mesh", "torus"):
            print(f"=== {nt} tiles, {topo} ===", flush=True)
            vdir, err = build_variant(nt, 4, 8, topo, jobs=16)
            if err:
                print("  build failed:", err)
                continue
            if not _validate(vdir):
                print("  INVALID (validation gate)")
                continue
            for W in (1, 4):
                # end-to-end decode on Llama f16 (the §4 bandwidth-heavy one)
                d = run_decode(vdir, "Llama-3.2-1B", 16, W, f"topo_W{W}")
                # NoC stats on the largest weight GEMM (heaviest traffic) and
                # the attention GEMM (cross-tile barrier + KV traffic).
                big = run_shape_noc(vdir, 16, 2048, 8192, "f16", W,
                                    f"topo_noc_big_W{W}")
                att = run_shape_noc(vdir, 16, 64, 512, "f16", W,
                                    f"topo_noc_att_W{W}")
                rows.append(dict(
                    num_tiles=nt, topology=topo, W=W,
                    diameter=noc_diameter(nt, topo),
                    llama_b16_cyc=d["cyc_total"],
                    llama_b16_toks=d["tokens_per_s"],
                    big_gemm_cyc=big["total_cycles"],
                    big_noc_avg_util=big.get("noc_avg_link_util", -1),
                    big_noc_max_util=big.get("noc_max_link_util", -1),
                    big_noc_hops=big["noc_flit_hops"],
                    big_eject_blocked=big.get("noc_eject_blocked", -1),
                    att_gemm_cyc=att["total_cycles"],
                    att_noc_avg_util=att.get("noc_avg_link_util", -1),
                    att_noc_max_util=att.get("noc_max_link_util", -1),
                    att_noc_hops=att["noc_flit_hops"],
                ))
                print("  " + json.dumps(rows[-1]), flush=True)
    out = os.path.join(HERE, "results", "topology_scaling.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
