#!/usr/bin/env python3
"""Workload-matrix evaluation for the five-knob DSE (operating-spectrum
upgrade).

12 anchors span memory-bound -> compute-bound (§design in ARCH_SPEC):
decode at batch 1/16/64 and ctx 512/2048/8192 across the five real model
structures (f16/i8/i4, MHA/GQA, dense/MoE-fragmented), chunked prefill
(M=256) as the compute-bound end, and a dense GEMM control. All anchors run
on the SAME validated variant builds as evaluate.py (§3.3 gate unchanged).

Per-anchor performance is normalized to the fixed reference architecture
(16 tiles / 4 per group / 8 MB / ring / W1 / D2048):
    speedup_w(cfg) = perf_w(cfg) / perf_w(baseline)
Aggregates:
    weighted  = exp(sum_i w_i * ln speedup_i)      (BO main objective)
    minmax    = min_i speedup_i                    (robust ranking)
    decode_agg / prefill_agg = geomean over the group's anchors
    area_mm2  = same first-order silicon model as evaluate.py
"""
import hashlib
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
EDGC = os.path.join(ROOT, "tools", "edgc")
MCACHE = os.path.join(HERE, "cache_matrix")

sys.path.insert(0, HERE)
from gen_variant import build_variant, check_feasible, variant_name  # noqa: E402
from evaluate import (KNOBS, area_model, geomean, _validate,  # noqa: E402
                      config_key)

CLOCK_HZ = 500e6
BASELINE = dict(num_tiles=16, tiles_per_group=4, shared_mb=8,
                topology="ring", W=1, dram_density=2048)

# ── The anchor set (id, spec, weight, group, expected regime) ──
ANCHORS = [
    dict(id="A", kind="decode", model="Llama-3.2-1B", batch=1, ctx=512,
         kv="auto", w=0.10, group="decode", regime="memory-bound extreme"),
    dict(id="B", kind="decode", model="Mistral-7B-INT4", batch=1, ctx=512,
         kv="auto", w=0.10, group="decode", regime="memory-bound extreme i4"),
    dict(id="C", kind="decode", model="Llama-3.2-1B", batch=16, ctx=512,
         kv="auto", w=0.10, group="decode", regime="M-filled, bw-limited"),
    dict(id="D", kind="decode", model="Mistral-7B-INT4", batch=16, ctx=512,
         kv="auto", w=0.10, group="decode", regime="balanced (spec-tree k=16)"),
    dict(id="E", kind="decode", model="Llama-3.2-1B", batch=64, ctx=512,
         kv="auto", w=0.05, group="decode", regime="balanced->compute (k=64)"),
    dict(id="F", kind="decode", model="Phi-3.5-mini", batch=16, ctx=2048,
         kv="auto", w=0.05, group="decode", regime="i8 + MHA full-KV"),
    dict(id="G", kind="decode", model="Llama-3.2-3B", batch=16, ctx=8192,
         kv="auto", w=0.05, group="decode", regime="L=8192 KV pressure"),
    dict(id="H", kind="decode", model="Mixtral-8x7B", batch=16, ctx=512,
         kv="auto", moe_frag=True, w=0.10, group="decode",
         regime="MoE M-fragmentation"),
    dict(id="I", kind="decode", model="Llama-3.2-1B", batch=256, ctx=2048,
         kv="dram", w=0.125, group="prefill",
         regime="chunked prefill, compute-bound"),
    dict(id="J", kind="decode", model="Mistral-7B-INT4", batch=256, ctx=2048,
         kv="dram", w=0.125, group="prefill",
         regime="chunked prefill i4 (far ridge)"),
    dict(id="K", kind="gemm", M=512, K=2048, N=2048, prec="f16",
         w=0.05, group="prefill", regime="dense GEMM control"),
    dict(id="L", kind="decode", model="Llama-3.2-3B", batch=16, ctx=512,
         kv="auto", w=0.05, group="decode", regime="mid dense, d_head=128"),
]
assert abs(sum(a["w"] for a in ANCHORS) - 1.0) < 1e-9


def _run_decode_anchor(vdir, cfg, a):
    yaml_path = os.path.join(vdir, "mobol_arch.yaml")
    sim = os.path.join(vdir, "build", "src", "trace_sim")
    od = os.path.join(vdir, "runs", "mx_" + a["id"] + "_" +
                      config_key(cfg))
    env = dict(os.environ, MOBOL_ARCH_YAML=yaml_path)
    cmd = ["python3", "decode_tier1.py", "--sim", sim, "--arch", yaml_path,
           "--models", a["model"], "--batch", str(a["batch"]),
           "--ctx", str(a["ctx"]), "--W", str(cfg["W"]),
           "--dram-density", str(cfg["dram_density"]), "--kv", a["kv"],
           "--out", od, "--json", os.path.join(od, "result.json")]
    if a.get("moe_frag"):
        cmd.append("--moe-frag")
    r = subprocess.run(cmd, cwd=EDGC, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"anchor {a['id']}: {r.stdout[-1200:]}{r.stderr[-800:]}")
    row = json.load(open(os.path.join(od, "result.json")))[0]
    if not row["causality_ok"]:
        raise RuntimeError(f"anchor {a['id']}: causality violated")
    return dict(perf=row["tokens_per_s"], mxu_util=row["mxu_util_pct"],
                ai=row["arith_intensity"], cyc=row["cyc_total"],
                dram_gb=row["dram_read_gb"], kv_resident=row.get("kv_resident"))


def _run_gemm_anchor(vdir, cfg, a):
    yaml_path = os.path.join(vdir, "mobol_arch.yaml")
    sim = os.path.join(vdir, "build", "src", "trace_sim")
    od = os.path.join(vdir, "runs", "mx_" + a["id"] + "_" + config_key(cfg))
    env = dict(os.environ, MOBOL_ARCH_YAML=yaml_path)
    ram = os.path.join(ROOT, "config/ramulator_3d_dram.yaml")
    W = cfg["W"]
    code = (
        "import sys, json; sys.path.insert(0, '.')\n"
        "import bench_gemm as bg\n"
        "from edgc import Sched\n"
        f"sched = Sched(label='W{W}', tile_link={W}, wr_ports={W}, "
        f"rd_ports={2*W}, dma_rate={W}, dram_density={cfg['dram_density']})\n"
        f"base = bg.build_gemm({a['M']}, {a['K']}, {a['N']}, {od!r}, sched, "
        f"{ram!r}, dbuf=True, prec={a['prec']!r})\n"
        f"s = bg.run({sim!r}, base, {ram!r}, {od!r})\n"
        "print(json.dumps(s))\n")
    r = subprocess.run(["python3", "-c", code], cwd=EDGC, env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"anchor {a['id']}: {r.stdout[-800:]}{r.stderr[-800:]}")
    s = json.loads(r.stdout.strip().splitlines()[-1])
    if not s["causality_ok"]:
        raise RuntimeError(f"anchor {a['id']}: causality violated")
    n_tiles = len(s["tiles"])
    util = 100.0 * sum(t["mxu_busy"] for t in s["tiles"]) / (
        n_tiles * s["total_cycles"])
    flops = 2 * a["M"] * a["K"] * a["N"]
    return dict(perf=CLOCK_HZ / s["total_cycles"], mxu_util=util,
                ai=flops / max(1, s["dram_read_bytes"]),
                cyc=s["total_cycles"],
                dram_gb=s["dram_read_bytes"] / 1e9, kv_resident=None)


def run_anchor(cfg, a, jobs=16):
    """Cached per-(config, anchor) raw metrics. Raises on failure."""
    os.makedirs(MCACHE, exist_ok=True)
    key = config_key(cfg) + "_" + a["id"]
    cpath = os.path.join(MCACHE, key + ".json")
    if os.path.exists(cpath):
        return json.load(open(cpath))
    # fast path: anchors A/C/D at ctx512 kv=auto equal the old evaluate.py
    # runs (llama_b1 / llama_b16 / mistral_b16) — reuse that cache.
    old_tag = {"A": "llama_b1", "C": "llama_b16", "D": "mistral_b16"}.get(a["id"])
    if old_tag:
        opath = os.path.join(HERE, "cache", config_key(cfg) + ".json")
        if os.path.exists(opath):
            od = json.load(open(opath))
            if od.get("valid"):
                row = od["metrics"][old_tag]
                out = dict(perf=row["tokens_per_s"],
                           mxu_util=row["mxu_util_pct"],
                           ai=row["arith_intensity"], cyc=row["cyc_total"],
                           dram_gb=row["dram_read_gb"],
                           kv_resident=row.get("kv_resident"))
                json.dump(out, open(cpath, "w"))
                return out

    vdir, err = build_variant(cfg["num_tiles"], cfg["tiles_per_group"],
                              cfg["shared_mb"], cfg["topology"], jobs=jobs)
    if err:
        raise RuntimeError(err)
    if not _validate(vdir):
        raise RuntimeError("validation gate INVALID")
    out = (_run_gemm_anchor if a["kind"] == "gemm"
           else _run_decode_anchor)(vdir, cfg, a)
    json.dump(out, open(cpath, "w"))
    return out


_BASE_PERF = {}


def baseline_perf():
    """Anchor perfs of the fixed reference architecture (computed once)."""
    if not _BASE_PERF:
        with ThreadPoolExecutor(max_workers=len(ANCHORS)) as ex:
            res = list(ex.map(lambda a: run_anchor(BASELINE, a), ANCHORS))
        for a, r in zip(ANCHORS, res):
            _BASE_PERF[a["id"]] = r["perf"]
    return _BASE_PERF


def evaluate_matrix(cfg, log=print):
    """Full-matrix evaluation -> per-anchor speedups + aggregate objectives.
    Cached; INVALID/failed points return valid=False."""
    os.makedirs(MCACHE, exist_ok=True)
    cpath = os.path.join(MCACHE, "agg_" + config_key(cfg) + ".json")
    if os.path.exists(cpath):
        return json.load(open(cpath))
    res = {"config": {k: cfg[k] for k in KNOBS}, "valid": False}

    def finish():
        json.dump(res, open(cpath, "w"), indent=1)
        return res

    err = check_feasible(cfg["num_tiles"], cfg["tiles_per_group"],
                         cfg["shared_mb"], cfg["topology"])
    if err:
        res["reason"] = err
        return finish()
    base = baseline_perf()
    try:
        with ThreadPoolExecutor(max_workers=len(ANCHORS)) as ex:
            raw = list(ex.map(lambda a: run_anchor(cfg, a), ANCHORS))
    except Exception as e:
        res["reason"] = str(e)[:500]
        return finish()

    anchors = {}
    for a, r in zip(ANCHORS, raw):
        anchors[a["id"]] = dict(r, speedup=r["perf"] / base[a["id"]],
                                weight=a["w"], group=a["group"])
    sp = {aid: v["speedup"] for aid, v in anchors.items()}
    weighted = math.exp(sum(a["w"] * math.log(max(sp[a["id"]], 1e-9))
                            for a in ANCHORS))
    res.update(
        valid=True, anchors=anchors,
        objectives=dict(
            weighted=weighted,
            minmax=min(sp.values()),
            decode_agg=geomean([sp[a["id"]] for a in ANCHORS
                                if a["group"] == "decode"]),
            prefill_agg=geomean([sp[a["id"]] for a in ANCHORS
                                 if a["group"] == "prefill"]),
            area_mm2=area_model(cfg)),
        variant=variant_name(cfg["num_tiles"], cfg["tiles_per_group"],
                             cfg["shared_mb"], cfg["topology"]))
    o = res["objectives"]
    log(f"[matrix] {res['variant']} W={cfg['W']} D={cfg['dram_density']}: "
        f"weighted={o['weighted']:.3f} minmax={o['minmax']:.3f} "
        f"dec={o['decode_agg']:.3f} pre={o['prefill_agg']:.3f} "
        f"area={o['area_mm2']:.2f}")
    return finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    for k in KNOBS:
        t = type(KNOBS[k][0])
        ap.add_argument(f"--{k.replace('_', '-')}", type=t, required=True)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in KNOBS}
    print(json.dumps(evaluate_matrix(cfg), indent=1))
