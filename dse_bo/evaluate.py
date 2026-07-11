#!/usr/bin/env python3
"""evaluate(config) for the five-knob co-design DSE.

config = dict(num_tiles, tiles_per_group, shared_mb, topology, W, dram_density)

Pipeline per point (§Phase 1.2):
  1. gen_variant: cached per-structure CMake build (address widths, fabric).
  2. validate_variant.sh: ctest + spec_target golden==reference==sim gate.
     A variant that fails is INVALID — its cycle numbers never reach BO.
  3. Workloads (run in parallel):
       - Llama-3.2-1B  decode batch=1  ctx=512   (Tier-1, f16)
       - Llama-3.2-1B  decode batch=16 ctx=512   (Tier-1, f16)
       - Mistral-7B-INT4 decode batch=16 ctx=512 (Tier-1, weight-stream axis)
       - spec_target (GQA 4-layer) prefill, bit-exact anchor + cycles
     KV placement is 'auto': resident in the buffer-die banks iff the whole
     KV cache fits num_banks * shared_mb (steady-state residency model).
  4. Metrics -> 4 objectives:
       decode_tput   (max)  geomean tok/s of the three decode runs
       mxu_util      (max)  mean decode MXU occupancy (%)
       arith_intensity (max) geomean effective FLOP / DRAM byte
       area_mm2      (min)  first-order silicon cost (see area_model)

Results cached in dse_bo/cache/ keyed by the full config; a cache hit does
not re-run anything (structural rebuilds are expensive).
"""
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
CACHE = os.path.join(HERE, "cache")
EDGC = os.path.join(ROOT, "tools", "edgc")

sys.path.insert(0, HERE)
from gen_variant import build_variant, check_feasible, variant_name  # noqa: E402

KNOBS = dict(num_tiles=(8, 16, 32), tiles_per_group=(2, 4, 8),
             shared_mb=(4, 8, 16), topology=("ring", "mesh", "torus"),
             W=(1, 2, 4), dram_density=(512, 1024, 2048, 4096, 8192))

DECODE_RUNS = [  # (tag, model, batch)
    ("llama_b1",  "Llama-3.2-1B",   1),
    ("llama_b16", "Llama-3.2-1B",   16),
    ("mistral_b16", "Mistral-7B-INT4", 16),
]
CTX = 512


# ── First-order area model ────────────────────────────────────
# Calibrated on the OpenROAD 3nm P&R numbers in pr/out/arch_model.json
# (baseline 16 tiles / 4 banks x 8 MB):
#   base die  : (macro 2 226 510 + std 14 926) um2 / 16 tiles
#   buffer die: macro 4 453 020 um2 / 32 MB, bank ctrl std 1 191 um2 / 4
#   HB keep-out: 156 800 um2 at (16 tile-links @W=1, 4 columns @2048b),
#     split half tile-link half DRAM-column, each scaling linearly in
#     provisioned wire count.
A_TILE_UM2 = (2226510.1 + 14926.2) / 16
A_SRAM_UM2_PER_MB = 4453020.2 / 32
A_BANKCTRL_UM2 = 1190.8 / 4
A_KOZ_TILE_LINKS = 156800.0 / 2
A_KOZ_DRAM_COLS = 156800.0 / 2
A_GRID_ROUTER_UM2 = 933.0   # extra 4-port router std area per tile (vs ring)


def area_model(cfg):
    nt, tpg, smb = cfg["num_tiles"], cfg["tiles_per_group"], cfg["shared_mb"]
    nb = nt // tpg
    a = nt * A_TILE_UM2
    a += nb * smb * A_SRAM_UM2_PER_MB + nb * A_BANKCTRL_UM2
    a += A_KOZ_TILE_LINKS * (nt * cfg["W"]) / 16.0
    a += A_KOZ_DRAM_COLS * (nb * cfg["dram_density"]) / (4 * 2048.0)
    if cfg["topology"] != "ring":
        a += nt * A_GRID_ROUTER_UM2
    return a / 1e6  # mm^2


def config_key(cfg, ctx=CTX):
    d = {k: cfg[k] for k in sorted(KNOBS)}
    if ctx != CTX:
        d["ctx"] = ctx   # suffix only for non-default ctx (cache back-compat)
    s = json.dumps(d, sort_keys=True)
    return hashlib.sha1(s.encode()).hexdigest()[:16]


def geomean(xs):
    xs = [max(x, 1e-12) for x in xs]
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def _validate(vdir):
    """Run the §3.3 gate once per variant; cache PASS/INVALID on disk."""
    import fcntl
    marker = os.path.join(vdir, ".validated")
    if os.path.exists(marker):
        return open(marker).read().strip() == "PASS"
    with open(os.path.join(vdir, ".vlock"), "w") as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        if os.path.exists(marker):
            return open(marker).read().strip() == "PASS"
        r = subprocess.run([os.path.join(HERE, "validate_variant.sh"), vdir],
                           capture_output=True, text=True)
        verdict = (r.stdout.strip().splitlines()[-1]
                   if r.stdout.strip() else "INVALID")
        open(marker, "w").write(verdict + "\n")
    return verdict == "PASS"


def evaluate(cfg, jobs=16, log=print, ctx=CTX):
    """Returns a dict with feasible/valid flags, metrics and objectives."""
    os.makedirs(CACHE, exist_ok=True)
    cpath = os.path.join(CACHE, config_key(cfg, ctx) + ".json")
    if os.path.exists(cpath):
        return json.load(open(cpath))

    res = {"config": {k: cfg[k] for k in KNOBS}, "ctx": ctx, "feasible": True,
           "valid": False}

    def finish():
        json.dump(res, open(cpath, "w"), indent=1)
        return res

    err = check_feasible(cfg["num_tiles"], cfg["tiles_per_group"],
                         cfg["shared_mb"], cfg["topology"])
    if err:
        res.update(feasible=False, reason=err)
        return finish()

    log(f"[evaluate] building {variant_name(cfg['num_tiles'], cfg['tiles_per_group'], cfg['shared_mb'], cfg['topology'])} ...")
    vdir, berr = build_variant(cfg["num_tiles"], cfg["tiles_per_group"],
                               cfg["shared_mb"], cfg["topology"], jobs=jobs)
    if berr:
        res.update(feasible=False, reason=berr)
        return finish()

    if not _validate(vdir):
        res.update(reason="validation gate INVALID (see validate.log)")
        return finish()

    yaml_path = os.path.join(vdir, "mobol_arch.yaml")
    sim = os.path.join(vdir, "build", "src", "trace_sim")
    env = dict(os.environ, MOBOL_ARCH_YAML=yaml_path)
    tag = config_key(cfg, ctx)
    outroot = os.path.join(vdir, "runs", tag)
    os.makedirs(outroot, exist_ok=True)

    # ── launch the four workloads in parallel ──
    procs = {}
    for rt, model, batch in DECODE_RUNS:
        od = os.path.join(outroot, rt)
        cmd = ["python3", "decode_tier1.py", "--sim", sim, "--arch", yaml_path,
               "--models", model, "--batch", str(batch), "--ctx", str(ctx),
               "--W", str(cfg["W"]), "--dram-density", str(cfg["dram_density"]),
               "--kv", "auto", "--out", od, "--json",
               os.path.join(od, "result.json")]
        procs[rt] = subprocess.Popen(cmd, cwd=EDGC, env=env,
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, text=True)
    sp_out = os.path.join(outroot, "spec_target")
    procs["spec"] = subprocess.Popen(
        ["python3", "edgc_cc.py", "--model", "spec_target", "--out", sp_out,
         "--check", "--sim", sim, "--W", str(cfg["W"]),
         "--dram-density", str(cfg["dram_density"]), "--json"],
        cwd=EDGC, env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)

    outputs = {k: p.communicate()[0] for k, p in procs.items()}
    fails = [k for k, p in procs.items() if p.returncode != 0]
    if fails:
        res.update(reason=f"workload failed: {fails}",
                   logs={k: outputs[k][-2000:] for k in fails})
        return finish()

    # ── collect metrics ──
    m = {}
    for rt, model, batch in DECODE_RUNS:
        row = json.load(open(os.path.join(outroot, rt, "result.json")))[0]
        if not row["causality_ok"]:
            res.update(reason=f"causality violated in {rt}")
            return finish()
        m[rt] = row
    spec = json.loads(outputs["spec"][outputs["spec"].index("["):])[0]
    if spec.get("golden_vs_ref") != "PASS" or spec.get("golden_vs_sim") != "PASS":
        res.update(reason=f"spec_target bit-exact FAIL: {spec}")
        return finish()
    m["spec_prefill"] = spec

    obj = dict(
        decode_tput=geomean([m[rt]["tokens_per_s"] for rt, _, _ in DECODE_RUNS]),
        mxu_util=sum(m[rt]["mxu_util_pct"] for rt, _, _ in DECODE_RUNS) / 3,
        arith_intensity=geomean([m[rt]["arith_intensity"]
                                 for rt, _, _ in DECODE_RUNS]),
        area_mm2=area_model(cfg),
    )
    res.update(valid=True, metrics=m, objectives=obj,
               variant=os.path.basename(vdir))
    log(f"[evaluate] {res['variant']} W={cfg['W']} D={cfg['dram_density']}: "
        + " ".join(f"{k}={v:.3f}" for k, v in obj.items()))
    return finish()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    for k, vals in KNOBS.items():
        t = type(vals[0])
        ap.add_argument(f"--{k.replace('_', '-')}", type=t, required=True)
    a = ap.parse_args()
    cfg = {k: getattr(a, k) for k in KNOBS}
    r = evaluate(cfg)
    print(json.dumps(r, indent=1))
