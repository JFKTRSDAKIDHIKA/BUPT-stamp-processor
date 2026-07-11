#!/usr/bin/env python3
"""Same-budget uniform random search over the identical five-knob space —
the control arm proving the BO loop beats chance (§0.2).

Shares evaluate()'s on-disk cache with bo_engine.py (identical configs are
simulated once), but every draw still counts as one evaluation for the
convergence curve, exactly like BO trials do.

  python3 random_baseline.py --budget 60 --seed 100 --out results/rand_seed100
"""
import argparse
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import evaluate, KNOBS  # noqa: E402
from bo_engine import OBJECTIVES, objective_vector, hv_of  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    outdir = a.out or os.path.join(HERE, "results", f"rand_seed{a.seed}")
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "log.jsonl"), "w")

    rng = random.Random(a.seed)
    observed = []
    for it in range(a.budget):
        t0 = time.time()
        params = {k: rng.choice(list(v)) for k, v in KNOBS.items()}
        res = evaluate(params)
        obj = objective_vector(res)
        observed.append(obj)
        hv = hv_of(observed)
        rec = dict(iter=it, method="random", params=params,
                   valid=bool(res.get("valid")), reason=res.get("reason"),
                   objectives=obj, hypervolume=hv,
                   wall_s=round(time.time() - t0, 1))
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"[{it:3d}] random   {params['num_tiles']:2d}t/{params['tiles_per_group']}g/"
              f"{params['shared_mb']:2d}MB/{params['topology']:5s}/W{params['W']}/D{params['dram_density']:4d} "
              f"tput={obj['decode_tput']:6.1f} area={obj['area_mm2']:5.2f} "
              f"HV={hv:9.1f} ({rec['wall_s']}s)", flush=True)

    json.dump(dict(budget=a.budget, seed=a.seed, final_hv=hv_of(observed)),
              open(os.path.join(outdir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
