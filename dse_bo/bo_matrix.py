#!/usr/bin/env python3
"""BO campaigns on the workload-matrix objective.

--mode bo     single-objective qLogNEI on `weighted` (方案1, BO main)
--mode random same-budget uniform control (§0.2 proof, shares the cache)
--mode moo    qNEHVI on {decode_agg, prefill_agg, area_mm2} (方案3 Pareto)

Same Ax GenerationStrategy machinery + GP/acquisition evidence logging as
bo_engine.py; per-iteration records go to <out>/log.jsonl.

  python3 bo_matrix.py --mode bo --budget 40 --seed 0
"""
import argparse
import json
import logging
import os
import random
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from matrix import evaluate_matrix, ANCHORS  # noqa: E402
from evaluate import KNOBS  # noqa: E402
from bo_engine import ax_parameters, gp_evidence  # noqa: E402

# Pessimal values for INVALID points (well below any observed valid config;
# keeps them out of optima/Pareto by construction).
PENALTY = dict(weighted=0.3, minmax=0.1, decode_agg=0.3, prefill_agg=0.3,
               area_mm2=20.0)
MOO = {  # objective -> (minimize, hypervolume/EHVI reference point)
    "decode_agg": (False, 0.7),
    "prefill_agg": (False, 0.7),
    "area_mm2": (True, 20.0),
}


def obj_vector(res):
    if res.get("valid"):
        return {k: float(res["objectives"][k]) for k in PENALTY}
    return dict(PENALTY)


def hv_moo(points):
    import torch
    from botorch.utils.multi_objective.hypervolume import Hypervolume
    from botorch.utils.multi_objective.pareto import is_non_dominated
    if not points:
        return 0.0
    Y = torch.tensor([[(-p[k] if MOO[k][0] else p[k]) for k in MOO]
                      for p in points], dtype=torch.double)
    ref = torch.tensor([(-v[1] if v[0] else v[1]) for v in MOO.values()],
                       dtype=torch.double)
    Y = Y[(Y > ref).all(dim=-1)]
    if Y.shape[0] == 0:
        return 0.0
    return Hypervolume(ref_point=ref).compute(Y[is_non_dominated(Y)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("bo", "random", "moo"), required=True)
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--sobol", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    outdir = a.out or os.path.join(HERE, "results",
                                   f"mx_{a.mode}_seed{a.seed}")
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "log.jsonl"), "w")

    if a.mode == "random":
        rng = random.Random(a.seed)
        gen = lambda: ({k: rng.choice(list(v)) for k, v in KNOBS.items()}, None)
        ax_client = None
    else:
        from ax.service.ax_client import AxClient, ObjectiveProperties
        from ax.generation_strategy.generation_strategy import (
            GenerationStrategy, GenerationStep)
        from ax.adapter.registry import Generators
        gs = GenerationStrategy(steps=[
            GenerationStep(generator=Generators.SOBOL, num_trials=a.sobol,
                           model_kwargs={"seed": a.seed},
                           should_deduplicate=True),
            GenerationStep(generator=Generators.BOTORCH_MODULAR,
                           num_trials=-1, should_deduplicate=True),
        ])
        ax_client = AxClient(generation_strategy=gs, verbose_logging=False,
                             random_seed=a.seed)
        if a.mode == "bo":
            objectives = {"weighted": ObjectiveProperties(minimize=False)}
        else:
            objectives = {k: ObjectiveProperties(minimize=v[0], threshold=v[1])
                          for k, v in MOO.items()}
        ax_client.create_experiment(name=f"mobol_matrix_{a.mode}_{a.seed}",
                                    parameters=ax_parameters(),
                                    objectives=objectives)

    observed, best = [], 0.0
    for it in range(a.budget):
        t0 = time.time()
        if ax_client is None:
            params, idx = gen()
            method = "random"
        else:
            params, idx = ax_client.get_next_trial()
            gr = ax_client.experiment.trials[idx].generator_runs[0]
            method = gr._model_key or "?"

        res = evaluate_matrix(dict(params))
        obj = obj_vector(res)
        if ax_client is not None:
            raw = ({"weighted": (obj["weighted"], 0.0)} if a.mode == "bo"
                   else {k: (obj[k], 0.0) for k in MOO})
            ax_client.complete_trial(trial_index=idx, raw_data=raw)
        observed.append(obj)
        best = max(best, obj["weighted"])
        hv = hv_moo(observed) if a.mode == "moo" else None

        rec = dict(iter=it, method=method, params=params,
                   valid=bool(res.get("valid")), reason=res.get("reason"),
                   objectives=obj, best_weighted=best, hypervolume=hv,
                   wall_s=round(time.time() - t0, 1))
        if res.get("valid"):
            rec["speedups"] = {k: round(v["speedup"], 4)
                               for k, v in res["anchors"].items()}
        if ax_client is not None:
            gm = gr.gen_metadata or {}
            if gm:
                rec["gen_metadata"] = {k: str(v)[:120] for k, v in gm.items()}
            if method not in ("Sobol", "random"):
                ev = gp_evidence(ax_client)
                if ev:
                    rec["gp"] = ev
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"[{it:3d}] {method:8s} "
              f"{params['num_tiles']:2d}t/{params['tiles_per_group']}g/"
              f"{params['shared_mb']:2d}MB/{params['topology']:5s}/"
              f"W{params['W']}/D{params['dram_density']:4d} "
              f"weighted={obj['weighted']:.3f} minmax={obj['minmax']:.3f} "
              f"dec={obj['decode_agg']:.3f} pre={obj['prefill_agg']:.3f} "
              f"best={best:.3f}" + (f" HV={hv:.3f}" if hv else "")
              + f" ({rec['wall_s']}s)", flush=True)

    json.dump(dict(mode=a.mode, budget=a.budget, seed=a.seed,
                   best_weighted=best,
                   final_hv=hv_moo(observed) if a.mode == "moo" else None),
              open(os.path.join(outdir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
