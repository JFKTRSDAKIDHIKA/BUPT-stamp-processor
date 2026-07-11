#!/usr/bin/env python3
"""Multi-objective Bayesian optimization over the five-knob design space.

Library: Ax 1.2 (service API) on BoTorch — GP surrogate (SingleTaskGP,
Matern-ARD kernel) + qLogNEHVI acquisition via Generators.BOTORCH_MODULAR.
NOT hand-rolled: Sobol initialization and every subsequent candidate come
from ax_client.get_next_trial(); the acquisition optimizer runs inside Ax.

"BO is actually working" evidence, logged per iteration to
results/<run>/log.jsonl (§0.2 hard deliverable):
  - generation method of each trial (Sobol vs BoTorchGenerator);
  - GP hyperparameters after each refit: ARD lengthscales per objective,
    observation noise, and the exact marginal log likelihood (LML);
  - the acquisition-selected candidate itself (proof the point came from
    qLogNEHVI, not chance) + gen_metadata when Ax exposes it;
  - dominated hypervolume of the observed set after every evaluation
    (the convergence curve compared against random_baseline.py).

  python3 bo_engine.py --budget 60 --seed 0 --out results/bo_seed0
"""
import argparse
import json
import logging
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import torch

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.generation_strategy.generation_strategy import (GenerationStrategy,
                                                        GenerationStep)
from ax.adapter.registry import Generators
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.multi_objective.pareto import is_non_dominated

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import evaluate, KNOBS  # noqa: E402

# Objective sense and hypervolume reference point (worst plausible values;
# fixed so BO and random curves are directly comparable).
OBJECTIVES = {
    "decode_tput":     dict(minimize=False, ref=20.0),
    "mxu_util":        dict(minimize=False, ref=0.0),
    "arith_intensity": dict(minimize=False, ref=5.0),
    "area_mm2":        dict(minimize=True,  ref=20.0),
}
PENALTY = {  # INVALID/infeasible points: pessimal values (kept out of the
    "decode_tput": 20.0, "mxu_util": 0.0,   # Pareto set by construction)
    "arith_intensity": 5.0, "area_mm2": 20.0,
}


def ax_parameters():
    return [
        dict(name="num_tiles", type="choice", values=[8, 16, 32],
             is_ordered=True, sort_values=True),
        dict(name="tiles_per_group", type="choice", values=[2, 4, 8],
             is_ordered=True, sort_values=True),
        dict(name="shared_mb", type="choice", values=[4, 8, 16],
             is_ordered=True, sort_values=True),
        dict(name="topology", type="choice",
             values=["ring", "mesh", "torus"], is_ordered=False),
        dict(name="W", type="choice", values=[1, 2, 4],
             is_ordered=True, sort_values=True),
        dict(name="dram_density", type="choice",
             values=[512, 1024, 2048, 4096, 8192],
             is_ordered=True, sort_values=True),
    ]


def objective_vector(res):
    """Map an evaluate() result to the objective dict (penalty if invalid)."""
    if res.get("valid"):
        return {k: float(res["objectives"][k]) for k in OBJECTIVES}
    return dict(PENALTY)


def hv_of(points):
    """Dominated hypervolume of observed objective vectors (maximization
    frame: minimized objectives are negated)."""
    if not points:
        return 0.0
    Y = torch.tensor([[(-v[k] if OBJECTIVES[k]["minimize"] else v[k])
                       for k in OBJECTIVES] for v in points], dtype=torch.double)
    ref = torch.tensor([(-OBJECTIVES[k]["ref"] if OBJECTIVES[k]["minimize"]
                         else OBJECTIVES[k]["ref"]) for k in OBJECTIVES],
                       dtype=torch.double)
    Y = Y[(Y > ref).all(dim=-1)]
    if Y.shape[0] == 0:
        return 0.0
    pareto = Y[is_non_dominated(Y)]
    return Hypervolume(ref_point=ref).compute(pareto)


def gp_evidence(ax_client):
    """Extract GP hyperparameters + LML from the current Ax adapter.
    Returns None while still in the Sobol phase."""
    try:
        adapter = ax_client.generation_strategy.adapter
        gen = adapter.generator
        gp = gen.surrogate.model
    except Exception:
        return None
    if gp is None:
        return None
    out = {"gp_class": type(gp).__name__}
    models = list(gp.models) if hasattr(gp, "models") else [gp]
    evid = []
    for sm in models:
        e = {}
        ker = getattr(sm.covar_module, "base_kernel", sm.covar_module)
        if getattr(ker, "lengthscale", None) is not None:
            # shape (n_outputs, 1, n_dims) -> rows = objective, cols = dim
            ls = ker.lengthscale.detach().squeeze(-2)
            if ls.dim() == 1:
                ls = ls.unsqueeze(0)
            e["ard_lengthscales"] = np.round(ls.numpy(), 3).tolist()
        try:
            noise = sm.likelihood.noise.detach().flatten().tolist()
            e["noise"] = [round(x, 5) for x in noise]
        except Exception:
            pass
        try:
            mll = ExactMarginalLogLikelihood(sm.likelihood, sm)
            with torch.no_grad():
                v = mll(sm(*sm.train_inputs), sm.train_targets)
            e["lml"] = [round(x, 3) for x in np.atleast_1d(
                v.detach().numpy()).tolist()]
            e["train_n"] = int(sm.train_targets.shape[-1])
        except Exception as ex:
            e["lml_error"] = str(ex)[:100]
        evid.append(e)
    out["models"] = evid
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=60)
    ap.add_argument("--sobol", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    outdir = a.out or os.path.join(HERE, "results", f"bo_seed{a.seed}")
    os.makedirs(outdir, exist_ok=True)
    logf = open(os.path.join(outdir, "log.jsonl"), "w")

    # should_deduplicate: the discrete space is small (1215 points); without
    # it qNEHVI re-proposes already-evaluated configs after rounding and
    # burns budget on duplicates (observed: 24/60 wasted in the first run).
    gs = GenerationStrategy(steps=[
        GenerationStep(generator=Generators.SOBOL, num_trials=a.sobol,
                       model_kwargs={"seed": a.seed},
                       should_deduplicate=True),
        GenerationStep(generator=Generators.BOTORCH_MODULAR, num_trials=-1,
                       should_deduplicate=True),
    ])
    ax_client = AxClient(generation_strategy=gs, verbose_logging=False,
                         random_seed=a.seed)
    ax_client.create_experiment(
        name=f"mobol_dse_seed{a.seed}",
        parameters=ax_parameters(),
        objectives={k: ObjectiveProperties(minimize=v["minimize"],
                                           threshold=v["ref"])
                    for k, v in OBJECTIVES.items()},
    )

    observed = []
    for it in range(a.budget):
        t0 = time.time()
        params, idx = ax_client.get_next_trial()
        tr = ax_client.experiment.trials[idx]
        gr = tr.generator_runs[0]
        method = gr._model_key or "?"

        res = evaluate(dict(params))
        obj = objective_vector(res)
        ax_client.complete_trial(
            trial_index=idx, raw_data={k: (v, 0.0) for k, v in obj.items()})
        observed.append(obj)
        hv = hv_of(observed)

        rec = dict(iter=it, method=method, params=params,
                   valid=bool(res.get("valid")),
                   reason=res.get("reason"), objectives=obj,
                   hypervolume=hv, wall_s=round(time.time() - t0, 1))
        gm = gr.gen_metadata or {}
        if gm:
            rec["gen_metadata"] = {k: str(v)[:120] for k, v in gm.items()}
        ev = gp_evidence(ax_client) if method != "Sobol" else None
        if ev:
            rec["gp"] = ev
        logf.write(json.dumps(rec) + "\n")
        logf.flush()
        print(f"[{it:3d}] {method:8s} {params['num_tiles']:2d}t/{params['tiles_per_group']}g/"
              f"{params['shared_mb']:2d}MB/{params['topology']:5s}/W{params['W']}/D{params['dram_density']:4d} "
              f"tput={obj['decode_tput']:6.1f} util={obj['mxu_util']:5.2f} "
              f"AI={obj['arith_intensity']:5.1f} area={obj['area_mm2']:5.2f} "
              f"HV={hv:9.1f} ({rec['wall_s']}s)"
              + ("" if res.get("valid") else f"  INVALID: {res.get('reason')}"),
              flush=True)

    json.dump(dict(budget=a.budget, seed=a.seed,
                   final_hv=hv_of(observed)),
              open(os.path.join(outdir, "summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
