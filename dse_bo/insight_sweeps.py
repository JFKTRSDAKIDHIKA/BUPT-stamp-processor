#!/usr/bin/env python3
"""Phase 2 targeted grids answering the four §Phase-2.3 questions.

All points go through evaluate() (same §3.3 validation gate + cache as BO),
so BO samples and these grids share one result pool.

Q1  bond-density knee vs (num_tiles, tiles_per_group):
      nt x tpg x density grid @ ring/8MB/W1.
Q2  shared-SRAM reuse knee (KV-cache residency capacity boundary):
      capacity = num_banks*shared_mb sweep via (tpg, smb) @ nt=16;
      f16 (Llama) vs INT4 (Mistral) contrast comes from the per-model
      metrics stored with every evaluation.
Q3  does adding tiles starve? nt x W grid @ D=2048.
Q4  bank:tile ratio (port/column sharing): tpg x W @ nt=16, D=2048.

  python3 insight_sweeps.py [--workers 2]
"""
import argparse
import itertools
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evaluate import evaluate  # noqa: E402


def worklist():
    pts = []   # (config, ctx)
    base = dict(topology="ring", shared_mb=8, W=1, dram_density=2048)
    # Q1: density knee vs structure
    for nt, tpg, d in itertools.product((8, 16, 32), (2, 4, 8),
                                        (512, 1024, 2048, 4096, 8192)):
        pts.append((dict(base, num_tiles=nt, tiles_per_group=tpg,
                         dram_density=d), 512))
    # Q2: shared capacity boundary at long context (KV: Llama-f16 67 MB,
    # Mistral-INT4 33.5 MB @ ctx 2048 — the residency knee crosses the
    # (num_banks x shared_mb) grid, at different points per precision)
    for nt, tpg, smb in itertools.product((16, 32), (2, 4, 8), (4, 8, 16)):
        pts.append((dict(base, num_tiles=nt, tiles_per_group=tpg,
                         shared_mb=smb), 2048))
    # Q3: tiles x W
    for nt, w in itertools.product((8, 16, 32), (1, 2, 4)):
        pts.append((dict(base, num_tiles=nt, tiles_per_group=4, W=w), 512))
    # Q4: tpg x W
    for tpg, w in itertools.product((2, 4, 8), (1, 2, 4)):
        pts.append((dict(base, num_tiles=16, tiles_per_group=tpg, W=w), 512))
    # dedupe
    seen, out = set(), []
    for p, ctx in pts:
        k = (tuple(sorted(p.items())), ctx)
        if k not in seen:
            seen.add(k)
            out.append((p, ctx))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    pts = worklist()
    print(f"{len(pts)} sweep points")
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(lambda pc: evaluate(pc[0], ctx=pc[1]), pts):
            c = r["config"]
            tag = (f"{c['num_tiles']}t/{c['tiles_per_group']}g/"
                   f"{c['shared_mb']}MB/{c['topology']}/W{c['W']}/"
                   f"D{c['dram_density']}/ctx{r.get('ctx', 512)}")
            if r.get("valid"):
                o = r["objectives"]
                print(f"{tag}: tput={o['decode_tput']:.1f} "
                      f"AI={o['arith_intensity']:.2f}", flush=True)
            else:
                print(f"{tag}: {r.get('reason')}", flush=True)


if __name__ == "__main__":
    main()
