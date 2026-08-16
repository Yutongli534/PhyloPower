#!/usr/bin/env python3
"""Route A: Monte-Carlo ground-truth PERMANOVA power for the benchmark pools.

Protocol (the shared yardstick for all three tools): for each
(metric, tier, n_per_group) cell, draw samples WITHOUT replacement from the
exported pool distance matrix (n per group), run skbio PERMANOVA with 999
permutations at alpha = 0.05, and record rejection. Truth power = rejection
rate over --draws draws.

Deliberately does NOT import phylopower (its embedded modules break
multiprocessing pickling); only skbio/pandas/numpy are used, so stdlib
multiprocessing with --jobs workers is safe.

Usage:
  python3 benchmark/py/run_truth.py --grid pilot     # tier {0,1.0} x n {6,10}
  python3 benchmark/py/run_truth.py --grid extended  # tier {0,0.5,1.0} x n {4,6,8,10,14}
Output: benchmark/results/accuracy_truth/truth_<metric>_<tag>_n<n>.csv
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "benchmark" / "data"
OUT = REPO / "benchmark" / "results" / "accuracy_truth"

SEED = 20260614
PERMUTATIONS = 999
ALPHA = 0.05
DRAWS = 5000

GRIDS = {
    "pilot": {"tiers": [0.0, 1.0], "ns": [6, 10]},
    "extended": {"tiers": [0.0, 0.5, 1.0], "ns": [4, 6, 8, 10, 14]},
}
METRICS = ["braycurtis", "wunifrac"]


def scale_tag(scale: float) -> str:
    s = f"{scale:.1f}" if scale == int(scale) else str(scale)
    return s.replace(".", "p")


def run_cell(metric: str, scale: float, n: int, draws: int = DRAWS) -> dict:
    from skbio import DistanceMatrix
    from skbio.stats.distance import permanova

    tag = scale_tag(scale)
    dm = pd.read_csv(DATA / f"dm_{metric}_scale{tag}.csv", index_col=0)
    grp = pd.read_csv(DATA / f"group_scale{tag}.csv")
    group_map = pd.Series(grp["group"].values, index=grp["sample_id"].astype(str))
    groups = sorted(group_map.unique())
    members = {g: group_map[group_map == g].index.to_numpy() for g in groups}

    # deterministic per-cell seed stream, independent of worker scheduling
    cell_key = abs(hash((metric, scale, n))) % (2**31)
    ss = np.random.SeedSequence([SEED, cell_key])
    draw_seeds = ss.spawn(draws)

    rejects = 0
    for i in range(draws):
        rng = np.random.default_rng(draw_seeds[i])
        picked = np.concatenate([rng.choice(members[g], size=n, replace=False) for g in groups])
        sub = dm.loc[picked, picked]
        labels = [g for g in groups for _ in range(n)]
        sk_dm = DistanceMatrix(np.ascontiguousarray(sub.to_numpy()), ids=list(sub.columns))
        p = permanova(sk_dm, labels, permutations=PERMUTATIONS, seed=int(rng.integers(2**31 - 1)))[
            "p-value"
        ]
        rejects += int(p < ALPHA)
    return {
        "metric": metric,
        "between_scale": scale,
        "n_per_group": n,
        "truth_power": rejects / draws,
        "n_draws": draws,
        "permutations": PERMUTATIONS,
        "alpha": ALPHA,
    }


def _run_and_write(args):
    metric, scale, n = args
    t0 = time.time()
    row = run_cell(metric, scale, n)
    row["seconds"] = round(time.time() - t0, 1)
    tag = scale_tag(scale)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(OUT / f"truth_{metric}_scale{tag}_n{n}.csv", index=False)
    print(row, flush=True)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=list(GRIDS), default="pilot")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--draws", type=int, default=DRAWS)
    args = parser.parse_args()

    grid = GRIDS[args.grid]
    cells = [
        (metric, scale, n)
        for metric in METRICS
        for scale in grid["tiers"]
        for n in grid["ns"]
    ]
    # skip cells that already have a shard with the requested draw count
    todo = []
    for metric, scale, n in cells:
        path = OUT / f"truth_{metric}_scale{scale_tag(scale)}_n{n}.csv"
        if path.exists() and int(pd.read_csv(path)["n_draws"].iloc[0]) >= args.draws:
            print(f"skip {metric} scale={scale} n={n} (done)", flush=True)
        else:
            todo.append((metric, scale, n))
    print(f"{len(todo)} cells to run with {args.jobs} workers", flush=True)
    with mp.Pool(processes=args.jobs) as pool:
        pool.map(_run_and_write, todo)
    print("TRUTH GRID DONE", flush=True)


if __name__ == "__main__":
    main()
