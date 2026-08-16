#!/usr/bin/env python3
"""Route A: PhyloPower power ESTIMATES for the accuracy benchmark.

Each "rep" is one standard PhyloPower estimate for a cell: 200 without-
replacement bootstraps from the pool distance matrix, skbio PERMANOVA with
199 permutations, alpha = 0.05 (i.e. the same settings as
run_phylopower_power.py), with the rep seed derived from 20260614.

Usage: python3 benchmark/py/run_accuracy_phylopower.py --grid pilot
Output: benchmark/results/accuracy_estimates/phylopower_<metric>_<tag>_n<n>.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

from phylopower import core  # noqa: E402,F401  (installs embedded module finder)
import semisynthetic_power as sp  # noqa: E402

DATA = REPO / "benchmark" / "data"
OUT = REPO / "benchmark" / "results" / "accuracy_estimates"

SEED = 20260614
BOOT_NUMBER = 200
PERMUTATIONS = 199
ALPHA = 0.05
N_REPS = 50

GRIDS = {
    "pilot": {"tiers": [0.0, 1.0], "ns": [6, 10]},
    "extended": {"tiers": [0.0, 0.5, 1.0], "ns": [4, 6, 8, 10, 14]},
}
METRICS = ["braycurtis", "wunifrac"]


def scale_tag(scale: float) -> str:
    s = f"{scale:.1f}" if scale == int(scale) else str(scale)
    return s.replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", choices=list(GRIDS), default="pilot")
    parser.add_argument("--reps", type=int, default=N_REPS)
    args = parser.parse_args()
    grid = GRIDS[args.grid]
    OUT.mkdir(parents=True, exist_ok=True)

    for metric in METRICS:
        for scale in grid["tiers"]:
            tag = scale_tag(scale)
            dm = pd.read_csv(DATA / f"dm_{metric}_scale{tag}.csv", index_col=0)
            grp = pd.read_csv(DATA / f"group_scale{tag}.csv")
            group_map = pd.Series(grp["group"].values, index=grp["sample_id"].astype(str), name="group")
            for n in grid["ns"]:
                path = OUT / f"phylopower_{metric}_scale{tag}_n{n}.csv"
                rows = []
                for rep in range(args.reps):
                    m = sp.summarize_distance_metrics_without_replacement(
                        dm=dm,
                        group_map=group_map,
                        boot_number=BOOT_NUMBER,
                        alpha=ALPHA,
                        n_jobs=1,  # embedded modules are not picklable; ~2 ms/boot
                        random_seed=SEED + rep,
                        n_per_group=n,
                        permutations=PERMUTATIONS,
                    )
                    rows.append(
                        {
                            "tool": "PhyloPower",
                            "metric": metric,
                            "between_scale": scale,
                            "n_per_group": n,
                            "rep": rep + 1,
                            "power_est": m["power"],
                            "boot_number": BOOT_NUMBER,
                            "permutations": PERMUTATIONS,
                        }
                    )
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"{metric} scale={scale} n={n}: {args.reps} reps done", flush=True)
    print("PHYLOPOWER ACCURACY DONE", flush=True)


if __name__ == "__main__":
    main()
