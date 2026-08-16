#!/usr/bin/env python3
"""PhyloPower benchmark driver.

For each (metric, effect tier, n per group): subsample n samples per group
without replacement from the exported synthetic-pool distance matrix and run a
bootstrap PERMANOVA power estimate with PhyloPower's own machinery
(semisynthetic_power.summarize_distance_metrics_without_replacement ->
phylopower.core.compute_permanova_p_value / compute_omega2).

Settings per task spec: bootstrap = 200, permutations = 199, seed = 20260614,
alpha = 0.05, serial execution (see N_JOBS note below).

Output: benchmark/results/power_phylopower.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from phylopower import core  # noqa: E402,F401  (installs embedded module finder)
import semisynthetic_power as sp  # noqa: E402

DATA_DIR = REPO / "benchmark" / "data"
OUT_DIR = REPO / "benchmark" / "results"

N_GRID = [4, 6, 8, 10, 14, 20]
BOOT_NUMBER = 200
PERMUTATIONS = 199
ALPHA = 0.05
SEED = 20260614
N_JOBS = 1  # embedded paper_core modules are not picklable for joblib workers;
            # each bootstrap is ~2 ms so serial execution is fast enough


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tier_summary = pd.read_csv(DATA_DIR / "tier_summary.csv")
    rows = []
    for _, tier in tier_summary.iterrows():
        scale = float(tier["between_scale"])
        metric = str(tier["metric"])
        tag = str(scale if scale != int(scale) else f"{scale:.1f}").replace(".", "p")
        dm = pd.read_csv(DATA_DIR / f"dm_{metric}_scale{tag}.csv", index_col=0)
        grp = pd.read_csv(DATA_DIR / f"group_scale{tag}.csv")
        group_map = pd.Series(grp["group"].values, index=grp["sample_id"].astype(str), name="group")
        for n in N_GRID:
            metrics = sp.summarize_distance_metrics_without_replacement(
                dm=dm,
                group_map=group_map,
                boot_number=BOOT_NUMBER,
                alpha=ALPHA,
                n_jobs=N_JOBS,
                random_seed=SEED,
                n_per_group=n,
                permutations=PERMUTATIONS,
            )
            rows.append(
                {
                    "tool": "PhyloPower",
                    "metric": metric,
                    "between_scale": scale,
                    "omega2": metrics["true_omega2"],
                    "mean_boot_omega2": metrics["mean_boot_omega2"],
                    "n_per_group": n,
                    "power": metrics["power"],
                    "n_sim": BOOT_NUMBER,
                    "alpha": ALPHA,
                }
            )
            print(f"{metric} scale={scale} n={n} power={metrics['power']:.3f}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(OUT_DIR / "power_phylopower.csv", index=False)


if __name__ == "__main__":
    main()
