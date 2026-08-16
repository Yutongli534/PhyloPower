#!/usr/bin/env python3
"""Generate a dense metagenomic Fig.4 panel-A study-size table.

This is a targeted rerun for the study-size family only. It keeps the same PCAM
pilot-aware evaluation logic as fig4.py, but evaluates n=4 and n=7 and uses a
denser effect grid so the left panel has more observed points.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import pcam_gen as P


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "archived_runs" / "fig4_new" / "fig4_metagenomics_panel_a_dense.csv"

PILOT_N = 10
PILOT_SEED = 1000 + PILOT_N * 1009 + 777
POOL_M = 300
BOOT = 100
EVAL_NS = (4, 7, 10, 30, 50, 80)

DENSE_GRID = [
    (0.50, 1.00),
    (0.55, 1.00),
    (0.60, 1.00),
    (0.65, 1.00),
    (0.70, 1.00),
    (0.75, 1.00),
    (0.80, 1.00),
    (0.84, 1.00),
    (0.88, 1.00),
    (0.91, 1.00),
    (0.94, 1.00),
    (0.96, 1.00),
    (0.98, 1.00),
    (1.00, 1.00),
    (1.00, 1.10),
    (1.00, 1.20),
    (1.00, 1.30),
    (1.00, 1.45),
    (1.00, 1.60),
    (1.00, 1.80),
]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for i, (pi, scale) in enumerate(DENSE_GRID):
        jobs.append(
            (
                float(pi),
                float(scale),
                POOL_M,
                PILOT_SEED + 7000 + i * 131,
                EVAL_NS,
                BOOT,
                PILOT_N,
                PILOT_SEED,
            )
        )
    print(f"[gene-panel-a-dense] jobs={len(jobs)} eval_ns={EVAL_NS}", flush=True)
    results = P.eval_pilot("gene", jobs, n_workers=6)

    rows = []
    for pn, pseed, pi, scale, omega, powers in results:
        for en in EVAL_NS:
            rows.append(
                {
                    "modality": "gene",
                    "panel": "b_dense",
                    "pilot": int(pn),
                    "pilot_seed": int(pseed),
                    "pi": float(pi),
                    "scale": float(scale),
                    "eval_n": int(en),
                    "true_omega2": float(omega),
                    "power": float(powers[en]),
                }
            )
    df = pd.DataFrame(rows).sort_values(["eval_n", "true_omega2", "pi", "scale"])
    df.to_csv(OUT, index=False)
    print(f"[gene-panel-a-dense] wrote {OUT} shape={df.shape}", flush=True)


if __name__ == "__main__":
    main()
