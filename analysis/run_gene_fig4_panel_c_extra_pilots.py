#!/usr/bin/env python3
"""Generate extra metagenomic Fig.4 panel-C pilot-extrapolation curves.

The original metagenomic Fig.4 table contains pilot n=4/7/10. This targeted
script adds simulated pilot n=30/50/80 using the same PCAM support-preserving
mechanism, then evaluates power at n=80.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import pcam_gen as P
from phylopower import core
from semisynthetic_power import summarize_distance_metrics_with_replacement


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "archived_runs" / "fig4_new" / "fig4_metagenomics_panel_c_extra_pilots.csv"

PILOTS = (30, 50, 80)
EVAL_N = 80
POOL_M = 300
BOOT = 100
PERMS = 99
SEED0 = 1000

GENE_GRID = [
    (0.50, 1.00),
    (0.60, 1.00),
    (0.68, 1.00),
    (0.75, 1.00),
    (0.82, 1.00),
    (0.88, 1.00),
    (0.93, 1.00),
    (0.97, 1.00),
    (1.00, 1.00),
    (1.00, 1.30),
    (1.00, 1.70),
]

_BASE = None
_PILOTS = {}


def _as_gene_raw_dict(base: dict, tab: pd.DataFrame, sgm: pd.Series) -> dict:
    groups = list(pd.unique(sgm))
    e = dict(base)
    e["abund"] = tab.to_numpy(dtype=float)
    e["L"] = np.log1p(e["abund"])
    e["groups"] = groups
    e["gs"] = {g: np.where(sgm.to_numpy() == g)[0] for g in groups}
    pall = np.concatenate([e["gs"][g] for g in groups])
    e["libs"] = e["abund"][:, pall].sum(axis=0)
    grand = e["L"][:, pall].mean(axis=1)
    e["dev"] = {g: e["L"][:, e["gs"][g]].mean(axis=1) - grand for g in groups}
    return e


def _make_gene_pilot(base: dict, pilot_n: int, seed: int) -> dict:
    observed_n = min(len(base["gs"][g]) for g in base["groups"])
    if pilot_n <= observed_n:
        return P.pilot_view(base, pilot_n, seed)
    tab, sgm = P.pcam_pool(base, pilot_n, seed, pi=1.0, scale=1.0, ndon=1)
    return _as_gene_raw_dict(base, tab, sgm)


def _init() -> None:
    global _BASE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    core.load_core_runtime()
    _BASE = P.load_modality("gene")


def _task(job: tuple[int, int, float, float, int]) -> dict:
    core.load_core_runtime()
    pilot_n, pilot_seed, pi, scale, gen_seed = job
    key = (pilot_n, pilot_seed)
    if key not in _PILOTS:
        _PILOTS[key] = _make_gene_pilot(_BASE, pilot_n, pilot_seed)
    pilot = _PILOTS[key]
    tab, sgm = P.pcam_pool(pilot, POOL_M, gen_seed, pi=pi, scale=scale, ndon=1)
    dm = P.recompute_distance(pilot, tab)
    omega = max(0.0, float(core.compute_omega2(dm, sgm)))
    metrics = summarize_distance_metrics_with_replacement(
        dm=dm,
        group_map=sgm,
        boot_number=BOOT,
        alpha=0.05,
        n_jobs=1,
        random_seed=gen_seed + EVAL_N + 31,
        n_per_group=EVAL_N,
        permutations=PERMS,
        omega2_floor=0.0,
    )
    return {
        "modality": "gene",
        "panel": "c_extra",
        "pilot": int(pilot_n),
        "pilot_seed": int(pilot_seed),
        "pi": float(pi),
        "scale": float(scale),
        "eval_n": int(EVAL_N),
        "true_omega2": float(omega),
        "power": float(metrics["power"]),
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    jobs = []
    for pn in PILOTS:
        pilot_seed = SEED0 + pn * 1009
        for i, (pi, scale) in enumerate(GENE_GRID):
            jobs.append((pn, pilot_seed, float(pi), float(scale), SEED0 + pn * 10000 + i * 131))
    print(f"[gene-panel-c-extra] jobs={len(jobs)} pilots={PILOTS}", flush=True)
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as ex:
        rows = list(ex.map(_task, jobs))
    df = pd.DataFrame(rows).sort_values(["pilot", "true_omega2", "pi", "scale"])
    df.to_csv(OUT, index=False)
    print(f"[gene-panel-c-extra] wrote {OUT} shape={df.shape}", flush=True)


if __name__ == "__main__":
    main()
