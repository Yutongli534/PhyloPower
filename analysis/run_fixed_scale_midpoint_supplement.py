#!/usr/bin/env python3
"""Compute fixed-scale midpoint supplements for the bottom row tree-error panels.

This skips the expensive power-uniform preview and evaluates only extra scales
selected from an existing curve's transition region.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path("/tmp") / "numba_cache_phylopower"))
os.environ["PATH"] = "/opt/miniconda3/bin:" + os.environ.get("PATH", "")
os.environ["PYTHONPATH"] = str(Path(__file__).resolve().parent) + ":" + os.environ.get("PYTHONPATH", "")

try:
    import psutil

    if psutil.cpu_count() is None:
        psutil.cpu_count = lambda logical=True: 8
except Exception:
    pass

from phylopower import core  # (import first: installs the embedded-module finder)
import gene_power_workflow as gene_wf
import protein_power_workflow as protein_wf
from semisynthetic_power import (
    ID_COLS,
    _effective_pool_size,
    _ordination_pool_dm,
    _ordination_pool_group_map,
    _read_group_map,
    _read_protein_long_table,
    _read_taxon_feature_table,
    generate_taxon_function_pool,
    generate_taxon_pool,
    summarize_distance_metrics_with_replacement,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "archived_runs" / "tree_error_fig5_like_quick"
PILOTS = {
    "gene": [4, 7, 10, 30, 50, 80],
    "protein": [7, 10, 17, 30, 50, 80],
}


def default_args(modality: str, out: Path) -> SimpleNamespace:
    if modality == "gene":
        return SimpleNamespace(
            table=core.DATAGENE_DIR / "table.csv",
            tree=core.DATAGENE_DIR / "rooted-tree.nwk",
            taxonomy=core.DATAGENE_DIR / "taxonomy.csv",
            group=core.DATAGENE_DIR / "group.csv",
            use_phylogeny=True,
            qiime_env="qiime2-metagenome-2024.10",
            tree_jitter_sigma=0.5,
            tree_nni_prob=0.5,
            tree_support_threshold=None,
            random_seed=20260614,
            sim_seed=20260614,
            pool_size_per_group=300,
            boot_number=12,
            permutations=49,
            alpha=0.05,
            eval_n=80,
            center_mode="omega-calibrated",
            cov_estimator="ledoit-wolf",
            embed_dim=None,
            cov_eb_pool=False,
            pool_cov=False,
            pool_dist="student-t",
            pool_df="auto",
            ordination_enhance_max=3.0,
            omega_calibrated_max_scale=20.0,
            adaptive_reference_n=17.0,
            omega2_floor=0.0,
            n_jobs=1,
            pcam_gene_blocks=8,
            pcam_ndon=1,
            out=out,
        )
    return SimpleNamespace(
        table=core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv",
        tree=core.DATAPRO_DIR / "rooted-tree.nwk",
        group=core.DATAPRO_DIR / "group.csv",
        tree_jitter_sigma=0.5,
        tree_nni_prob=0.5,
        tree_support_threshold=None,
        random_seed=20260614,
        sim_seed=20260614,
        pool_size_per_group=160,
        boot_number=12,
        permutations=49,
        alpha=0.05,
        eval_n=80,
        center_mode="omega-calibrated",
        cov_estimator="ledoit-wolf",
        embed_dim=None,
        pool_cov=False,
        cov_eb_pool=False,
        protein_transform="vst",
        lowrank_rank=5,
        agg_clades=300,
        pool_dist="student-t",
        pool_df="auto",
        protein_generator="template-mask",
        omega2_floor=0.0,
        n_jobs=1,
        out=out,
    )


def existing_curve(modality: str, pilot: int) -> pd.DataFrame:
    if modality == "gene":
        path = BASE / "gene_pilot_extrapolation" / "gene_power_curves_raw.csv"
    else:
        path = BASE / "protein_pilot_extrapolation_refit" / "protein_power_curves_raw_combined.csv"
    df = pd.read_csv(path)
    return df[df["pilot_n"].eq(pilot)].copy().sort_values("scale")


def choose_midpoint_scales(df: pd.DataFrame, max_new: int) -> List[float]:
    rows = df.sort_values("scale")[["scale", "power"]].dropna().drop_duplicates("scale")
    vals = rows.to_numpy(float)
    candidates: List[tuple[float, float]] = []
    for left, right in zip(vals[:-1], vals[1:]):
        s0, p0 = left
        s1, p1 = right
        if s1 <= s0:
            continue
        in_window = (
            0.10 <= p0 <= 0.97
            or 0.10 <= p1 <= 0.97
            or (min(p0, p1) < 0.80 < max(p0, p1))
            or (min(p0, p1) < 0.50 < max(p0, p1))
        )
        if in_window:
            candidates.append((abs(p1 - p0), (s0 + s1) / 2.0))
            if abs(p1 - p0) >= 0.18:
                candidates.append((abs(p1 - p0) * 0.8, s0 + (s1 - s0) * 0.33))
                candidates.append((abs(p1 - p0) * 0.8, s0 + (s1 - s0) * 0.67))
    candidates.sort(reverse=True)
    existing = set(np.round(rows["scale"].to_numpy(float), 10))
    selected: List[float] = []
    for _score, scale in candidates:
        rounded = round(float(scale), 10)
        if rounded in existing or any(abs(float(scale) - s) < 1e-8 for s in selected):
            continue
        selected.append(float(scale))
        if len(selected) >= max_new:
            break
    return sorted(selected)


def build_gene_model_for_pilot(args: SimpleNamespace, pilot: int) -> Dict:
    group_map = _read_group_map(args.group)
    table, agm = _read_taxon_feature_table(args.table, group_map)
    observed_n = int(agm.value_counts().min())
    seed = int(args.random_seed) + pilot * 1009
    if pilot <= observed_n:
        rng = np.random.default_rng(seed)
        picked: List[str] = []
        for g in sorted(agm.unique()):
            members = agm[agm == g].index.to_numpy()
            picked.extend(rng.choice(members, size=pilot, replace=False).tolist())
        ptable = table[picked].copy()
        pgm = agm.loc[picked]
    else:
        ptable, pgm, _ = generate_taxon_pool(
            table,
            agm,
            pool_size_per_group=pilot,
            random_seed=int(args.sim_seed),
            between_scale=1.0,
            residual_scale=1.0,
            noise_multiplier=0.10,
        )
    return gene_wf.build_gene_model(ptable, pgm, args, stem=f"fixed_midpoint_pilot{pilot}")


def build_protein_model_for_pilot(args: SimpleNamespace, pilot: int) -> Dict:
    group_map = _read_group_map(args.group)
    long_df, agm = _read_protein_long_table(args.table, group_map)
    observed_n = int(agm.value_counts().min())
    seed = int(args.random_seed) + pilot * 1009
    if pilot <= observed_n:
        pgm = protein_wf._subsample_group_map(agm, pilot, seed)
        pdf = long_df[ID_COLS + list(pgm.index)].copy()
    else:
        pdf, pgm, _ = generate_taxon_function_pool(
            long_df,
            agm,
            pool_size_per_group=pilot,
            random_seed=int(args.sim_seed),
            between_scale=1.0,
            residual_scale=1.0,
            noise_multiplier=0.10,
            detection_slope=1.0,
            protein_generator=args.protein_generator,
        )
    return protein_wf.build_protein_model(pdf, pgm, args)


def compute_fixed_scales(model: Dict, args: SimpleNamespace, scales: Iterable[float], seed: int) -> pd.DataFrame:
    eval_n = int(args.eval_n)
    psize = _effective_pool_size(args.pool_size_per_group, eval_n)
    pool_gm = _ordination_pool_group_map(model, psize)
    rows = []
    for i, sc in enumerate(scales):
        dm = _ordination_pool_dm(model, float(sc), psize, seed + 19000 + i * 997)
        metrics = summarize_distance_metrics_with_replacement(
            dm=dm,
            group_map=pool_gm,
            boot_number=int(args.boot_number),
            alpha=float(args.alpha),
            n_jobs=int(args.n_jobs),
            random_seed=seed + 7000 + i,
            n_per_group=eval_n,
            permutations=int(args.permutations),
            omega2_floor=getattr(args, "omega2_floor", None),
        )
        rows.append(
            {
                "scale": float(sc),
                "true_omega2": float(metrics["true_omega2"]),
                "power": float(metrics["power"]),
                "mode": "fixed_midpoint",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["gene", "protein"], required=True)
    parser.add_argument("--pilot", type=int, required=True)
    parser.add_argument("--max-new", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=BASE / "fixed_midpoint_supplement")
    args0 = parser.parse_args()

    out_dir = args0.out_dir / args0.modality / f"p{args0.pilot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    args = default_args(args0.modality, out_dir)
    base = existing_curve(args0.modality, args0.pilot)
    max_new = int(args0.max_new if args0.max_new is not None else (10 if args0.modality == "gene" else 12))
    scales = choose_midpoint_scales(base, max_new=max_new)
    if not scales:
        raise SystemExit(f"no midpoint scales selected for {args0.modality} p={args0.pilot}")
    pd.DataFrame({"scale": scales}).to_csv(out_dir / "selected_scales.csv", index=False)
    seed = int(args.random_seed) + args0.pilot * 1009
    if args0.modality == "gene":
        model = build_gene_model_for_pilot(args, args0.pilot)
    else:
        model = build_protein_model_for_pilot(args, args0.pilot)
    rows = compute_fixed_scales(model, args, scales, seed=seed)
    rows.insert(0, "pilot_n", int(args0.pilot))
    rows.insert(1, "eval_n", int(args.eval_n))
    rows.to_csv(out_dir / "fixed_midpoint_rows.csv", index=False)
    print(f"[fixed] {args0.modality} p={args0.pilot}: {len(rows)} rows -> {out_dir / 'fixed_midpoint_rows.csv'}", flush=True)


if __name__ == "__main__":
    main()
