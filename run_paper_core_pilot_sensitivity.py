#!/usr/bin/env python3
"""Pilot-size sensitivity analysis for the raw-pool estimator.

The default layout compares:
  - gene pilot consistency at eval n=10
  - protein pilot consistency at eval n=17
  - gene pilot extrapolation at eval n=80
  - protein pilot extrapolation at eval n=80

Outputs:
  - sensitivity_raw_points.csv
  - sensitivity_curve_fits.csv
  - sensitivity_overlap_summary.csv
  - sensitivity_pilot_curves.png/pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _fig4_curve_plotting import binned_monotone, fit_binned_null_hill, null_started_hill_sigmoid
from phylopower import core as base_core
from phylopower import paper_core
from semisynthetic_power import _read_group_map, _read_protein_long_table, _read_taxon_feature_table
import gene_power_workflow as gene_wf
import protein_power_workflow as protein_wf


def _parse_ints(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _parse_floats(value: str) -> List[float]:
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def _unique_ints(values: Iterable[int]) -> List[int]:
    out: List[int] = []
    seen: set[int] = set()
    for value in values:
        ivalue = int(value)
        if ivalue not in seen:
            out.append(ivalue)
            seen.add(ivalue)
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if not np.isfinite(float(obj)) else float(obj)
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _effective_pool_size(requested: int, max_eval_n: int) -> int:
    return paper_core._effective_reusable_pool_size(
        requested_pool_size=int(requested),
        max_n=int(max_eval_n),
    )


def _precompute_gene_scenarios(
    *,
    raw_dict: dict,
    args: SimpleNamespace,
    pilot_n: int,
    seed: int,
    pool_size_per_group: int,
) -> List[Dict[str, Any]]:
    scenarios = gene_wf.precompute_pcam_scenarios(
        raw_dict,
        args,
        pilot_n,
        seed,
        pool_size_per_group=pool_size_per_group,
    )
    for scenario in scenarios:
        print(
            f"[sensitivity gene] selected pilot={pilot_n} "
            f"pi={scenario['pi']:.3f} scale={scenario['scale']:.3f} "
            f"omega2={scenario['true_omega2']:.4f}",
            flush=True,
        )
    return scenarios


def _evaluate_gene_scenarios(
    *,
    scenarios: List[Dict[str, Any]],
    args: SimpleNamespace,
    eval_n: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        metrics = gene_wf.summarize_distance_metrics_with_replacement(
            dm=scenario["dm"],
            group_map=scenario["group_map"],
            boot_number=args.boot_number,
            alpha=args.alpha,
            n_jobs=1,
            random_seed=int(seed) + int(scenario["scenario_index"]),
            n_per_group=int(eval_n),
            permutations=args.permutations,
            omega2_floor=args.omega2_floor,
        )
        rows.append(
            {
                "workflow": "gene",
                "pilot_n": int(scenario["pilot_n"]),
                "pilot_source": scenario["pilot_source"],
                "eval_n": int(eval_n),
                "effect_parameter": f"pi={scenario['pi']:.3g},scale={scenario['scale']:.3g}",
                "true_omega2": float(metrics["true_omega2"]),
                "power": float(metrics["power"]),
                "pool_size_per_group": int(scenario["pool_size_per_group"]),
                "scenario_index": int(scenario["scenario_index"]),
            }
        )
        print(
            f"[sensitivity gene] eval_n={eval_n} pilot={scenario['pilot_n']} "
            f"omega2={metrics['true_omega2']:.4f} power={metrics['power']:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _run_gene(args: argparse.Namespace, eval_ns: Iterable[int]) -> pd.DataFrame:
    gargs = paper_core._gene_args(
        table=args.gene_table,
        tree=args.gene_tree,
        taxonomy=args.gene_taxonomy,
        group=args.gene_group,
        out=args.out,
        target_power=args.target_power,
        target_omega2=args.target_omega2,
        qiime_env=args.qiime_env,
        alpha=args.alpha,
        pool_size_per_group=args.pool_size_per_group,
        boot_number=args.boot_number,
        permutations=args.permutations,
        pcam_gene_blocks=args.pcam_gene_blocks,
        pcam_ndon=args.pcam_ndon,
        pcam_grid=args.gene_grid,
        pcam_power_points=args.gene_power_points,
        pcam_near_zero_points=args.gene_near_zero_points,
        pcam_near_zero_omega_max=args.gene_near_zero_omega_max,
        pcam_pi_candidates=args.gene_pi_candidates,
        pcam_scale_candidates=args.gene_scale_candidates,
        pcam_scale_max=args.gene_scale_max,
        pcam_scale_extend_max=args.gene_scale_extend_max,
        fit_bin_width=args.gene_fit_bin_width,
        omega2_floor=args.omega2_floor,
        scenario_n_jobs=args.scenario_n_jobs,
        random_seed=args.random_seed,
        tree_jitter_sigma=0.0,
        tree_nni_prob=0.0,
        tree_support_threshold=None,
        use_phylogeny=True,
    )
    group_map = _read_group_map(args.gene_group)
    table, aligned_group_map = _read_taxon_feature_table(args.gene_table, group_map)
    raw_dict = gene_wf._build_gene_raw_dict(table, aligned_group_map, gargs)
    max_eval = max(int(x) for x in eval_ns)
    pool_size = _effective_pool_size(args.pool_size_per_group, max_eval)
    frames: List[pd.DataFrame] = []
    for pilot_n in _parse_ints(args.gene_pilot_ns):
        seed = int(args.random_seed + pilot_n * 1009)
        scenarios = _precompute_gene_scenarios(
            raw_dict=raw_dict,
            args=gargs,
            pilot_n=pilot_n,
            seed=seed,
            pool_size_per_group=pool_size,
        )
        for eval_n in eval_ns:
            frames.append(
                _evaluate_gene_scenarios(
                    scenarios=scenarios,
                    args=gargs,
                    eval_n=int(eval_n),
                    seed=seed + int(eval_n) * 37,
                )
            )
    return pd.concat(frames, ignore_index=True)


def _run_protein(args: argparse.Namespace, eval_ns: Iterable[int]) -> pd.DataFrame:
    pargs = paper_core._protein_args(
        table=args.protein_table,
        tree=args.protein_tree,
        group=args.protein_group,
        out=args.out,
        target_power=args.target_power,
        target_omega2=args.target_omega2,
        alpha=args.alpha,
        pool_size_per_group=args.pool_size_per_group,
        boot_number=args.boot_number,
        permutations=args.permutations,
        edge_fraction=args.edge_fraction,
        marginal_strength=args.marginal_strength,
        eb_k=args.eb_k,
        mdctf_strengths=args.protein_strengths,
        mdctf_strength_candidates=args.mdctf_strength_candidates,
        mdctf_strength_max=args.mdctf_strength_max,
        mdctf_power_points=args.mdctf_power_points,
        mdctf_plateau_points=args.mdctf_plateau_points,
        mdctf_preview_pool_size=args.mdctf_preview_pool_size,
        mdctf_refine_target_points=args.mdctf_refine_target_points,
        power_preview_boot_number=args.power_preview_boot_number,
        power_preview_permutations=args.power_preview_permutations,
        fit_bin_width=args.protein_fit_bin_width,
        omega2_floor=args.omega2_floor,
        scenario_n_jobs=args.scenario_n_jobs,
        random_seed=args.random_seed,
        tree_jitter_sigma=0.0,
        tree_nni_prob=0.0,
        tree_support_threshold=None,
    )
    group_map = _read_group_map(args.protein_group)
    long_df, aligned_group_map = _read_protein_long_table(args.protein_table, group_map)
    raw_dict = protein_wf._build_protein_raw_dict(long_df, aligned_group_map, pargs)
    max_eval = max(int(x) for x in eval_ns)
    pool_size = _effective_pool_size(args.pool_size_per_group, max_eval)
    frames: List[pd.DataFrame] = []
    for pilot_n in _parse_ints(args.protein_pilot_ns):
        seed = int(args.random_seed + pilot_n * 1009)
        scenarios = paper_core._precompute_protein_scenarios(
            raw_dict=raw_dict,
            args=pargs,
            pilot_n=pilot_n,
            seed=seed,
            pool_size_per_group=pool_size,
            strength_eval_n=max_eval,
        )
        for eval_n in eval_ns:
            df = paper_core._evaluate_precomputed_protein_scenarios(
                scenarios=scenarios,
                args=pargs,
                n_per_group=int(eval_n),
                seed=seed + int(eval_n) * 37,
            )
            df.insert(0, "workflow", "protein")
            df["effect_parameter"] = "strength=" + df["strength"].map(lambda x: f"{float(x):.3g}")
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _fit_one(sub: pd.DataFrame, target_omega2: float, bin_width: float) -> Dict[str, Any]:
    dat = sub[["true_omega2", "power"]].dropna().copy()
    if dat.empty:
        return {"fit_status": "no_data", "power_at_target": np.nan}
    xmax = max(float(dat["true_omega2"].max()) * 1.1, float(target_omega2) * 1.25, 0.06)
    x = np.linspace(0.0, xmax, 500)
    binned = binned_monotone(dat, bin_width)
    y, params = fit_binned_null_hill(binned, x)
    if params is None:
        bx = binned["true_omega2"].to_numpy(dtype=float)
        by = binned["power_mono"].to_numpy(dtype=float)
        power = float(np.interp(target_omega2, bx, by, left=by[0], right=by[-1])) if len(bx) else np.nan
        return {"fit_status": "fallback_interp", "power_at_target": power}
    power = float(null_started_hill_sigmoid(np.array([target_omega2]), params["h"], params["x0"], params["floor"])[0])
    return {
        "fit_status": "ok",
        "power_at_target": power,
        "h": float(params["h"]),
        "x0": float(params["x0"]),
        "floor": float(params["floor"]),
        "x_grid": x,
        "y_grid": y,
    }


def _joint_n_hill(omega: np.ndarray, n: np.ndarray, h: float, x0: float, beta: float, floor: float, ref_n: float) -> np.ndarray:
    x = np.clip(np.asarray(omega, dtype=float), 0.0, None)
    n_arr = np.clip(np.asarray(n, dtype=float), 1.0, None)
    x50 = np.clip(float(x0) * np.power(n_arr / float(ref_n), -float(beta)), 1e-8, None)
    xh = np.power(x + 1e-8, float(h))
    denom = xh + np.power(x50, float(h))
    return floor + (1.0 - floor) * (xh / np.clip(denom, 1e-12, None))


def _fit_joint_n(
    all_sub: pd.DataFrame,
    panel_sub: pd.DataFrame,
    eval_n: int,
    target_omega2: float,
    bin_width: float,
) -> Dict[str, Any]:
    dat = all_sub[["true_omega2", "power", "eval_n"]].dropna().copy()
    if dat["eval_n"].nunique() < 3 or len(dat) < 12:
        return _fit_one(panel_sub, target_omega2, bin_width)
    try:
        from scipy.optimize import curve_fit
    except Exception:
        return _fit_one(panel_sub, target_omega2, bin_width)

    dat["true_omega2"] = dat["true_omega2"].clip(lower=0.0)
    dat["power"] = dat["power"].clip(0.0, 1.0)
    ref_n = float(np.median(dat["eval_n"].to_numpy(dtype=float)))
    x_max = max(float(dat["true_omega2"].max()) * 1.1, float(target_omega2) * 1.5, 0.06)
    x_grid = np.linspace(0.0, x_max, 500)

    def model(xdata: tuple[np.ndarray, np.ndarray], h: float, x0: float, beta: float, floor: float) -> np.ndarray:
        omega, n = xdata
        return _joint_n_hill(omega, n, h, x0, beta, floor, ref_n)

    omega = dat["true_omega2"].to_numpy(dtype=float)
    n_arr = dat["eval_n"].to_numpy(dtype=float)
    y = dat["power"].to_numpy(dtype=float)
    try:
        popt, _ = curve_fit(
            model,
            (omega, n_arr),
            y,
            p0=(2.0, max(float(target_omega2), np.nanmedian(omega[omega > 0]) if np.any(omega > 0) else 0.03), 0.7, 0.05),
            bounds=([0.25, 1e-5, 0.0, 0.0], [12.0, max(x_max * 5.0, 0.5), 4.0, 0.5]),
            maxfev=50000,
        )
    except Exception:
        return _fit_one(panel_sub, target_omega2, bin_width)

    y_grid = _joint_n_hill(x_grid, np.full_like(x_grid, float(eval_n)), *popt, ref_n=ref_n)
    power = float(_joint_n_hill(np.array([target_omega2]), np.array([float(eval_n)]), *popt, ref_n=ref_n)[0])
    return {
        "fit_status": "joint_n",
        "power_at_target": power,
        "h": float(popt[0]),
        "x0_ref_n": float(popt[1]),
        "sample_size_beta": float(popt[2]),
        "floor": float(popt[3]),
        "joint_ref_n": ref_n,
        "joint_eval_n_count": int(dat["eval_n"].nunique()),
        "joint_point_count": int(len(dat)),
        "x_grid": x_grid,
        "y_grid": y_grid,
    }


def _adaptive_transition_xmax(sub: pd.DataFrame, args: argparse.Namespace) -> float | None:
    """Choose an x-axis maximum that emphasizes the empirical transition region.

    Power curves often include many high-effect scenarios with empirical power
    equal to one. Showing the full effect range then compresses the visually
    important low-omega transition into the left edge. This zoom rule keeps the
    non-saturated region and the first plateau point for each pilot, while
    hiding the uninformative saturated tail.
    """
    mode = str(getattr(args, "x_axis_mode", "auto")).strip().lower()
    if mode in {"full", "none"}:
        return None
    if sub.empty:
        return None
    dat = sub[["pilot_n", "true_omega2", "power"]].dropna().copy()
    if dat.empty:
        return None
    dat["true_omega2"] = dat["true_omega2"].clip(lower=0.0)
    all_max = float(dat["true_omega2"].max())
    if all_max <= 0:
        return None

    saturation = float(getattr(args, "x_axis_saturation_threshold", 0.98))
    margin = float(getattr(args, "x_axis_margin", 1.15))
    candidates: List[float] = []
    nonsat = dat[dat["power"] < saturation]
    if not nonsat.empty:
        candidates.append(float(nonsat["true_omega2"].max()))
    for _, pilot_df in dat.sort_values("true_omega2").groupby("pilot_n"):
        plateau = pilot_df[pilot_df["power"] >= saturation]
        if not plateau.empty:
            candidates.append(float(plateau["true_omega2"].iloc[0]))

    if not candidates:
        return None
    lower = max(float(args.target_omega2) * 1.4, float(args.target_omega2) + 0.01)
    xmax = max(max(candidates) * margin, lower)
    return min(all_max, xmax)


def _summarize_and_plot(raw: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested_panels = [
        (
            "gene",
            args.gene_eval_observed,
            f"Gene pilot consistency (eval n={args.gene_eval_observed})",
            args.gene_fit_bin_width,
        ),
        (
            "protein",
            args.protein_eval_observed,
            f"Protein pilot consistency (eval n={args.protein_eval_observed})",
            args.protein_fit_bin_width,
        ),
        (
            "gene",
            args.eval_extrapolate,
            f"Gene pilot extrapolation (eval n={args.eval_extrapolate})",
            args.gene_fit_bin_width,
        ),
        (
            "protein",
            args.eval_extrapolate,
            f"Protein pilot extrapolation (eval n={args.eval_extrapolate})",
            args.protein_fit_bin_width,
        ),
    ]
    panels = []
    seen_panel_keys: set[tuple[str, int]] = set()
    for panel in requested_panels:
        key = (panel[0], int(panel[1]))
        has_data = bool(
            (
                raw["workflow"].eq(panel[0])
                & raw["eval_n"].eq(int(panel[1]))
            ).any()
        )
        if key in seen_panel_keys or not has_data:
            continue
        seen_panel_keys.add(key)
        panels.append(panel)
    if not panels:
        raise ValueError("No requested sensitivity panels matched the computed raw points.")
    ncols = 1 if len(panels) == 1 else 2
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(7.6 * ncols, 4.0 * nrows),
        squeeze=False,
    )
    fit_rows: List[Dict[str, Any]] = []
    curve_rows: List[Dict[str, Any]] = []
    overlap_rows: List[Dict[str, Any]] = []

    for ax, (workflow, eval_n, title, bin_width) in zip(axes.flat, panels):
        sub_panel = raw[(raw["workflow"].eq(workflow)) & (raw["eval_n"].eq(int(eval_n)))].copy()
        pilots = sorted(sub_panel["pilot_n"].unique())
        cmap = cm.viridis
        norm = plt.Normalize(min(pilots), max(pilots)) if pilots else plt.Normalize(0, 1)
        fitted_by_pilot: Dict[int, Dict[str, Any]] = {}
        for pilot_n in pilots:
            sub = sub_panel[sub_panel["pilot_n"].eq(pilot_n)].sort_values("true_omega2")
            all_sub = raw[(raw["workflow"].eq(workflow)) & (raw["pilot_n"].eq(pilot_n))].copy()
            color = cmap(norm(pilot_n))
            ax.scatter(sub["true_omega2"], sub["power"], s=25, alpha=0.45, color=color, edgecolors="none")
            if str(getattr(args, "fit_mode", "per-panel")).strip().lower() == "joint-n":
                fit = _fit_joint_n(all_sub, sub, int(eval_n), args.target_omega2, bin_width)
            else:
                fit = _fit_one(sub, args.target_omega2, bin_width)
            fitted_by_pilot[int(pilot_n)] = fit
            fit_rows.append(
                {
                    "workflow": workflow,
                    "eval_n": int(eval_n),
                    "pilot_n": int(pilot_n),
                    "fit_status": fit.get("fit_status"),
                    "power_at_target_omega2": fit.get("power_at_target"),
                    "target_omega2": args.target_omega2,
                    "joint_eval_n_count": fit.get("joint_eval_n_count"),
                    "joint_point_count": fit.get("joint_point_count"),
                }
            )
            if "x_grid" in fit:
                label = f"pilot n={pilot_n}"
                if fit.get("fit_status") == "joint_n":
                    label += " (joint-n)"
                ax.plot(fit["x_grid"], fit["y_grid"], color=color, lw=2.0, label=label)
                for xval, yval in zip(fit["x_grid"], fit["y_grid"]):
                    curve_rows.append(
                        {
                            "workflow": workflow,
                            "eval_n": int(eval_n),
                            "pilot_n": int(pilot_n),
                            "true_omega2": float(xval),
                            "fitted_power": float(yval),
                        }
                    )
            else:
                ax.plot(sub["true_omega2"], sub["power"], color=color, lw=1.2, label=f"pilot n={pilot_n}")

        if pilots:
            ref = max(pilots)
            ref_fit = fitted_by_pilot.get(int(ref), {})
            if "x_grid" in ref_fit:
                ref_x = ref_fit["x_grid"]
                ref_y = ref_fit["y_grid"]
                for pilot_n in pilots:
                    fit = fitted_by_pilot.get(int(pilot_n), {})
                    if pilot_n == ref or "x_grid" not in fit:
                        continue
                    xmax = min(float(ref_x.max()), float(fit["x_grid"].max()), float(args.target_omega2) * 1.4)
                    grid = np.linspace(0.0, max(xmax, args.target_omega2), 250)
                    left = np.interp(grid, fit["x_grid"], fit["y_grid"])
                    right = np.interp(grid, ref_x, ref_y)
                    overlap_rows.append(
                        {
                            "workflow": workflow,
                            "eval_n": int(eval_n),
                            "pilot_n": int(pilot_n),
                            "reference_pilot_n": int(ref),
                            "mean_abs_power_diff": float(np.mean(np.abs(left - right))),
                            "max_abs_power_diff": float(np.max(np.abs(left - right))),
                            "power_at_target_diff": float(
                                fit.get("power_at_target", np.nan) - ref_fit.get("power_at_target", np.nan)
                            ),
                        }
                    )

        ax.axvline(args.target_omega2, color="red", ls="--", lw=1.0, alpha=0.7)
        ax.axhline(args.target_power, color="gray", ls=":", lw=1.0)
        ax.set_title(title)
        ax.set_xlabel(r"realized $\omega^2$")
        ax.set_ylabel("power")
        xmax = _adaptive_transition_xmax(sub_panel, args)
        if xmax is not None and xmax > 0:
            ax.set_xlim(0, xmax)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.22)
        if pilots:
            ax.legend(frameon=False, fontsize=9)

    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out / "sensitivity_pilot_curves.png", dpi=260)
    fig.savefig(args.out / "sensitivity_pilot_curves.pdf")
    plt.close(fig)
    return pd.DataFrame(fit_rows), pd.DataFrame(overlap_rows)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pilot-size sensitivity analysis for the raw-pool estimator.")
    parser.add_argument("--target-omega2", type=float, default=0.05)
    parser.add_argument("--target-power", type=float, default=0.8)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--boot-number", type=int, default=100)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--pool-size-per-group", type=int, default=1000)
    parser.add_argument("--omega2-floor", type=float, default=0.0)
    parser.add_argument(
        "--x-axis-mode",
        type=str,
        default="auto",
        choices=["auto", "full"],
        help="Use 'auto' to zoom each panel to the transition region; use 'full' to show all omega2 points.",
    )
    parser.add_argument("--x-axis-saturation-threshold", type=float, default=0.98)
    parser.add_argument("--x-axis-margin", type=float, default=1.15)
    parser.add_argument(
        "--fit-mode",
        type=str,
        default="per-panel",
        choices=["per-panel", "joint-n"],
        help="per-panel fits each displayed sample size independently; joint-n fits power=f(omega2,n) using all evaluated sample sizes.",
    )
    parser.add_argument("--random-seed", type=int, default=20260614)
    parser.add_argument(
        "--scenario-n-jobs",
        type=int,
        default=1,
        help="Parallel workers for raw-pool scenario precomputation.",
    )
    parser.add_argument("--workflow", choices=["both", "gene", "protein"], default="both")
    parser.add_argument("--out", type=Path, default=ROOT / "_pilot_sensitivity")

    parser.add_argument("--gene-table", type=Path, default=base_core.DATAGENE_DIR / "table.csv")
    parser.add_argument("--gene-tree", type=Path, default=base_core.DATAGENE_DIR / "rooted-tree.nwk")
    parser.add_argument("--gene-taxonomy", type=Path, default=base_core.DATAGENE_DIR / "taxonomy.csv")
    parser.add_argument("--gene-group", type=Path, default=base_core.DATAGENE_DIR / "group.csv")
    parser.add_argument("--gene-pilot-ns", type=str, default="4,7,10")
    parser.add_argument("--gene-eval-observed", type=int, default=10)
    parser.add_argument(
        "--gene-eval-ns",
        type=str,
        default=None,
        help="Comma list of gene eval sample sizes to compute for joint-n fitting, e.g. 10,20,30,40,50,60,70,80.",
    )
    parser.add_argument("--gene-fit-bin-width", type=float, default=0.008)
    parser.add_argument("--qiime-env", type=str, default="qiime2-metagenome-2024.10")
    parser.add_argument("--pcam-gene-blocks", type=str, default="auto")
    parser.add_argument("--pcam-ndon", type=int, default=1)
    parser.add_argument("--gene-grid", type=str, default="auto", choices=["auto", "omega-uniform", "fixed"])
    parser.add_argument("--gene-power-points", type=int, default=12)
    parser.add_argument("--gene-near-zero-points", type=int, default=4)
    parser.add_argument("--gene-near-zero-omega-max", type=str, default="auto")
    parser.add_argument("--gene-pi-candidates", type=int, default=17)
    parser.add_argument("--gene-scale-candidates", type=int, default=6)
    parser.add_argument("--gene-scale-max", type=float, default=1.7)
    parser.add_argument("--gene-scale-extend-max", type=float, default=3.0)

    parser.add_argument("--protein-table", type=Path, default=base_core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv")
    parser.add_argument("--protein-tree", type=Path, default=base_core.DATAPRO_DIR / "rooted-tree.nwk")
    parser.add_argument("--protein-group", type=Path, default=base_core.DATAPRO_DIR / "group.csv")
    parser.add_argument("--protein-pilot-ns", type=str, default="7,10,17")
    parser.add_argument("--protein-eval-observed", type=int, default=17)
    parser.add_argument(
        "--protein-eval-ns",
        type=str,
        default=None,
        help="Comma list of protein eval sample sizes to compute for joint-n fitting.",
    )
    parser.add_argument("--protein-fit-bin-width", type=float, default=0.003)
    parser.add_argument(
        "--protein-strengths",
        type=str,
        default="auto",
        help="Protein MDC-TF-MC strengths. Use 'auto' for adaptive omega-uniform selection.",
    )
    parser.add_argument("--mdctf-strength-candidates", type=int, default=15)
    parser.add_argument("--mdctf-strength-max", type=float, default=4.0)
    parser.add_argument("--mdctf-power-points", type=int, default=12)
    parser.add_argument("--mdctf-plateau-points", type=int, default=5)
    parser.add_argument("--mdctf-preview-pool-size", type=int, default=180)
    parser.add_argument("--mdctf-refine-target-points", type=int, default=0)
    parser.add_argument("--power-preview-boot-number", type=int, default=20)
    parser.add_argument("--power-preview-permutations", type=int, default=49)
    parser.add_argument("--edge-fraction", type=str, default="auto")
    parser.add_argument("--marginal-strength", type=str, default="auto")
    parser.add_argument("--eb-k", type=str, default="auto")
    parser.add_argument("--eval-extrapolate", type=int, default=80)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = create_parser().parse_args(list(argv) if argv is not None else None)
    args.out.mkdir(parents=True, exist_ok=True)
    args.edge_fraction = paper_core._resolve_auto_edge_fraction(args.edge_fraction)
    base_core.load_core_runtime()
    frames: List[pd.DataFrame] = []
    evals_gene = (
        _unique_ints(_parse_ints(args.gene_eval_ns))
        if args.gene_eval_ns
        else _unique_ints([args.gene_eval_observed, args.eval_extrapolate])
    )
    evals_protein = (
        _unique_ints(_parse_ints(args.protein_eval_ns))
        if args.protein_eval_ns
        else _unique_ints([args.protein_eval_observed, args.eval_extrapolate])
    )
    if args.workflow in {"both", "gene"}:
        frames.append(_run_gene(args, evals_gene))
    if args.workflow in {"both", "protein"}:
        frames.append(_run_protein(args, evals_protein))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(args.out / "sensitivity_raw_points.csv", index=False)
    fits, overlap = _summarize_and_plot(raw, args)
    fits.to_csv(args.out / "sensitivity_curve_fits.csv", index=False)
    overlap.to_csv(args.out / "sensitivity_overlap_summary.csv", index=False)
    summary = {
        "target_omega2": args.target_omega2,
        "target_power": args.target_power,
        "alpha": args.alpha,
        "boot_number": args.boot_number,
        "permutations": args.permutations,
        "pool_size_per_group": args.pool_size_per_group,
        "x_axis_mode": args.x_axis_mode,
        "x_axis_saturation_threshold": args.x_axis_saturation_threshold,
        "x_axis_margin": args.x_axis_margin,
        "fit_mode": args.fit_mode,
        "gene_pilot_ns": _parse_ints(args.gene_pilot_ns),
        "protein_pilot_ns": _parse_ints(args.protein_pilot_ns),
        "gene_eval_ns": evals_gene,
        "gene_grid": args.gene_grid,
        "gene_power_points": args.gene_power_points,
        "gene_near_zero_points": args.gene_near_zero_points,
        "gene_near_zero_omega_max": args.gene_near_zero_omega_max,
        "gene_pi_candidates": args.gene_pi_candidates,
        "gene_scale_candidates": args.gene_scale_candidates,
        "gene_scale_max": args.gene_scale_max,
        "gene_scale_extend_max": args.gene_scale_extend_max,
        "protein_eval_ns": evals_protein,
        "protein_strengths": args.protein_strengths
        if str(args.protein_strengths).strip().lower() in {"auto", "power-uniform"}
        else _parse_floats(args.protein_strengths),
        "edge_fraction": args.edge_fraction,
        "marginal_strength": args.marginal_strength,
        "eb_k": args.eb_k,
        "mdctf_strength_candidates": args.mdctf_strength_candidates,
        "mdctf_strength_max": args.mdctf_strength_max,
        "mdctf_power_points": args.mdctf_power_points,
        "mdctf_plateau_points": args.mdctf_plateau_points,
        "mdctf_preview_pool_size": args.mdctf_preview_pool_size,
        "mdctf_refine_target_points": args.mdctf_refine_target_points,
        "power_preview_boot_number": args.power_preview_boot_number,
        "power_preview_permutations": args.power_preview_permutations,
    }
    (args.out / "sensitivity_summary.json").write_text(
        json.dumps(_json_safe(summary), indent=2),
        encoding="utf-8",
    )
    print(args.out)
    print(args.out / "sensitivity_pilot_curves.png")


if __name__ == "__main__":
    main()
