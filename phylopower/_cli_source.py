#!/usr/bin/env python3
# Maintainable source embedded into the generated single-file runner.
"""Raw-pool minimum sample-size estimator.

For each candidate per-group sample size, this module builds synthetic raw
feature pools, recomputes modality-specific distances, fits a monotone power
curve over realized omega squared, and reports the smallest sample size whose
fitted power reaches the requested target.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import core as base_core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_MPL_CACHE = Path(tempfile.gettempdir()) / "phylopower-matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))

from _fig4_curve_plotting import (  # noqa: E402
    binned_monotone,
    fit_binned_null_hill,
    null_started_hill_sigmoid,
)
from _protein_mdctf_mc import mdctf_mc_pool  # noqa: E402
from semisynthetic_power import (  # noqa: E402
    _read_group_map,
    _read_protein_long_table,
    _read_taxon_feature_table,
)
import gene_power_workflow as gene_wf  # noqa: E402
import protein_power_workflow as protein_wf  # noqa: E402


DEFAULT_COARSE_STEP = 10
DEFAULT_STABILITY_WINDOW = 1
DEFAULT_SCENARIO_POINTS = 15
DEFAULT_SCENARIO_N_JOBS = 4


def _resolve_auto_edge_fraction(value: float | str) -> float:
    """Resolve MDC-TF-MC edge-fraction defaults."""
    if str(value).strip().lower() == "auto":
        return 1.0
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("edge_fraction must be in [0, 1] or 'auto'.")
    return resolved


@dataclass
class CurveEvaluation:
    n_per_group: int
    search_stage: str
    fit_status: str
    fit_method: str
    fitted_power_at_target_omega2: float
    required_omega2_for_target_power: float
    curve_reaches_target_power: bool
    target_omega2_bracketed: bool
    near_zero_omega2_max: float
    near_zero_point_count: int
    below_target_point_count: int
    above_target_point_count: int
    nearest_omega2_distance: float
    low_omega_support_warning: bool
    target_omega2: float
    target_power: float
    qualifies: bool
    h: Optional[float] = None
    x0: Optional[float] = None
    floor: Optional[float] = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(float(value)):
            return None
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def _fit_curve_at_target(
    df: pd.DataFrame,
    *,
    target_omega2: float,
    target_power: float,
    alpha: float,
    fit_bin_width: float,
) -> Dict[str, Any]:
    valid = df[["true_omega2", "power"]].dropna().copy()
    if valid.empty:
        return {
            "fit_status": "no_valid_points",
            "fit_method": "none",
            "fitted_power_at_target_omega2": np.nan,
            "required_omega2_for_target_power": np.nan,
            "curve_reaches_target_power": False,
            "target_omega2_bracketed": False,
            "near_zero_omega2_max": np.nan,
            "near_zero_point_count": 0,
            "below_target_point_count": 0,
            "above_target_point_count": 0,
            "nearest_omega2_distance": np.nan,
            "low_omega_support_warning": True,
            "h": None,
            "x0": None,
            "floor": None,
        }

    valid["true_omega2"] = valid["true_omega2"].clip(lower=0.0)
    valid["power"] = valid["power"].clip(0.0, 1.0)
    valid = valid.sort_values("true_omega2").reset_index(drop=True)
    binned = binned_monotone(valid, fit_bin_width)
    max_observed_omega2 = float(valid["true_omega2"].max())
    if np.isfinite(float(target_omega2)) and float(target_omega2) > 0:
        near_zero_hi = max(float(target_omega2) * 2.0, float(target_omega2) + 0.02)
        near_zero_hi = min(float(near_zero_hi), max(0.06, 0.15 * max_observed_omega2))
    else:
        near_zero_hi = float(valid["true_omega2"].quantile(0.25))
    near_zero_count = int((valid["true_omega2"] <= near_zero_hi).sum())
    below_target_count = int((valid["true_omega2"] <= float(target_omega2)).sum())
    above_target_count = int((valid["true_omega2"] >= float(target_omega2)).sum())
    nearest_omega2_distance = float(np.min(np.abs(valid["true_omega2"].to_numpy(dtype=float) - float(target_omega2))))
    target_bracketed = bool(
        len(binned)
        and float(binned["true_omega2"].min()) <= target_omega2 <= float(binned["true_omega2"].max())
    )
    support_warning = (
        (not target_bracketed)
        or near_zero_count < 3
        or below_target_count < 2
        or above_target_count < 1
    )

    x_max = max(
        float(target_omega2) * 2.0,
        float(valid["true_omega2"].max()) * 2.0,
        0.05,
    )
    x_grid = np.linspace(0.0, x_max, 600)
    _, params = fit_binned_null_hill(binned, x_grid)

    if params is not None:
        power_at_target = float(
            null_started_hill_sigmoid(
                np.array([target_omega2]),
                params["h"],
                params["x0"],
                params["floor"],
            )[0]
        )
        curve_values = null_started_hill_sigmoid(x_grid, params["h"], params["x0"], params["floor"])
        reaches = bool(np.nanmax(curve_values) >= target_power)
        if reaches:
            hit_idx = int(np.argmax(curve_values >= target_power))
            required = float(x_grid[hit_idx])
        else:
            required = np.nan
        return {
            "fit_status": "ok",
            "fit_method": "binned_null_started_hill",
            "fitted_power_at_target_omega2": power_at_target,
            "required_omega2_for_target_power": required,
            "curve_reaches_target_power": reaches,
            "target_omega2_bracketed": target_bracketed,
            "near_zero_omega2_max": float(near_zero_hi),
            "near_zero_point_count": near_zero_count,
            "below_target_point_count": below_target_count,
            "above_target_point_count": above_target_count,
            "nearest_omega2_distance": nearest_omega2_distance,
            "low_omega_support_warning": bool(support_warning),
            "h": float(params["h"]),
            "x0": float(params["x0"]),
            "floor": float(params["floor"]),
        }

    if len(binned) >= 2 and binned["true_omega2"].nunique() >= 2:
        bx = binned["true_omega2"].to_numpy(dtype=float)
        by = binned["power_mono"].to_numpy(dtype=float)
        power_at_target = float(np.interp(target_omega2, bx, by, left=by[0], right=by[-1]))
        hit = np.flatnonzero(by >= target_power)
        required = float(bx[hit[0]]) if len(hit) else np.nan
        return {
            "fit_status": "fallback_monotone_interpolation",
            "fit_method": "binned_monotone_interpolation",
            "fitted_power_at_target_omega2": power_at_target,
            "required_omega2_for_target_power": required,
            "curve_reaches_target_power": bool(len(hit)),
            "target_omega2_bracketed": target_bracketed,
            "near_zero_omega2_max": float(near_zero_hi),
            "near_zero_point_count": near_zero_count,
            "below_target_point_count": below_target_count,
            "above_target_point_count": above_target_count,
            "nearest_omega2_distance": nearest_omega2_distance,
            "low_omega_support_warning": bool(support_warning),
            "h": None,
            "x0": None,
            "floor": None,
        }

    return {
        "fit_status": "insufficient_unique_points",
        "fit_method": "none",
        "fitted_power_at_target_omega2": np.nan,
        "required_omega2_for_target_power": np.nan,
        "curve_reaches_target_power": False,
        "target_omega2_bracketed": target_bracketed,
        "near_zero_omega2_max": float(near_zero_hi),
        "near_zero_point_count": near_zero_count,
        "below_target_point_count": below_target_count,
        "above_target_point_count": above_target_count,
        "nearest_omega2_distance": nearest_omega2_distance,
        "low_omega_support_warning": bool(support_warning),
        "h": None,
        "x0": None,
        "floor": None,
    }


def _default_max_n(observed_n: int, max_n: Optional[int]) -> int:
    if max_n is not None:
        return int(max_n)
    return max(int(observed_n) * 3, int(observed_n) + 20)


def _candidate_ns(min_n: int, max_n: int, coarse_step: int) -> List[int]:
    coarse_start = ((min_n + coarse_step - 1) // coarse_step) * coarse_step
    return sorted(set([min_n, max_n] + list(range(coarse_start, max_n + 1, coarse_step))))


def _limit_sorted_values(values: List[float], count: int) -> List[float]:
    count = max(1, int(count))
    if len(values) <= count:
        return values
    indices = np.rint(np.linspace(0, len(values) - 1, count)).astype(int).tolist()
    selected: List[int] = []
    for idx in indices:
        idx = int(np.clip(idx, 0, len(values) - 1))
        if idx not in selected:
            selected.append(idx)
    cursor = 0
    while len(selected) < count and cursor < len(values):
        if cursor not in selected:
            selected.append(cursor)
        cursor += 1
    return [values[i] for i in sorted(selected[:count])]


def _search_minimum_n(
    *,
    min_n: int,
    max_n: int,
    target_power: float,
    target_omega2: float,
    alpha: float,
    fit_bin_width: float,
    coarse_step: int,
    stability_window: int,
    evaluate_curve_fn: Callable[[int, str], pd.DataFrame],
) -> Tuple[Optional[int], pd.DataFrame, pd.DataFrame]:
    evaluated: Dict[int, CurveEvaluation] = {}
    scenario_frames: List[pd.DataFrame] = []

    def evaluate(n_per_group: int, stage: str) -> bool:
        if n_per_group in evaluated:
            return bool(evaluated[n_per_group].qualifies)
        scenario_df = evaluate_curve_fn(n_per_group, stage).copy()
        scenario_df["n_per_group"] = int(n_per_group)
        scenario_df["search_stage"] = stage
        scenario_frames.append(scenario_df)
        fit = _fit_curve_at_target(
            scenario_df,
            target_omega2=target_omega2,
            target_power=target_power,
            alpha=alpha,
            fit_bin_width=fit_bin_width,
        )
        fitted_power = float(fit["fitted_power_at_target_omega2"])
        # An unbracketed target is evaluated by curve extrapolation only; never
        # let it qualify, so an out-of-grid target cannot produce a confident
        # minimum n.
        qualifies = bool(
            fit["target_omega2_bracketed"]
            and np.isfinite(fitted_power)
            and fitted_power >= target_power
        )
        evaluated[n_per_group] = CurveEvaluation(
            n_per_group=int(n_per_group),
            search_stage=stage,
            fit_status=str(fit["fit_status"]),
            fit_method=str(fit["fit_method"]),
            fitted_power_at_target_omega2=fitted_power,
            required_omega2_for_target_power=float(fit["required_omega2_for_target_power"])
            if np.isfinite(fit["required_omega2_for_target_power"])
            else np.nan,
            curve_reaches_target_power=bool(fit["curve_reaches_target_power"]),
            target_omega2_bracketed=bool(fit["target_omega2_bracketed"]),
            near_zero_omega2_max=float(fit["near_zero_omega2_max"])
            if np.isfinite(fit["near_zero_omega2_max"])
            else np.nan,
            near_zero_point_count=int(fit["near_zero_point_count"]),
            below_target_point_count=int(fit["below_target_point_count"]),
            above_target_point_count=int(fit["above_target_point_count"]),
            nearest_omega2_distance=float(fit["nearest_omega2_distance"])
            if np.isfinite(fit["nearest_omega2_distance"])
            else np.nan,
            low_omega_support_warning=bool(fit["low_omega_support_warning"]),
            target_omega2=float(target_omega2),
            target_power=float(target_power),
            qualifies=qualifies,
            h=fit["h"],
            x0=fit["x0"],
            floor=fit["floor"],
        )
        return qualifies

    previous: Optional[int] = None
    coarse_hit: Optional[int] = None
    for candidate in _candidate_ns(min_n, max_n, coarse_step):
        if evaluate(candidate, "coarse"):
            coarse_hit = candidate
            break
        previous = candidate

    minimum_n: Optional[int] = None
    if coarse_hit is not None:
        fine_start = min_n if previous is None else previous + 1
        for candidate in range(fine_start, coarse_hit + 1):
            window = list(range(candidate, candidate + max(1, stability_window)))
            if window[-1] > max_n:
                break
            if all(evaluate(n, "fine" if n <= coarse_hit else "fine_stability") for n in window):
                minimum_n = candidate
                break

    if minimum_n is not None:
        for candidate in (minimum_n - 1, minimum_n, minimum_n + 1):
            if min_n <= candidate <= max_n and candidate not in evaluated:
                evaluate(candidate, "plot_context")

    power_df = pd.DataFrame([asdict(evaluated[n]) for n in sorted(evaluated)])
    metrics_df = pd.concat(scenario_frames, ignore_index=True) if scenario_frames else pd.DataFrame()

    # Warn regardless of whether a minimum n was found: an unbracketed target
    # means every fit at the target was extrapolated, which the caller must see
    # even when no n qualifies.
    if evaluated and any(not ev.target_omega2_bracketed for ev in evaluated.values()):
        print(
            f"\n  WARNING: target ω²={float(target_omega2):.5f} falls outside the "
            f"simulated range [{metrics_df['true_omega2'].min():.4f}, "
            f"{metrics_df['true_omega2'].max():.4f}]. "
            f"Power at target ω² is EXTRAPOLATED and is not allowed to qualify, "
            f"so no minimum n is reported. Increase "
            f"--pcam-scale-extend-max (gene) or --mdctf-strength-max (protein) "
            f"to cover the target effect size.\n",
            flush=True,
        )

    return minimum_n, power_df, metrics_df


def _curve_support_flags(power_df: pd.DataFrame) -> Dict[str, bool]:
    """Aggregate the per-n target-omega2 support flags for summary.json."""
    if power_df.empty:
        return {"target_omega2_bracketed": False, "low_omega_support_warning": True}
    return {
        "target_omega2_bracketed": bool(power_df["target_omega2_bracketed"].all()),
        "low_omega_support_warning": bool(power_df["low_omega_support_warning"].any()),
    }


def _with_power_uncertainty(metrics_df: pd.DataFrame, boot_number: int) -> pd.DataFrame:
    """Attach per-scenario Monte Carlo uncertainty for the power estimate.

    Each scenario's power is a binomial proportion over ``boot_number``
    bootstrap replicates, so its Monte Carlo standard error is
    sqrt(p(1-p)/B) and a 95% Wilson interval gives a robust confidence range.
    """
    if metrics_df.empty or "power" not in metrics_df.columns or int(boot_number) <= 0:
        return metrics_df
    p = metrics_df["power"].astype(float).clip(0.0, 1.0)
    n = float(int(boot_number))
    z = 1.959964
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    out = metrics_df.copy()
    out["power_mcse"] = np.sqrt(p * (1.0 - p) / n)
    out["power_wilson95_lower"] = (centre - half).clip(0.0, 1.0)
    out["power_wilson95_upper"] = (centre + half).clip(0.0, 1.0)
    return out


def _write_outputs(
    *,
    out: Path,
    summary: Dict[str, Any],
    power_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    power_df.to_csv(out / "power_by_sample_size.csv", index=False)
    metrics_df.to_csv(out / "scenario_metrics_by_sample_size.csv", index=False)
    (out / "summary.json").write_text(json.dumps(_json_safe(summary), indent=2), encoding="utf-8")
    _write_sample_size_decision_plot(out=out, summary=summary, power_df=power_df, metrics_df=metrics_df)


def _write_sample_size_decision_plot(
    *,
    out: Path,
    summary: Dict[str, Any],
    power_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    if power_df.empty or metrics_df.empty:
        return

    def finite_float(value: Any) -> Optional[float]:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return None
        return out if np.isfinite(out) else None

    mpl_cache = Path(tempfile.gettempdir()) / "phylopower-matplotlib"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    minimum_n = summary.get("minimum_n_per_group")
    if minimum_n is not None:
        minimum_n_int = int(minimum_n)
        curve_df = power_df.loc[power_df["n_per_group"].astype(int) == minimum_n_int].copy()
    else:
        minimum_n_int = None
        curve_df = power_df.sort_values("n_per_group").tail(1).copy()
    curve_df = curve_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["n_per_group"])
    if curve_df.empty:
        return
    curve_df = curve_df.sort_values("n_per_group")
    display_ns = [int(n) for n in curve_df["n_per_group"].tolist()]

    target_power = float(summary["target_power"])
    target_omega2 = float(summary["target_omega2"])
    workflow = str(summary.get("workflow", "workflow"))

    metrics_plot = metrics_df.loc[metrics_df["n_per_group"].isin(display_ns)].copy()
    metrics_plot = metrics_plot.replace([np.inf, -np.inf], np.nan).dropna(subset=["true_omega2", "power"])
    if metrics_plot.empty:
        return
    x_max = max(float(metrics_plot["true_omega2"].max()), target_omega2)
    x_grid = np.linspace(0.0, max(x_max * 1.05, 1e-6), 300)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))

    for _, row in curve_df.iterrows():
        n_value = int(row["n_per_group"])
        data = metrics_plot.loc[metrics_plot["n_per_group"] == n_value].sort_values("true_omega2")
        if data.empty:
            continue
        is_minimum = minimum_n_int is not None and n_value == minimum_n_int
        color = "#2F855A" if is_minimum else "#2B6CB0"
        linewidth = 3.0
        h = finite_float(row.get("h", np.nan))
        x0 = finite_float(row.get("x0", np.nan))
        floor = finite_float(row.get("floor", np.nan))
        if h is not None and x0 is not None and floor is not None:
            y_grid = null_started_hill_sigmoid(x_grid, h, x0, floor)
        else:
            grouped = data.groupby("true_omega2", as_index=False)["power"].mean().sort_values("true_omega2")
            y_grid = np.interp(
                x_grid,
                grouped["true_omega2"].to_numpy(dtype=float),
                grouped["power"].to_numpy(dtype=float),
                left=float(grouped["power"].iloc[0]),
                right=float(grouped["power"].iloc[-1]),
            )
        ax.plot(
            x_grid,
            y_grid,
            color=color,
            linewidth=linewidth,
            label=f"minimum n = {n_value}" if is_minimum else None,
            zorder=3,
        )
        ax.scatter(
            data["true_omega2"],
            data["power"],
            s=20,
            color=color,
            alpha=0.38,
            edgecolors="none",
            label="scenario points" if is_minimum else None,
            zorder=2,
        )

    ax.axhline(
        target_power,
        color="#C53030",
        linestyle="--",
        linewidth=1.5,
        label=f"target power = {target_power:.2f}",
    )
    ax.axvline(
        target_omega2,
        color="#C53030",
        linestyle=":",
        linewidth=1.6,
        label=f"target $\\omega^2$ = {target_omega2:.4g}",
    )

    if minimum_n_int is not None:
        match = curve_df.loc[curve_df["n_per_group"] == minimum_n_int]
        y = (
            float(match["fitted_power_at_target_omega2"].iloc[0])
            if not match.empty and np.isfinite(float(match["fitted_power_at_target_omega2"].iloc[0]))
            else target_power
        )
        ax.scatter([target_omega2], [y], s=80, color="#2F855A", zorder=5)
        x_text_offset = -12 if target_omega2 > 0.72 * max(x_grid) else 8
        x_text_align = "right" if x_text_offset < 0 else "left"
        ax.annotate(
            f"minimum n = {minimum_n_int}",
            xy=(target_omega2, y),
            xytext=(x_text_offset, 10),
            textcoords="offset points",
            fontsize=10,
            color="#22543D",
            ha=x_text_align,
        )
    else:
        ax.text(
            0.98,
            0.08,
            "target not reached",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#C53030",
        )

    ax.text(
        0.02,
        0.96,
        f"target power = {target_power:.2f}\ntarget $\\omega^2$ = {target_omega2:.4g}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#CBD5E0", "alpha": 0.85},
    )
    ax.set_title(f"{workflow.capitalize()} $\\omega^2$–power curves")
    ax.set_xlabel("Realized $\\omega^2$")
    ax.set_ylabel("Power")
    ax.set_xlim(0.0, max(x_grid))
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "sample_size_decision.png", dpi=300)
    plt.close(fig)


def _effective_tree_path_for_raw_workflow(
    *,
    tree: Path,
    out: Path,
    random_seed: int,
    sigma: float,
    nni_prob: float,
    support_threshold: Optional[float],
    stem: str,
) -> Path:
    if sigma <= 0.0 and nni_prob <= 0.0:
        return tree
    from skbio import TreeNode

    out.mkdir(parents=True, exist_ok=True)
    base_tree = TreeNode.read(str(tree))
    perturbed = base_core.materialize_perturbed_tree(
        base_tree=base_tree,
        sigma=float(sigma),
        nni_prob=float(nni_prob),
        random_seed=int(random_seed),
        out_dir=out,
        output_stem=stem,
        output_format="newick",
        support_threshold=support_threshold,
    )
    return Path(perturbed) if perturbed is not None else tree


def _compute_observed_pilot_omega2(pilot: dict) -> float:
    """Observed effect size of the resolved pilot under the active distance engine."""
    dm, group_map = gene_wf.P.real_distance(pilot)
    return max(0.0, float(base_core.compute_omega2(dm, group_map)))


def _resolve_target_omega2(
    requested_target_omega2: Optional[float],
    observed_pilot_omega2: float,
) -> tuple[float, str]:
    if requested_target_omega2 is None:
        return float(observed_pilot_omega2), "observed_pilot"
    return float(requested_target_omega2), "user"


def _gene_args(
    *,
    table: Path,
    tree: Path,
    taxonomy: Path,
    group: Path,
    out: Path,
    target_power: float,
    target_omega2: Optional[float],
    qiime_env: str,
    alpha: float,
    pool_size_per_group: int,
    boot_number: int,
    permutations: int,
    pcam_gene_blocks: int | str,
    pcam_ndon: int,
    pcam_grid: str,
    pcam_power_points: int,
    pcam_near_zero_points: int,
    pcam_near_zero_omega_max: str,
    pcam_pi_candidates: int,
    pcam_scale_candidates: int,
    pcam_scale_max: float,
    pcam_scale_extend_max: float,
    fit_bin_width: float,
    omega2_floor: float,
    scenario_n_jobs: int,
    random_seed: int,
    tree_jitter_sigma: float,
    tree_nni_prob: float,
    tree_support_threshold: Optional[float],
    use_phylogeny: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        table=table,
        tree=tree,
        taxonomy=taxonomy,
        group=group,
        out=out,
        target_power=target_power,
        target_omega2=target_omega2,
        qiime_env=qiime_env,
        alpha=alpha,
        eval_n=None,
        pool_size_per_group=pool_size_per_group,
        boot_number=boot_number,
        permutations=permutations,
        pcam_gene_blocks=pcam_gene_blocks,
        pcam_ndon=pcam_ndon,
        pcam_grid=pcam_grid,
        pcam_power_points=pcam_power_points,
        pcam_near_zero_points=pcam_near_zero_points,
        pcam_near_zero_omega_max=pcam_near_zero_omega_max,
        pcam_pi_candidates=pcam_pi_candidates,
        pcam_scale_candidates=pcam_scale_candidates,
        pcam_scale_max=pcam_scale_max,
        pcam_scale_extend_max=pcam_scale_extend_max,
        fit_bin_width=fit_bin_width,
        omega2_floor=omega2_floor,
        scenario_n_jobs=scenario_n_jobs,
        random_seed=random_seed,
        tree_jitter_sigma=tree_jitter_sigma,
        tree_nni_prob=tree_nni_prob,
        tree_support_threshold=tree_support_threshold,
        use_phylogeny=use_phylogeny,
    )


def compute_gene_min_sample_size(
    *,
    table: Path | str = base_core.DATAGENE_DIR / "table.csv",
    tree: Path | str = base_core.DATAGENE_DIR / "rooted-tree.nwk",
    taxonomy: Path | str = base_core.DATAGENE_DIR / "taxonomy.csv",
    group: Path | str = base_core.DATAGENE_DIR / "group.csv",
    target_power: float,
    target_omega2: Optional[float] = None,
    out: Path | str = "gene_min_sample_size_output",
    pilot_n: Optional[int] = None,
    min_n: int = 2,
    max_n: Optional[int] = None,
    qiime_env: str = "qiime2-metagenome-2024.10",
    alpha: float = 0.05,
    pool_size_per_group: int = 1000,
    boot_number: int = 200,
    permutations: int = 199,
    pcam_gene_blocks: int | str = "auto",
    pcam_ndon: int = 1,
    pcam_grid: str = "auto",
    pcam_power_points: int = DEFAULT_SCENARIO_POINTS,
    pcam_near_zero_points: int = 4,
    pcam_near_zero_omega_max: str = "auto",
    pcam_pi_candidates: int = 17,
    pcam_scale_candidates: int = 6,
    pcam_scale_max: float = 1.7,
    pcam_scale_extend_max: float = 3.0,
    fit_bin_width: float = 0.003,
    omega2_floor: float = 0.0,
    scenario_n_jobs: int = DEFAULT_SCENARIO_N_JOBS,
    coarse_step: int = DEFAULT_COARSE_STEP,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
    random_seed: int = 20260614,
    tree_jitter_sigma: float = 0.0,
    tree_nni_prob: float = 0.0,
    tree_support_threshold: Optional[float] = None,
    use_phylogeny: bool = True,
) -> Dict[str, Any]:
    base_core.load_core_runtime()
    out_path = Path(out)
    args = _gene_args(
        table=Path(table),
        tree=Path(tree),
        taxonomy=Path(taxonomy),
        group=Path(group),
        out=out_path,
        target_power=target_power,
        target_omega2=target_omega2,
        qiime_env=qiime_env,
        alpha=alpha,
        pool_size_per_group=pool_size_per_group,
        boot_number=boot_number,
        permutations=permutations,
        pcam_gene_blocks=pcam_gene_blocks,
        pcam_ndon=pcam_ndon,
        pcam_grid=pcam_grid,
        pcam_power_points=pcam_power_points,
        pcam_near_zero_points=pcam_near_zero_points,
        pcam_near_zero_omega_max=pcam_near_zero_omega_max,
        pcam_pi_candidates=pcam_pi_candidates,
        pcam_scale_candidates=pcam_scale_candidates,
        pcam_scale_max=pcam_scale_max,
        pcam_scale_extend_max=pcam_scale_extend_max,
        fit_bin_width=fit_bin_width,
        omega2_floor=omega2_floor,
        scenario_n_jobs=scenario_n_jobs,
        random_seed=random_seed,
        tree_jitter_sigma=tree_jitter_sigma,
        tree_nni_prob=tree_nni_prob,
        tree_support_threshold=tree_support_threshold,
        use_phylogeny=use_phylogeny,
    )
    args.tree = _effective_tree_path_for_raw_workflow(
        tree=args.tree,
        out=out_path,
        random_seed=random_seed,
        sigma=tree_jitter_sigma,
        nni_prob=tree_nni_prob,
        support_threshold=tree_support_threshold,
        stem="gene_tree_perturbed",
    )
    group_map = _read_group_map(args.group)
    table_df, aligned_group_map = _read_taxon_feature_table(args.table, group_map)
    observed_n = int(aligned_group_map.value_counts().min())
    resolved_pilot_n = int(pilot_n if pilot_n is not None else observed_n)
    resolved_min_n = max(2, int(min_n))
    resolved_max_n = _default_max_n(observed_n, max_n)
    if resolved_max_n < resolved_min_n:
        raise ValueError(f"max_n ({resolved_max_n}) must be >= min_n ({resolved_min_n}).")
    raw_dict = gene_wf._build_gene_raw_dict(table_df, aligned_group_map, args)
    pilot_for_target, pilot_source_for_target = gene_wf._gene_pilot_raw_view(
        raw_dict,
        resolved_pilot_n,
        int(random_seed + resolved_pilot_n * 1009),
        args,
    )
    observed_pilot_omega2 = _compute_observed_pilot_omega2(pilot_for_target)
    resolved_target_omega2, target_omega2_source = _resolve_target_omega2(
        target_omega2,
        observed_pilot_omega2,
    )
    args.target_omega2 = resolved_target_omega2
    reusable_pool_size = _effective_reusable_pool_size(
        requested_pool_size=pool_size_per_group,
        max_n=resolved_max_n,
    )
    scenario_seed = int(random_seed + resolved_pilot_n * 1009)
    scenarios = gene_wf.precompute_pcam_scenarios(
        raw_dict,
        args,
        resolved_pilot_n,
        scenario_seed,
        pool_size_per_group=reusable_pool_size,
    )

    def evaluate_curve(n_per_group: int, stage: str) -> pd.DataFrame:
        args.eval_n = int(n_per_group)
        seed = int(random_seed + resolved_pilot_n * 1009 + n_per_group * 37)
        print(
            f"[gene raw-pool] {stage}: reusing {len(scenarios)} scenarios, "
            f"pilot_n={resolved_pilot_n}, eval_n={n_per_group}",
            flush=True,
        )
        return gene_wf.evaluate_precomputed_pcam_scenarios(
            scenarios,
            args,
            eval_n=n_per_group,
            seed=seed,
        )

    minimum_n, power_df, metrics_df = _search_minimum_n(
        min_n=resolved_min_n,
        max_n=resolved_max_n,
        target_power=target_power,
        target_omega2=resolved_target_omega2,
        alpha=alpha,
        fit_bin_width=fit_bin_width,
        coarse_step=coarse_step,
        stability_window=stability_window,
        evaluate_curve_fn=evaluate_curve,
    )
    summary = {
        "workflow": "gene",
        "engine": "raw_pool",
        "generator": "PCAM",
        "distance": "Gemelli",
        "minimum_n_per_group": minimum_n,
        "target_power": float(target_power),
        "target_omega2": float(resolved_target_omega2),
        "target_omega2_source": target_omega2_source,
        **_curve_support_flags(power_df),
        "observed_pilot_omega2": float(observed_pilot_omega2),
        "observed_pilot_source": pilot_source_for_target,
        "alpha": float(alpha),
        "observed_n_per_group": observed_n,
        "pilot_n_per_group": resolved_pilot_n,
        "sweep_min_n": resolved_min_n,
        "sweep_max_n": resolved_max_n,
        "coarse_step": int(coarse_step),
        "stability_window": int(stability_window),
        "pool_size_per_group": int(pool_size_per_group),
        "boot_number": int(boot_number),
        "permutations": int(permutations),
        "fit_bin_width": float(fit_bin_width),
        "scenario_n_jobs": int(scenario_n_jobs),
        "pcam_gene_blocks": args.pcam_gene_blocks,
        "pcam_grid": pcam_grid,
        "pcam_power_points": int(pcam_power_points),
        "pcam_near_zero_points": int(pcam_near_zero_points),
        "pcam_near_zero_omega_max": pcam_near_zero_omega_max,
        "pcam_pi_candidates": int(pcam_pi_candidates),
        "pcam_scale_candidates": int(pcam_scale_candidates),
        "pcam_scale_max": float(pcam_scale_max),
        "pcam_scale_extend_max": float(pcam_scale_extend_max),
        "tree_jitter_sigma": float(tree_jitter_sigma),
        "tree_nni_prob": float(tree_nni_prob),
    }
    metrics_df = _with_power_uncertainty(metrics_df, int(boot_number))
    _write_outputs(out=out_path, summary=summary, power_df=power_df, metrics_df=metrics_df)
    return {"summary": summary, "power_by_sample_size": power_df, "scenario_metrics_by_sample_size": metrics_df}


def _protein_args(
    *,
    table: Path,
    tree: Path,
    group: Path,
    out: Path,
    target_power: float,
    target_omega2: Optional[float],
    alpha: float,
    pool_size_per_group: int,
    boot_number: int,
    permutations: int,
    edge_fraction: float | str,
    marginal_strength: str,
    eb_k: str,
    mdctf_strengths: str,
    mdctf_strength_candidates: int,
    mdctf_strength_max: float,
    mdctf_power_points: int,
    mdctf_plateau_points: int,
    mdctf_preview_pool_size: int,
    mdctf_refine_target_points: int,
    power_preview_boot_number: int,
    power_preview_permutations: int,
    fit_bin_width: float,
    omega2_floor: float,
    scenario_n_jobs: int,
    random_seed: int,
    tree_jitter_sigma: float,
    tree_nni_prob: float,
    tree_support_threshold: Optional[float],
) -> SimpleNamespace:
    return SimpleNamespace(
        table=table,
        tree=tree,
        group=group,
        out=out,
        target_power=target_power,
        target_omega2=target_omega2,
        alpha=alpha,
        eval_n=None,
        pool_size_per_group=pool_size_per_group,
        boot_number=boot_number,
        permutations=permutations,
        edge_fraction=_resolve_auto_edge_fraction(edge_fraction),
        marginal_strength=marginal_strength,
        eb_k=eb_k,
        mdctf_strengths=mdctf_strengths,
        mdctf_strength_candidates=mdctf_strength_candidates,
        mdctf_strength_max=mdctf_strength_max,
        mdctf_power_points=mdctf_power_points,
        mdctf_plateau_points=mdctf_plateau_points,
        mdctf_preview_pool_size=mdctf_preview_pool_size,
        mdctf_refine_target_points=mdctf_refine_target_points,
        power_preview_boot_number=power_preview_boot_number,
        power_preview_permutations=power_preview_permutations,
        fit_bin_width=fit_bin_width,
        omega2_floor=omega2_floor,
        scenario_n_jobs=scenario_n_jobs,
        random_seed=random_seed,
        tree_jitter_sigma=tree_jitter_sigma,
        tree_nni_prob=tree_nni_prob,
        tree_support_threshold=tree_support_threshold,
        protein_transform="none",
        lowrank_rank=5,
        agg_clades=300,
    )


def _effective_reusable_pool_size(
    *,
    requested_pool_size: int,
    max_n: int,
    pool_min: int = 200,
    pool_multiplier: int = 10,
) -> int:
    dynamic_size = max(int(pool_min), int(pool_multiplier) * int(max_n), int(max_n))
    return max(int(max_n), min(int(requested_pool_size), dynamic_size))


def _mdctf_mc_power_uniform_strengths(
    pilot: dict,
    args: SimpleNamespace,
    eval_n: int,
    seed: int,
) -> List[float]:
    """Choose strength values using the same MDC-TF-MC generator used for final scenarios.

    The preview is used to map perturbation strength to realized omega^2, then
    formal scenarios are chosen to spread points across the observed
    realized-effect range for that pilot. If the default 0--1 strength range
    does not cover the target effect window, the preview scan is extended up to
    ``mdctf_strength_max``. Extra formal target-refinement points are disabled
    by default and only added when ``mdctf_refine_target_points`` is positive.
    """
    rows: List[Dict[str, float]] = []
    preview_boot = int(args.power_preview_boot_number)
    preview_perms = int(args.power_preview_permutations)
    preview_m = int(args.mdctf_preview_pool_size)
    scenario_n_jobs = max(1, int(getattr(args, "scenario_n_jobs", 1)))

    def preview_one(i: int, strength: float) -> Dict[str, float]:
        point_seed = seed + i * 7919
        table, group_map = mdctf_mc_pool(
            pilot,
            preview_m,
            point_seed,
            float(strength),
            edge_fraction=args.edge_fraction,
            marginal_strength=args.marginal_strength,
            eb_k=args.eb_k,
        )
        dm = protein_wf.P.recompute_distance(pilot, table)
        omega2 = (
            0.0
            if np.isclose(strength, 0.0)
            else max(0.0, float(base_core.compute_omega2(dm, group_map)))
        )
        if np.isclose(strength, 0.0):
            power = protein_wf._exchangeable_null_power(
                dm,
                eval_n=int(eval_n),
                boot=preview_boot,
                perms=preview_perms,
                seed=point_seed + 41,
            )
        else:
            metrics = protein_wf.summarize_distance_metrics_with_replacement(
                dm=dm,
                group_map=group_map,
                boot_number=preview_boot,
                alpha=args.alpha,
                n_jobs=1,
                random_seed=point_seed + 31,
                n_per_group=int(eval_n),
                permutations=preview_perms,
                omega2_floor=args.omega2_floor,
            )
            power = float(metrics["power"])
        return {"strength": float(strength), "omega2": float(omega2), "power": float(power)}

    def preview_many(strength_values: Iterable[float], offset: int = 0) -> List[Dict[str, float]]:
        indexed = [(int(offset + i), float(strength)) for i, strength in enumerate(strength_values)]
        if scenario_n_jobs == 1 or len(indexed) <= 1:
            out = [preview_one(i, strength) for i, strength in indexed]
        else:
            with ThreadPoolExecutor(max_workers=scenario_n_jobs) as ex:
                out = list(ex.map(lambda item: preview_one(item[0], item[1]), indexed))
        for row in out:
            print(
                f"[protein raw-pool preview] s={row['strength']:.3f} "
                f"omega2={row['omega2']:.4f} power={row['power']:.3f}",
                flush=True,
            )
        return out

    primary = np.linspace(0.0, 1.0, int(args.mdctf_strength_candidates))
    rows.extend(preview_many(primary))

    preview0 = pd.DataFrame(rows).sort_values("strength")
    omega0 = np.maximum.accumulate(preview0["omega2"].to_numpy(dtype=float))
    max_strength = max(1.0, float(getattr(args, "mdctf_strength_max", 1.0)))
    target_omega = max(float(args.target_omega2) * 1.25, float(args.target_omega2) + 0.01)
    needs_extension = (
        max_strength > 1.0
        and float(np.nanmax(omega0)) < target_omega
    )
    if needs_extension:
        extra_count = max(4, int(args.mdctf_strength_candidates))
        extra = np.linspace(1.0, max_strength, extra_count + 1)[1:]
        print(
            f"[protein raw-pool preview] extending strength scan to {max_strength:.3g} "
            f"because max preview omega2={float(np.nanmax(omega0)):.4f} < target window {target_omega:.4f}",
            flush=True,
        )
        offset = len(rows)
        rows.extend(preview_many(extra, offset=offset))

    preview = pd.DataFrame(rows).sort_values("strength")
    power = np.maximum.accumulate(preview["power"].to_numpy(dtype=float))
    omega = np.maximum.accumulate(preview["omega2"].to_numpy(dtype=float))
    strength = preview["strength"].to_numpy(dtype=float)
    if np.nanmax(omega) <= 1e-10:
        return [0.0, 1.0]

    plateau_idx = np.flatnonzero(power >= 1.0 - 1e-12)
    transition_hi = (
        float(omega[plateau_idx[0]])
        if len(plateau_idx) and plateau_idx[0] > 0
        else float(omega[-1])
    )
    omega_targets = np.linspace(0.0, transition_hi, int(args.mdctf_power_points))
    unique_omega, idx = np.unique(omega, return_index=True)
    unique_strength = strength[idx]
    selected = np.interp(
        np.clip(omega_targets, unique_omega[0], unique_omega[-1]),
        unique_omega,
        unique_strength,
    ).tolist()
    target_omega = float(getattr(args, "target_omega2", np.nan))
    if np.isfinite(target_omega):
        if unique_omega[0] <= target_omega <= unique_omega[-1]:
            selected.append(
                float(
                    np.interp(
                        target_omega,
                        unique_omega,
                        unique_strength,
                    )
                )
            )
        elif target_omega > unique_omega[-1]:
            selected.append(float(strength[-1]))
    selected.extend([0.0, 1.0])
    if not len(plateau_idx):
        selected.append(float(strength[-1]))
    if int(args.mdctf_plateau_points) > 0 and len(plateau_idx):
        plateau_stop = min(float(strength[-1]), max(1.0, float(strength[plateau_idx[0]])))
        selected.extend(
            np.linspace(float(strength[plateau_idx[0]]), plateau_stop, int(args.mdctf_plateau_points)).tolist()
        )
    out: List[float] = []
    for value in sorted(selected):
        clipped = float(np.clip(np.round(value, 4), 0.0, float(strength[-1])))
        if not out or abs(clipped - out[-1]) > 1e-4:
            out.append(clipped)
    return _limit_sorted_values(out, int(args.mdctf_power_points))


def _precompute_protein_scenarios(
    *,
    raw_dict: dict,
    args: SimpleNamespace,
    pilot_n: int,
    seed: int,
    pool_size_per_group: int,
    strength_eval_n: int,
) -> List[Dict[str, Any]]:
    pilot, source = protein_wf._protein_pilot_raw_view(raw_dict, pilot_n, seed, args)
    if str(args.mdctf_strengths).strip().lower() in {"auto", "power-uniform"}:
        strengths = _mdctf_mc_power_uniform_strengths(
            pilot, args, int(strength_eval_n), seed + 70000
        )
    else:
        strengths = [float(x) for x in str(args.mdctf_strengths).split(",") if x.strip()]
    print(
        f"[protein raw-pool] precomputing {len(strengths)} scenarios "
        f"(pilot_n={pilot_n}, pool_n={pool_size_per_group}, generator=MDC-TF-MC, "
        f"marginal_strength={args.marginal_strength}, eb_k={args.eb_k}, strengths={strengths})",
        flush=True,
    )

    def build_scenario(scenario_index: int, strength: float) -> Dict[str, Any]:
        point_seed = seed + 9000 + int(scenario_index) * 1291
        table, scenario_group_map = mdctf_mc_pool(
            pilot,
            int(pool_size_per_group),
            point_seed,
            float(strength),
            edge_fraction=args.edge_fraction,
            marginal_strength=args.marginal_strength,
            eb_k=args.eb_k,
        )
        dm = protein_wf.P.recompute_distance(pilot, table)
        omega2 = (
            0.0
            if np.isclose(strength, 0.0)
            else max(0.0, float(base_core.compute_omega2(dm, scenario_group_map)))
        )
        scenario = {
            "scenario_index": int(scenario_index),
            "pilot_n": int(pilot_n),
            "pilot_source": source,
            "strength": float(strength),
            "dm": dm,
            "group_map": scenario_group_map,
            "true_omega2": float(omega2),
            "mode": "exchangeable_null" if np.isclose(strength, 0.0) else "labeled_bootstrap",
            "pool_size_per_group": int(pool_size_per_group),
            "point_seed": int(point_seed),
        }
        print(
            f"[protein raw-pool] scenario s={strength:.4f} omega2={omega2:.4f}",
            flush=True,
        )
        return scenario

    def build_many(strength_values: Iterable[float], offset: int = 0) -> List[Dict[str, Any]]:
        indexed = [(int(offset + i), float(strength)) for i, strength in enumerate(strength_values)]
        scenario_n_jobs = max(1, int(getattr(args, "scenario_n_jobs", 1)))
        if scenario_n_jobs == 1 or len(indexed) <= 1:
            return [build_scenario(i, strength) for i, strength in indexed]
        with ThreadPoolExecutor(max_workers=scenario_n_jobs) as ex:
            return list(ex.map(lambda item: build_scenario(item[0], item[1]), indexed))

    scenarios: List[Dict[str, Any]] = build_many(strengths)

    refine_points = int(getattr(args, "mdctf_refine_target_points", 0) or 0)
    if refine_points > 0 and len(scenarios) >= 2:
        target = float(args.target_omega2)
        ordered = sorted(
            [s for s in scenarios if not np.isclose(float(s["strength"]), 0.0)],
            key=lambda s: float(s["true_omega2"]),
        )
        bracket: Optional[tuple[Dict[str, Any], Dict[str, Any]]] = None
        for left, right in zip(ordered[:-1], ordered[1:]):
            if float(left["true_omega2"]) <= target <= float(right["true_omega2"]):
                bracket = (left, right)
                break
        if bracket is None and len(ordered) >= 2:
            nearest_idx = int(
                np.argmin([abs(float(s["true_omega2"]) - target) for s in ordered])
            )
            if nearest_idx == 0:
                bracket = (ordered[0], ordered[1])
            elif nearest_idx == len(ordered) - 1:
                bracket = (ordered[-2], ordered[-1])
            else:
                lo = ordered[nearest_idx - 1]
                hi = ordered[nearest_idx + 1]
                bracket = (
                    lo
                    if abs(float(lo["true_omega2"]) - target)
                    < abs(float(hi["true_omega2"]) - target)
                    else hi,
                    ordered[nearest_idx],
                )
        if bracket is not None:
            s0 = float(bracket[0]["strength"])
            s1 = float(bracket[1]["strength"])
            low_s, high_s = sorted((s0, s1))
            existing = {round(float(s["strength"]), 4) for s in scenarios}
            refine_strengths = [
                float(np.round(x, 4))
                for x in np.linspace(low_s, high_s, refine_points + 2)[1:-1]
            ]
            refine_strengths = [x for x in refine_strengths if round(x, 4) not in existing]
            if refine_strengths:
                print(
                    f"[protein raw-pool] refining around target omega2={target:.4f} "
                    f"using strengths={refine_strengths}",
                    flush=True,
                )
                start_idx = len(scenarios)
                scenarios.extend(build_many(refine_strengths, offset=start_idx))

    scenarios.sort(key=lambda s: float(s["strength"]))
    return scenarios


def _evaluate_precomputed_protein_scenarios(
    *,
    scenarios: List[Dict[str, Any]],
    args: SimpleNamespace,
    n_per_group: int,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for scenario in scenarios:
        strength = float(scenario["strength"])
        if np.isclose(strength, 0.0):
            power = protein_wf._exchangeable_null_power(
                scenario["dm"],
                eval_n=int(n_per_group),
                boot=args.boot_number,
                perms=args.permutations,
                seed=int(scenario["point_seed"]) + int(n_per_group) * 101 + 41,
            )
            true_omega2 = float(scenario["true_omega2"])
            failed_bootstraps = 0
        else:
            metrics = protein_wf.summarize_distance_metrics_with_replacement(
                dm=scenario["dm"],
                group_map=scenario["group_map"],
                boot_number=args.boot_number,
                alpha=args.alpha,
                n_jobs=1,
                random_seed=int(seed) + int(scenario["scenario_index"]),
                n_per_group=int(n_per_group),
                permutations=args.permutations,
                omega2_floor=args.omega2_floor,
            )
            power = float(metrics["power"])
            true_omega2 = float(metrics["true_omega2"])
            failed_bootstraps = int(metrics.get("failed_bootstraps", 0))
        rows.append(
            {
                "pilot_n": int(scenario["pilot_n"]),
                "pilot_source": scenario["pilot_source"],
                "strength": strength,
                "eval_n": int(n_per_group),
                "true_omega2": true_omega2,
                "power": power,
                "mode": scenario["mode"],
                "scenario_index": int(scenario["scenario_index"]),
                "pool_size_per_group": int(scenario["pool_size_per_group"]),
                "failed_bootstraps": failed_bootstraps,
            }
        )
        print(
            f"[protein raw-pool] eval_n={n_per_group} s={strength:.3f} "
            f"omega2={true_omega2:.4f} power={power:.3f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def compute_protein_min_sample_size(
    *,
    table: Path | str = base_core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv",
    tree: Path | str = base_core.DATAPRO_DIR / "rooted-tree.nwk",
    group: Path | str = base_core.DATAPRO_DIR / "group.csv",
    target_power: float,
    target_omega2: Optional[float] = None,
    out: Path | str = "protein_min_sample_size_output",
    pilot_n: Optional[int] = None,
    min_n: int = 2,
    max_n: Optional[int] = None,
    alpha: float = 0.05,
    pool_size_per_group: int = 1000,
    boot_number: int = 200,
    permutations: int = 199,
    edge_fraction: float | str = "auto",
    marginal_strength: str = "auto",
    eb_k: str = "auto",
    mdctf_strengths: str = "auto",
    mdctf_strength_candidates: int = 15,
    mdctf_strength_max: float = 4.0,
    mdctf_power_points: int = DEFAULT_SCENARIO_POINTS,
    mdctf_plateau_points: int = 0,
    mdctf_preview_pool_size: int = 180,
    mdctf_refine_target_points: int = 0,
    power_preview_boot_number: int = 20,
    power_preview_permutations: int = 49,
    fit_bin_width: float = 0.0015,
    omega2_floor: float = 0.0,
    scenario_n_jobs: int = DEFAULT_SCENARIO_N_JOBS,
    coarse_step: int = DEFAULT_COARSE_STEP,
    stability_window: int = DEFAULT_STABILITY_WINDOW,
    random_seed: int = 20260614,
    tree_jitter_sigma: float = 0.0,
    tree_nni_prob: float = 0.0,
    tree_support_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    base_core.load_core_runtime()
    out_path = Path(out)
    args = _protein_args(
        table=Path(table),
        tree=Path(tree),
        group=Path(group),
        out=out_path,
        target_power=target_power,
        target_omega2=target_omega2,
        alpha=alpha,
        pool_size_per_group=pool_size_per_group,
        boot_number=boot_number,
        permutations=permutations,
        edge_fraction=edge_fraction,
        marginal_strength=marginal_strength,
        eb_k=eb_k,
        mdctf_strengths=mdctf_strengths,
        mdctf_strength_candidates=mdctf_strength_candidates,
        mdctf_strength_max=mdctf_strength_max,
        mdctf_power_points=mdctf_power_points,
        mdctf_plateau_points=mdctf_plateau_points,
        mdctf_preview_pool_size=mdctf_preview_pool_size,
        mdctf_refine_target_points=mdctf_refine_target_points,
        power_preview_boot_number=power_preview_boot_number,
        power_preview_permutations=power_preview_permutations,
        fit_bin_width=fit_bin_width,
        omega2_floor=omega2_floor,
        scenario_n_jobs=scenario_n_jobs,
        random_seed=random_seed,
        tree_jitter_sigma=tree_jitter_sigma,
        tree_nni_prob=tree_nni_prob,
        tree_support_threshold=tree_support_threshold,
    )
    args.tree = _effective_tree_path_for_raw_workflow(
        tree=args.tree,
        out=out_path,
        random_seed=random_seed,
        sigma=tree_jitter_sigma,
        nni_prob=tree_nni_prob,
        support_threshold=tree_support_threshold,
        stem="protein_tree_perturbed",
    )
    group_map = _read_group_map(args.group)
    long_df, aligned_group_map = _read_protein_long_table(args.table, group_map)
    observed_n = int(aligned_group_map.value_counts().min())
    resolved_pilot_n = int(pilot_n if pilot_n is not None else observed_n)
    resolved_min_n = max(2, int(min_n))
    resolved_max_n = _default_max_n(observed_n, max_n)
    if resolved_max_n < resolved_min_n:
        raise ValueError(f"max_n ({resolved_max_n}) must be >= min_n ({resolved_min_n}).")
    raw_dict = protein_wf._build_protein_raw_dict(long_df, aligned_group_map, args)
    pilot_for_target, pilot_source_for_target = protein_wf._protein_pilot_raw_view(
        raw_dict,
        resolved_pilot_n,
        int(random_seed + resolved_pilot_n * 1009),
        args,
    )
    observed_pilot_omega2 = _compute_observed_pilot_omega2(pilot_for_target)
    resolved_target_omega2, target_omega2_source = _resolve_target_omega2(
        target_omega2,
        observed_pilot_omega2,
    )
    args.target_omega2 = resolved_target_omega2
    reusable_pool_size = _effective_reusable_pool_size(
        requested_pool_size=pool_size_per_group,
        max_n=resolved_max_n,
    )
    scenario_seed = int(random_seed + resolved_pilot_n * 1009)
    scenarios = _precompute_protein_scenarios(
        raw_dict=raw_dict,
        args=args,
        pilot_n=resolved_pilot_n,
        seed=scenario_seed,
        pool_size_per_group=reusable_pool_size,
        strength_eval_n=resolved_max_n,
    )

    def evaluate_curve(n_per_group: int, stage: str) -> pd.DataFrame:
        args.eval_n = int(n_per_group)
        seed = int(random_seed + resolved_pilot_n * 1009 + n_per_group * 37)
        print(
            f"[protein raw-pool] {stage}: reusing {len(scenarios)} scenarios, "
            f"pilot_n={resolved_pilot_n}, eval_n={n_per_group}",
            flush=True,
        )
        return _evaluate_precomputed_protein_scenarios(
            scenarios=scenarios,
            args=args,
            n_per_group=n_per_group,
            seed=seed,
        )

    minimum_n, power_df, metrics_df = _search_minimum_n(
        min_n=resolved_min_n,
        max_n=resolved_max_n,
        target_power=target_power,
        target_omega2=resolved_target_omega2,
        alpha=alpha,
        fit_bin_width=fit_bin_width,
        coarse_step=coarse_step,
        stability_window=stability_window,
        evaluate_curve_fn=evaluate_curve,
    )
    summary = {
        "workflow": "protein",
        "engine": "raw_pool",
        "generator": "MDC-TF-MC",
        "distance": "PhyloFunc",
        "minimum_n_per_group": minimum_n,
        "target_power": float(target_power),
        "target_omega2": float(resolved_target_omega2),
        "target_omega2_source": target_omega2_source,
        **_curve_support_flags(power_df),
        "observed_pilot_omega2": float(observed_pilot_omega2),
        "observed_pilot_source": pilot_source_for_target,
        "alpha": float(alpha),
        "observed_n_per_group": observed_n,
        "pilot_n_per_group": resolved_pilot_n,
        "sweep_min_n": resolved_min_n,
        "sweep_max_n": resolved_max_n,
        "coarse_step": int(coarse_step),
        "stability_window": int(stability_window),
        "pool_size_per_group": int(pool_size_per_group),
        "effective_pool_size_per_group": int(reusable_pool_size),
        "scenario_reuse": True,
        "boot_number": int(boot_number),
        "permutations": int(permutations),
        "fit_bin_width": float(fit_bin_width),
        "scenario_n_jobs": int(scenario_n_jobs),
        "edge_fraction": float(args.edge_fraction),
        "marginal_strength": str(marginal_strength),
        "eb_k": str(eb_k),
        "mdctf_strengths": str(mdctf_strengths),
        "mdctf_strength_max": float(mdctf_strength_max),
        "mdctf_refine_target_points": int(mdctf_refine_target_points),
        "tree_jitter_sigma": float(tree_jitter_sigma),
        "tree_nni_prob": float(tree_nni_prob),
    }
    metrics_df = _with_power_uncertainty(metrics_df, int(boot_number))
    _write_outputs(out=out_path, summary=summary, power_df=power_df, metrics_df=metrics_df)
    return {"summary": summary, "power_by_sample_size": power_df, "scenario_metrics_by_sample_size": metrics_df}


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-power", type=float, required=True)
    parser.add_argument(
        "--target-omega2",
        type=float,
        default=None,
        help="Target realized omega^2. If omitted, the observed omega^2 of the resolved pilot is used.",
    )
    parser.add_argument("--pilot-n", type=int, default=None)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--pool-size-per-group", type=int, default=1000)
    parser.add_argument("--boot-number", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=199)
    parser.add_argument("--coarse-step", type=int, default=DEFAULT_COARSE_STEP)
    parser.add_argument("--stability-window", type=int, default=DEFAULT_STABILITY_WINDOW)
    parser.add_argument("--omega2-floor", type=float, default=0.0)
    parser.add_argument(
        "--scenario-n-jobs",
        type=int,
        default=DEFAULT_SCENARIO_N_JOBS,
        help="Parallel workers for raw-pool scenario precomputation; use 1 for serial execution.",
    )
    parser.add_argument("--random-seed", type=int, default=20260614)
    parser.add_argument("--tree-jitter-sigma", type=float, default=0.0)
    parser.add_argument("--tree-nni-prob", type=float, default=0.0)
    parser.add_argument("--tree-support-threshold", type=float, default=None)
    parser.add_argument("--out", type=Path, required=True)


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Minimum sample-size estimator using raw-pool workflows."
    )
    sub = parser.add_subparsers(dest="workflow", required=True)

    gene = sub.add_parser("gene", help="PCAM + Gemelli workflow")
    gene.add_argument("--table", type=Path, default=base_core.DATAGENE_DIR / "table.csv")
    gene.add_argument("--tree", type=Path, default=base_core.DATAGENE_DIR / "rooted-tree.nwk")
    gene.add_argument("--taxonomy", type=Path, default=base_core.DATAGENE_DIR / "taxonomy.csv")
    gene.add_argument("--group", type=Path, default=base_core.DATAGENE_DIR / "group.csv")
    gene.add_argument("--qiime-env", type=str, default="qiime2-metagenome-2024.10")
    gene.add_argument(
        "--pcam-gene-blocks",
        type=str,
        default="auto",
        help="PCAM phylogenetic block count; 'auto' uses sqrt(n_features) clipped to 6..24.",
    )
    gene.add_argument("--pcam-ndon", type=int, default=1)
    gene.add_argument(
        "--pcam-grid",
        type=str,
        default="auto",
        choices=["auto", "omega-uniform", "fixed"],
        help="PCAM formal effect grid. auto/omega-uniform previews candidate effects and selects by realized omega2.",
    )
    gene.add_argument("--pcam-power-points", type=int, default=DEFAULT_SCENARIO_POINTS)
    gene.add_argument("--pcam-near-zero-points", type=int, default=4)
    gene.add_argument("--pcam-near-zero-omega-max", type=str, default="auto")
    gene.add_argument("--pcam-pi-candidates", type=int, default=17)
    gene.add_argument("--pcam-scale-candidates", type=int, default=6)
    gene.add_argument("--pcam-scale-max", type=float, default=1.7)
    gene.add_argument("--pcam-scale-extend-max", type=float, default=3.0)
    gene.add_argument("--fit-bin-width", type=float, default=0.003)
    gene.add_argument("--use-phylogeny", dest="use_phylogeny", action="store_true", default=True)
    gene.add_argument("--no-use-phylogeny", dest="use_phylogeny", action="store_false")
    _add_common_args(gene)
    gene.set_defaults(func=lambda a: compute_gene_min_sample_size(**_namespace_to_kwargs(a)))

    protein = sub.add_parser("protein", help="MDC-TF + PhyloFunc workflow")
    protein.add_argument(
        "--table",
        type=Path,
        default=base_core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv",
    )
    protein.add_argument("--tree", type=Path, default=base_core.DATAPRO_DIR / "rooted-tree.nwk")
    protein.add_argument("--group", type=Path, default=base_core.DATAPRO_DIR / "group.csv")
    protein.add_argument(
        "--edge-fraction",
        type=str,
        default="auto",
        help="MDC-TF-MC edge-effect fraction in [0,1], or 'auto' (=1.0) for broad omega coverage.",
    )
    protein.add_argument(
        "--marginal-strength",
        type=str,
        default="auto",
        help="Positive-abundance marginal calibration strength for MDC-TF-MC; use 0 to disable.",
    )
    protein.add_argument(
        "--eb-k",
        type=str,
        default="auto",
        help="Empirical-Bayes pseudo-count for MDC-TF-MC positive marginal calibration.",
    )
    protein.add_argument("--mdctf-strengths", type=str, default="auto")
    protein.add_argument("--mdctf-strength-candidates", type=int, default=15)
    protein.add_argument(
        "--mdctf-strength-max",
        type=float,
        default=4.0,
        help="Maximum MDC-TF-MC strength for auto preview when strength<=1 does not bracket target omega2.",
    )
    protein.add_argument("--mdctf-power-points", type=int, default=DEFAULT_SCENARIO_POINTS)
    protein.add_argument("--mdctf-plateau-points", type=int, default=0)
    protein.add_argument("--mdctf-preview-pool-size", type=int, default=180)
    protein.add_argument(
        "--mdctf-refine-target-points",
        type=int,
        default=0,
        help="Add this many formal MDC-TF-MC scenarios between the formal points bracketing target omega2.",
    )
    protein.add_argument("--power-preview-boot-number", type=int, default=20)
    protein.add_argument("--power-preview-permutations", type=int, default=49)
    protein.add_argument("--fit-bin-width", type=float, default=0.0015)
    _add_common_args(protein)
    protein.set_defaults(func=lambda a: compute_protein_min_sample_size(**_namespace_to_kwargs(a)))
    return parser


def _namespace_to_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    data = vars(args).copy()
    data.pop("func", None)
    data.pop("workflow", None)
    return data


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = create_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    result = args.func(args)
    summary = result["summary"]
    minimum_n = summary.get("minimum_n_per_group")
    if minimum_n is None:
        print(
            f"[{summary['workflow']}] No minimum n found in "
            f"[{summary['sweep_min_n']}, {summary['sweep_max_n']}]."
        )
        if not summary.get("target_omega2_bracketed", True):
            print(
                f"[{summary['workflow']}] Target ω² lies outside the simulated "
                f"range; the extrapolated power at the target is not adopted. "
                f"Widen the effect grid (see the warning above) and rerun."
            )
    else:
        print(f"[{summary['workflow']}] Minimum sample size per group: {minimum_n}")
    print(f"[{summary['workflow']}] Outputs written to {Path(args.out)}")


if __name__ == "__main__":
    main()
