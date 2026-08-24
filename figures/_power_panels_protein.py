#!/usr/bin/env python3
"""Metaproteomic power-curve panel helpers shared by the Figure 5/Figure 8 scripts.

Reads the archived MDC-TF-MC power-curve grid; no simulations are run here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from _fig4_curve_plotting import binned_monotone, fit_binned_null_hill  # noqa: E402

DATA = ROOT / "data" / "archived_runs" / "fig4_new" / "fig4_mdctf_mc_power_curves.csv"

PILOT_KEYS = [7, 10, 17, 30, 50, 80]
PILOT_COLORS = {
    7: "#4b006e",
    10: "#35679a",
    17: "#ffcf24",
    30: "#0f8b7c",
    50: "#b45309",
    80: "#7c3aed",
}


def apply_local_style() -> None:
    for font_path in (
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial.ttf",
    ):
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 13,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 15,
            "axes.linewidth": 0.95,
            "axes.edgecolor": "#444444",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "legend.frameon": False,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.62,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def fit_curve(sub: pd.DataFrame, x: np.ndarray, bin_width: float) -> np.ndarray:
    binned = binned_monotone(sub, bin_width)
    y, _params = fit_binned_null_hill(binned, x)
    if y is not None:
        return y

    ordered = sub.sort_values("true_omega2")
    sx = ordered["true_omega2"].to_numpy(float)
    sy = np.maximum.accumulate(ordered["power"].clip(0, 1).to_numpy(float))
    uniq_x, first_idx = np.unique(sx, return_index=True)
    uniq_y = sy[first_idx]
    return np.interp(x, uniq_x, uniq_y, left=uniq_y[0], right=uniq_y[-1])


def bootstrap_curve_band(
    sub: pd.DataFrame,
    x: np.ndarray,
    *,
    bin_width: float,
    seed: int,
    n_boot: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = fit_curve(sub, x, bin_width)
    rng = np.random.default_rng(seed)
    curves: list[np.ndarray] = []
    rows = sub[["true_omega2", "power"]].reset_index(drop=True)
    for _ in range(n_boot):
        sampled = rows.iloc[rng.integers(0, len(rows), len(rows))].copy()
        try:
            curve = fit_curve(sampled, x, bin_width)
        except Exception:
            continue
        if np.all(np.isfinite(curve)):
            curves.append(curve)

    if len(curves) >= 40:
        stack = np.vstack(curves)
        lo = np.nanpercentile(stack, 2.5, axis=0)
        hi = np.nanpercentile(stack, 97.5, axis=0)
    else:
        # Fallback to a conservative binomial Monte-Carlo interval around the
        # fitted curve. Existing power estimates were generated with boot=100.
        se = np.sqrt(np.clip(center * (1.0 - center), 0.0, None) / 100.0)
        lo = center - 1.96 * se
        hi = center + 1.96 * se
    return center, np.clip(lo, 0, 1), np.clip(hi, 0, 1)


def finish_axis(ax, xmax: float, show_ylabel: bool) -> None:
    ax.axhline(0.8, color="#7a8798", ls=":", lw=1.25, zorder=0)
    ax.axhline(0.05, color="#404040", ls=":", lw=1.0, zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("ω²")
    ax.set_ylabel("Power" if show_ylabel else "")
    ax.grid(True)
    ax.tick_params(axis="both", length=4, width=0.9, color="#444444")


def draw_points_panel(ax, suball: pd.DataFrame, *, by: str, keys: list[int], title: str, bin_width: float, legend_loc: str = "lower right", legend_ncol: int = 1) -> None:
    xmax = max(0.08, float(suball["true_omega2"].max()) * 1.06)
    x = np.linspace(0, xmax, 600)
    for key in keys:
        sub = suball[suball[by] == key].sort_values("true_omega2")
        label = f"n={key}" if by == "eval_n" else f"pilot n={key}"
        color = PILOT_COLORS[key]
        ax.scatter(
            sub["true_omega2"],
            sub["power"],
            s=24,
            color=color,
            alpha=0.68,
            edgecolor="white",
            linewidth=0.45,
            zorder=3,
        )
        y = fit_curve(sub[["true_omega2", "power"]], x, bin_width)
        ax.plot(x, y, color=color, lw=2.35, label=label, zorder=4)
    ax.set_title(title)
    finish_axis(ax, xmax, show_ylabel=True)
    leg = ax.legend(loc=legend_loc, ncol=legend_ncol, handlelength=2.2, borderaxespad=0.35)
    if legend_loc != "lower right":
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_edgecolor("#dddddd")


def draw_band_panel(ax, suball: pd.DataFrame, *, by: str, keys: list[int], title: str, bin_width: float, seed_base: int) -> None:
    xmax = max(0.08, float(suball["true_omega2"].max()) * 1.06)
    x = np.linspace(0, xmax, 600)
    for i, key in enumerate(keys):
        sub = suball[suball[by] == key].sort_values("true_omega2")
        label = f"n={key}" if by == "eval_n" else f"pilot n={key}"
        color = PILOT_COLORS[key]
        center, lo, hi = bootstrap_curve_band(
            sub[["true_omega2", "power"]],
            x,
            bin_width=bin_width,
            seed=seed_base + key * 101,
        )
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0, zorder=2)
        ax.plot(x, center, color=color, lw=2.45, label=label, zorder=4)
    ax.set_title(title)
    finish_axis(ax, xmax, show_ylabel=True)
    ax.legend(loc="lower right", ncol=1, handlelength=2.6, borderaxespad=0.35)
