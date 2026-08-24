#!/usr/bin/env python3
"""Metagenomic power-curve panel helpers shared by the Figure 5/Figure 8 scripts.

Reads the archived PCAM power-curve tables; no simulations are run here.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

import sys  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from _fig4_curve_plotting import binned_monotone, fit_binned_null_hill  # noqa: E402

DATA = ROOT / "data" / "figdata" / "fig4_power_curves.csv"
OUTDIR = ROOT / "data" / "archived_runs" / "fig4_new"
PANEL_A_DENSE = OUTDIR / "fig4_metagenomics_panel_a_dense.csv"
PANEL_A_4710_REFINED = OUTDIR / "fig4_metagenomics_panel_a_4710_refined.csv"
PANEL_C_EXTRA = OUTDIR / "fig4_metagenomics_panel_c_extra_pilots.csv"

PALETTE = ["#4b006e", "#35679a", "#ffcf24", "#0f8b7c", "#b45309", "#7c3aed"]
PILOTS = [4, 7, 10]
PILOTS_EXTENDED = [4, 7, 10, 30, 50, 80]
STUDY_SIZES = [4, 7, 10, 30, 50, 80]
PILOT_COLORS = {k: PALETTE[i] for i, k in enumerate(PILOTS)}
PILOT_EXTENDED_COLORS = {k: PALETTE[i] for i, k in enumerate(PILOTS_EXTENDED)}
SIZE_COLORS = {k: PALETTE[i] for i, k in enumerate(STUDY_SIZES)}


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
    sx = ordered["true_omega2"].clip(lower=0).to_numpy(float)
    sy = np.maximum.accumulate(ordered["power"].clip(0, 1).to_numpy(float))
    uniq_x, first_idx = np.unique(sx, return_index=True)
    uniq_y = sy[first_idx]
    return np.interp(x, uniq_x, uniq_y, left=uniq_y[0], right=uniq_y[-1])


def ceiling_hill_curve(sub: pd.DataFrame, x: np.ndarray, bin_width: float) -> np.ndarray:
    binned = binned_monotone(sub, bin_width)
    if len(binned) < 4 or binned["true_omega2"].nunique() < 3:
        return fit_curve(sub, x, bin_width)
    bx = binned["true_omega2"].to_numpy(float)
    by = binned["power_mono"].to_numpy(float)
    bw = np.maximum(1.0, binned["weight"].to_numpy(float))
    x_scale = max(float(np.nanmax(bx)), 1e-3)
    pos_x = bx[bx > 0]
    floor0 = float(np.clip(by[0], 0.0, 0.18))
    ceil0 = float(np.clip(max(by[-1], np.nanmax(by), floor0 + 0.05), floor0 + 0.05, 1.0))

    def model(xv, h, x0, floor, ceiling):
        xv = np.asarray(xv, dtype=float)
        ratio = np.power(np.clip(xv, 0.0, None) / max(float(x0), 1e-9), h)
        return np.clip(floor + (ceiling - floor) * ratio / (1.0 + ratio), 0.0, 1.0)

    try:
        params, _ = curve_fit(
            model,
            bx,
            by,
            p0=[2.0, float(np.median(pos_x)) if len(pos_x) else x_scale / 2.0, floor0, ceil0],
            sigma=1.0 / np.sqrt(bw),
            absolute_sigma=False,
            bounds=(
                [0.2, 1e-8, 0.0, max(0.1, float(np.nanmax(by)))],
                [16.0, max(0.8, 5.0 * x_scale), 0.25, 1.0],
            ),
            maxfev=30000,
        )
        return model(x, *params)
    except Exception:
        return fit_curve(sub, x, bin_width)


def curve_band(
    sub: pd.DataFrame,
    x: np.ndarray,
    *,
    bin_width: float,
    mc_boot: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = fit_curve(sub, x, bin_width)
    se = np.sqrt(np.clip(center * (1.0 - center), 0.0, None) / float(mc_boot))
    lo = np.clip(center - 1.96 * se, 0.0, 1.0)
    hi = np.clip(center + 1.96 * se, 0.0, 1.0)
    return center, lo, hi


def finish_axis(ax, xmax: float, show_ylabel: bool) -> None:
    ax.axhline(0.8, color="#7a8798", ls=":", lw=1.25, zorder=0)
    ax.axhline(0.05, color="#404040", ls=":", lw=1.0, zorder=0)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("ω²")
    ax.set_ylabel("Power" if show_ylabel else "")
    ax.grid(True)
    ax.tick_params(axis="both", length=4, width=0.9, color="#444444")


def draw_points_panel(
    ax,
    suball: pd.DataFrame,
    *,
    by: str,
    keys: list[int],
    colors: dict[int, str],
    title: str,
    bin_width: float,
    legend_loc: str = "lower right",
    legend_ncol: int = 1,
) -> None:
    xmax = max(0.08, float(suball["true_omega2"].max()) * 1.06)
    x = np.linspace(0, xmax, 600)
    for key in keys:
        sub = suball[suball[by] == key].sort_values("true_omega2")
        if sub.empty:
            continue
        label = f"n={key}" if by == "eval_n" else f"pilot n={key}"
        color = colors[key]
        ax.scatter(
            sub["true_omega2"],
            sub["power"],
            s=30,
            color=color,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        y = ceiling_hill_curve(sub[["true_omega2", "power"]], x, bin_width)
        ax.plot(x, y, color=color, lw=2.35, label=label, zorder=4)
    ax.set_title(title)
    finish_axis(ax, xmax, show_ylabel=True)
    leg = ax.legend(loc=legend_loc, ncol=legend_ncol, handlelength=2.2, borderaxespad=0.35)
    if legend_loc != "lower right":
        leg.set_frame_on(True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_edgecolor("#dddddd")


def draw_band_panel(
    ax,
    suball: pd.DataFrame,
    *,
    by: str,
    keys: list[int],
    colors: dict[int, str],
    title: str,
    bin_width: float,
) -> None:
    xmax = max(0.08, float(suball["true_omega2"].max()) * 1.06)
    x = np.linspace(0, xmax, 600)
    for key in keys:
        sub = suball[suball[by] == key].sort_values("true_omega2")
        if sub.empty:
            continue
        label = f"n={key}" if by == "eval_n" else f"pilot n={key}"
        color = colors[key]
        center, lo, hi = curve_band(sub[["true_omega2", "power"]], x, bin_width=bin_width)
        ax.fill_between(x, lo, hi, color=color, alpha=0.13, lw=0, zorder=2)
        ax.plot(x, center, color=color, lw=2.45, label=label, zorder=4)
    ax.set_title(title)
    finish_axis(ax, xmax, show_ylabel=True)
    ax.legend(loc="lower right", ncol=1, handlelength=2.6, borderaxespad=0.35)
