"""Shared Fig4 power-curve plotting helpers.

The fitted curve starts from each curve's empirical omega=0 power. It is not
anchored to the nominal alpha, and it does not freely invent a lower asymptote
when an observed null point exists.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def null_started_hill_sigmoid(x: np.ndarray, h: float, x0: float, floor: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x0 = max(float(x0), 1e-9)
    ratio = np.power(np.clip(x, 0.0, None) / x0, h)
    floor = float(np.clip(floor, 0.0, 0.5))
    y = floor + (1.0 - floor) * ratio / (1.0 + ratio)
    return np.clip(y, 0.0, 1.0)


def pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    values: list[float] = []
    weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for i, (yi, wi) in enumerate(zip(y, w)):
        values.append(float(yi))
        weights.append(float(wi))
        starts.append(i)
        ends.append(i + 1)
        while len(values) >= 2 and values[-2] > values[-1]:
            total_w = weights[-2] + weights[-1]
            merged = (values[-2] * weights[-2] + values[-1] * weights[-1]) / total_w
            values[-2] = merged
            weights[-2] = total_w
            ends[-2] = ends[-1]
            values.pop()
            weights.pop()
            starts.pop()
            ends.pop()
    out = np.empty_like(y, dtype=float)
    for v, st, en in zip(values, starts, ends):
        out[st:en] = v
    return np.clip(out, 0.0, 1.0)


def binned_monotone(sub: pd.DataFrame, width: float) -> pd.DataFrame:
    dat = sub[["true_omega2", "power"]].dropna().copy()
    dat["true_omega2"] = dat["true_omega2"].clip(lower=0.0)
    dat["power"] = dat["power"].clip(0.0, 1.0)
    dat["bin"] = np.floor(dat["true_omega2"] / width).astype(int)
    dat.loc[dat["true_omega2"].eq(0.0), "bin"] = -1
    b = (
        dat.groupby("bin", as_index=False)
        .agg(true_omega2=("true_omega2", "mean"), power=("power", "mean"), weight=("power", "size"))
        .sort_values("true_omega2")
        .reset_index(drop=True)
    )
    if len(b) >= 2:
        b["power_mono"] = pava(b["power"].to_numpy(float), b["weight"].to_numpy(float))
    else:
        b["power_mono"] = b["power"]
    return b


def fit_binned_null_hill(b: pd.DataFrame, x: np.ndarray) -> tuple[np.ndarray | None, dict | None]:
    if len(b) < 4 or b["true_omega2"].nunique() < 3:
        return None, None
    bx = b["true_omega2"].to_numpy(float)
    by = b["power_mono"].to_numpy(float)
    bw = np.maximum(1.0, b["weight"].to_numpy(float))
    x_scale = max(float(np.nanmax(bx)), 1e-3)
    pos_x = bx[bx > 0]
    null_rows = b[b["true_omega2"].le(1e-12)]
    floor = float(null_rows["power_mono"].mean()) if len(null_rows) else float(np.clip(by[0], 0.0, 0.2))
    floor = float(np.clip(floor, 0.0, 0.5))
    p0 = [2.0, float(np.median(pos_x)) if len(pos_x) else x_scale / 2]
    try:
        params, _ = curve_fit(
            lambda xv, h, x0: null_started_hill_sigmoid(xv, h, x0, floor),
            bx,
            by,
            p0=p0,
            sigma=1.0 / np.sqrt(bw),
            absolute_sigma=False,
            bounds=([0.2, 1e-8], [16.0, max(0.5, 5.0 * x_scale)]),
            maxfev=30000,
        )
    except Exception:
        return None, None
    y = null_started_hill_sigmoid(x, float(params[0]), float(params[1]), floor)
    return y, {"h": float(params[0]), "x0": float(params[1]), "floor": floor}


def draw_binned_null_hill_group(
    ax,
    sub: pd.DataFrame,
    *,
    color: str,
    label: str,
    x: np.ndarray,
    bin_width: float,
    raw_alpha: float = 0.22,
    raw_size: float = 18,
    point_size: float = 34,
) -> dict | None:
    sub = sub.sort_values("true_omega2")
    ax.scatter(sub["true_omega2"], sub["power"], s=raw_size, color=color, alpha=raw_alpha, linewidths=0)
    b = binned_monotone(sub, bin_width)
    ax.scatter(
        b["true_omega2"],
        b["power_mono"],
        s=point_size,
        color=color,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.6,
        label=label,
        zorder=3,
    )
    y, params = fit_binned_null_hill(b, x)
    if y is not None:
        ax.plot(x, y, color=color, lw=2.1)
    return params
