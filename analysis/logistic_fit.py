"""Free (unanchored) logistic power-curve fit.

Drop-in replacement for core.fit_sigmoid_curve / core.anchored_sigmoid_curve, but WITHOUT anchoring
the curve to pass through (0, alpha): no synthetic (0, alpha) point is added and the lower asymptote
is left free, determined by the data. `alpha` is accepted only for call-signature compatibility and
is unused. core is left untouched; the workflows and figures import these instead.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


def logistic_curve(x, k, x0, alpha=None, floor=0.0):
    """Logistic with lower asymptote `floor` (= type-I rate alpha for a power curve):
    power(x) = floor + (1-floor)/(1+exp(-k*(x-x0))).  floor=0 -> the old plain logistic.
    (`alpha` is kept only for positional back-compat with old callers and is ignored.)"""
    x = np.asarray(x, dtype=float)
    return np.clip(floor + (1.0 - floor) / (1.0 + np.exp(-k * (x - x0))), 0.0, 1.0)


def fit_logistic(scenario_metrics_df: pd.DataFrame, alpha=None, floor=0.0) -> dict:
    """Fit a logistic with lower asymptote pinned at `floor` (set floor=alpha so the curve starts at
    the type-I rate, e.g. 0.05; floor=0 reproduces the old free fit)."""
    valid = (scenario_metrics_df.dropna(subset=["true_omega2", "power"])
             .sort_values("true_omega2").reset_index(drop=True))
    if valid.empty:
        return {"status": "no_valid_points", "x": np.array([]), "y": np.array([]), "params": None}
    x = np.clip(valid["true_omega2"].to_numpy(dtype=float), 0.0, None)
    y = np.clip(valid["power"].to_numpy(dtype=float), 0.0, 1.0)
    if len(np.unique(x)) < 2:
        return {"status": "insufficient_unique_x", "x": x, "y": y, "params": None}
    x_scale = max(float(np.nanmax(x)), 1.0)
    pos_x = x[x > 0]
    p0 = [max(0.5, 5.0 / x_scale), float(np.median(pos_x)) if len(pos_x) else 0.5 * x_scale]
    try:
        params, _ = curve_fit(lambda xv, k, x0: logistic_curve(xv, k, x0, floor=floor), x, y, p0=p0,
                              bounds=([1e-6, 0.0], [1e3, 5.0 * x_scale]), maxfev=20000)
        status = "ok"
    except Exception:
        params = p0; status = "fallback_initial_guess"
    return {"status": status, "x": x, "y": y, "params": {"k": float(params[0]), "x0": float(params[1]), "floor": float(floor)}}
