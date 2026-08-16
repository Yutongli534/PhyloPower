"""MDC-TF with empirical-Bayes marginal calibration.

The calibration is deliberately post-hoc on positive abundances only:
the taxon-function detection mask is unchanged, so edge count, taxon degree,
function degree, and connectance remain inherited from MDC-TF templates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from _protein_mdctf_curve import mdctf_pool


def _auto_eb_k(d: dict) -> float:
    all_idx = np.concatenate([d["gs"][g] for g in d["groups"]])
    counts = (np.asarray(d["abund"][:, all_idx], dtype=float) > 0).sum(axis=1)
    positive = counts[counts > 0]
    return float(np.median(positive)) if len(positive) else 1.0


def calibrate_positive_marginals(
    d: dict,
    tab: pd.DataFrame,
    *,
    strength: float | str = "auto",
    eb_k: float | str = "auto",
) -> pd.DataFrame:
    """Match pilot positive log-abundance moments without changing support.

    For each taxon-function feature, positive synthetic log-abundances are
    partially mapped to the pilot positive mean and variance. The empirical
    Bayes weight is small for rare features, which avoids fitting noise from
    very few pilot positives.
    """
    # copy=True is required: under pandas 3 Copy-on-Write, to_numpy() returns a
    # read-only view and the in-place updates below would raise ValueError.
    syn = tab.to_numpy(dtype=float, copy=True)
    before_totals = np.maximum(syn.sum(axis=0), 1e-12)

    all_idx = np.concatenate([d["gs"][g] for g in d["groups"]])
    real = np.asarray(d["abund"][:, all_idx], dtype=float)
    floor = np.zeros(real.shape[0], dtype=float)
    for i in range(real.shape[0]):
        pos = real[i, real[i] > 0]
        floor[i] = float(pos.min() / 100.0) if len(pos) else 1e-12

    gamma = 1.0 if strength == "auto" else float(np.clip(float(strength), 0.0, 1.0))
    k_value = _auto_eb_k(d) if eb_k == "auto" else float(eb_k)
    for i in range(real.shape[0]):
        rpos = real[i, real[i] > 0]
        smask = syn[i] > 0
        if len(rpos) < 3 or int(smask.sum()) < 3:
            continue
        rz = np.log1p(rpos)
        sz = np.log1p(syn[i, smask])
        r_sd = float(np.std(rz, ddof=1))
        s_sd = float(np.std(sz, ddof=1))
        if not np.isfinite(r_sd) or not np.isfinite(s_sd) or s_sd <= 1e-9:
            continue
        mu_r = float(np.mean(rz))
        mu_s = float(np.mean(sz))
        corrected = mu_r + (r_sd / s_sd) * (sz - mu_s)
        w = gamma * (len(rpos) / (len(rpos) + k_value))
        new_z = (1.0 - w) * sz + w * corrected
        syn[i, smask] = np.maximum(np.expm1(new_z), floor[i])

    after_totals = np.maximum(syn.sum(axis=0), 1e-12)
    syn *= before_totals / after_totals
    return pd.DataFrame(syn, index=tab.index, columns=tab.columns)


def mdctf_mc_pool(
    d: dict,
    M: int,
    seed: int,
    effect_strength: float,
    *,
    edge_fraction: float | str = "auto",
    marginal_strength: float | str = "auto",
    eb_k: float | str = "auto",
    balanced_templates: bool = True,
    residual_mode: str = "random",
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a structure-preserving MDC-TF pool with marginal calibration."""
    tab, sgm = mdctf_pool(
        d,
        M,
        seed,
        effect_strength,
        edge_fraction=edge_fraction,
        balanced_templates=balanced_templates,
        residual_mode=residual_mode,
    )
    if marginal_strength != "auto" and float(marginal_strength) <= 0:
        return tab, sgm
    return calibrate_positive_marginals(d, tab, strength=marginal_strength, eb_k=eb_k), sgm
