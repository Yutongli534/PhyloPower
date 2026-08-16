"""Reproducibility regression tests for seeded PERMANOVA and tree perturbation."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ``phylopower`` must be imported before the top-level workflow modules so the
# embedded-module finder is installed first.
import phylopower  # noqa: F401
from phylopower import core


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _toy_distance(seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    from scipy.spatial.distance import pdist, squareform

    rng = np.random.default_rng(seed)
    pts = np.vstack([rng.normal(0.0, 1.0, (6, 3)), rng.normal(0.8, 1.0, (6, 3))])
    ids = [f"s{i}" for i in range(12)]
    dm = pd.DataFrame(squareform(pdist(pts)), index=ids, columns=ids)
    gm = pd.Series(["A"] * 6 + ["B"] * 6, index=ids, name="group")
    return dm, gm


def test_permanova_p_value_is_seed_reproducible() -> None:
    dm, gm = _toy_distance()
    p1, failed1 = core.compute_permanova_p_value_with_status(
        dm, group_map=gm, permutations=99, seed=123
    )
    p2, failed2 = core.compute_permanova_p_value_with_status(
        dm, group_map=gm, permutations=99, seed=123
    )
    assert not failed1 and not failed2
    assert p1 == p2


def test_bootstrap_power_is_bit_identical_across_runs() -> None:
    from semisynthetic_power import summarize_distance_metrics_with_replacement

    dm, gm = _toy_distance()
    kwargs = dict(
        dm=dm,
        group_map=gm,
        boot_number=20,
        alpha=0.05,
        n_jobs=1,
        random_seed=20260614,
        n_per_group=6,
        permutations=49,
    )
    first = summarize_distance_metrics_with_replacement(**kwargs)
    second = summarize_distance_metrics_with_replacement(**kwargs)
    assert first == second
    assert np.isfinite(first["power"])


def test_tree_rng_does_not_depend_on_python_hash_seed() -> None:
    snippet = (
        "from phylopower import core; "
        "print(core.make_tree_rng(20260614, 'gene_tree_perturbed').random())"
    )
    draws = []
    for hash_seed in ("0", "42"):
        result = subprocess.run(
            [sys.executable, "-c", snippet],
            check=True,
            capture_output=True,
            text=True,
            cwd=PROJECT_ROOT,
            env={**os.environ, "PYTHONHASHSEED": hash_seed},
        )
        draws.append(result.stdout.strip())
    assert draws[0] == draws[1]


def test_exchangeable_null_power_is_seed_reproducible() -> None:
    from _protein_mdctf_optimized_curve import _exchangeable_null_power

    dm, _ = _toy_distance()
    kwargs = dict(eval_n=3, boot=5, perms=49, seed=777)
    first = _exchangeable_null_power(dm, **kwargs)
    second = _exchangeable_null_power(dm, **kwargs)
    assert first == second


def _fake_curve_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_omega2": [0.0, 0.02, 0.04, 0.05],
            "power": [0.05, 0.9, 0.95, 1.0],
        }
    )


def _search(target_omega2: float):
    from phylopower import paper_core

    return paper_core._search_minimum_n(
        min_n=2,
        max_n=6,
        target_power=0.8,
        target_omega2=target_omega2,
        alpha=0.05,
        fit_bin_width=0.003,
        coarse_step=2,
        stability_window=1,
        evaluate_curve_fn=lambda n, stage: _fake_curve_df(),
    )


def test_unbracketed_target_never_qualifies(capsys) -> None:
    from phylopower import paper_core

    minimum_n, power_df, _ = _search(target_omega2=0.5)
    assert minimum_n is None
    assert not power_df["target_omega2_bracketed"].any()
    assert not power_df["qualifies"].any()
    assert "falls outside the simulated range" in capsys.readouterr().out
    flags = paper_core._curve_support_flags(power_df)
    assert flags == {"target_omega2_bracketed": False, "low_omega_support_warning": True}


def test_bracketed_target_still_qualifies(capsys) -> None:
    minimum_n, power_df, _ = _search(target_omega2=0.03)
    assert minimum_n == 2
    assert power_df["target_omega2_bracketed"].all()
    assert "falls outside the simulated range" not in capsys.readouterr().out
