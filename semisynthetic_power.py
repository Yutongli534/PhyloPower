#!/usr/bin/env python3
"""Experimental semi-synthetic power analysis entrypoint for Phylopower.

This script intentionally lives outside ``phylopower.core``.  It builds a
larger synthetic sample pool from pilot data, then evaluates sample sizes by
sampling that pool with replacement, matching the Kelly/micropower bootstrap
estimand while keeping duplicate draws negligible by using a large pool.  The
downstream distance, PERMANOVA, omega-squared, RWCT, and power-curve machinery
is reused from ``phylopower``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial.distance import pdist, squareform

from phylopower import core


ID_COLS = ["Taxon", "Function"]
REEXEC_ENV_VAR = "PHYLOPOWER_SEMISYNTHETIC_REEXECED"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-value))


def _logit(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _safe_label(value: object) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return label.strip("_") or "group"


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _maybe_reexec_taxon_in_qiime_env(args: argparse.Namespace) -> None:
    """Run taxon mode inside the requested QIIME conda env when needed."""

    if getattr(args, "mode", None) not in {"taxon", "compare-taxon", "sensitivity-taxon"}:
        return
    if _module_available("qiime2"):
        return
    if os.environ.get(REEXEC_ENV_VAR) == "1":
        return
    qiime_env = getattr(args, "qiime_env", None)
    if not qiime_env:
        return
    if os.environ.get("CONDA_DEFAULT_ENV") == qiime_env:
        return
    conda = shutil.which("conda")
    if conda is None:
        return

    env = os.environ.copy()
    env[REEXEC_ENV_VAR] = "1"
    command = [conda, "run", "-n", str(qiime_env), "python", str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execvpe(conda, command, env)


def _read_group_map(path: Path) -> pd.Series:
    df = pd.read_csv(path, encoding="utf-8-sig")
    required = {"sample_id", "group_name"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} must contain columns: sample_id, group_name")
    return pd.Series(df["group_name"].astype(str).values, index=df["sample_id"].astype(str), name="group")


def _parse_int_list(value: str) -> List[int]:
    out = [int(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _parse_float_list(value: str) -> List[float]:
    out = [float(part.strip()) for part in str(value).split(",") if part.strip()]
    if not out:
        raise ValueError("Expected at least one float.")
    return out


def _subsample_group_map(group_map: pd.Series, n_per_group: int, seed: int) -> pd.Series:
    rng = np.random.default_rng(int(seed))
    selected: List[str] = []
    for group_name in sorted(group_map.dropna().unique()):
        members = group_map[group_map == group_name].index.to_numpy()
        if len(members) < n_per_group:
            raise ValueError(
                f"Group {group_name!r} has only {len(members)} samples; "
                f"cannot create pilot_n={n_per_group}."
            )
        selected.extend(rng.choice(members, size=n_per_group, replace=False).tolist())
    return group_map.loc[selected].copy()


def _resolve_max_n(observed_n: int, max_n: Optional[int]) -> int:
    return int(max_n) if max_n is not None else max(observed_n * 3, observed_n + 20)


def _resolve_pool_size(observed_n: int, max_n: int, pool_size_per_group: Optional[int]) -> int:
    resolved = int(pool_size_per_group) if pool_size_per_group is not None else max(5 * max_n, 10 * observed_n, 50)
    if resolved < max_n:
        raise ValueError(
            f"--pool-size-per-group ({resolved}) must be >= max_n ({max_n}) "
            "so the semi-synthetic pool can approximate a prospective sampling population."
        )
    return resolved


def _expected_duplicate_stats(pool_size: int, draw_size: int) -> Dict[str, float]:
    if pool_size <= 0 or draw_size <= 0:
        return {
            "pool_size_per_group": float(pool_size),
            "draw_n_per_group": float(draw_size),
            "expected_unique_per_group": np.nan,
            "expected_duplicate_slots_per_group": np.nan,
            "expected_duplicate_fraction": np.nan,
        }
    expected_unique = float(pool_size) * (1.0 - (1.0 - 1.0 / float(pool_size)) ** int(draw_size))
    expected_duplicates = float(draw_size) - expected_unique
    return {
        "pool_size_per_group": float(pool_size),
        "draw_n_per_group": float(draw_size),
        "expected_unique_per_group": expected_unique,
        "expected_duplicate_slots_per_group": expected_duplicates,
        "expected_duplicate_fraction": expected_duplicates / float(draw_size),
    }


def _sample_groups_without_replacement(
    dm_df: pd.DataFrame,
    group_map: pd.Series,
    n_per_group: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(int(seed))
    aligned_group_map = group_map.loc[dm_df.index]
    selected: List[str] = []
    for group_name in sorted(aligned_group_map.dropna().unique()):
        members = aligned_group_map[aligned_group_map == group_name].index.to_numpy()
        if len(members) < n_per_group:
            raise ValueError(
                f"Group {group_name!r} has only {len(members)} synthetic samples; "
                f"cannot sample n_per_group={n_per_group} without replacement. "
                "Increase --pool-size-per-group."
            )
        selected.extend(rng.choice(members, size=n_per_group, replace=False).tolist())
    sampled_dm = dm_df.loc[selected, selected].copy()
    sampled_group_map = aligned_group_map.loc[selected].copy()
    return sampled_dm, sampled_group_map


def _as_distance_frame(dm: Any) -> pd.DataFrame:
    if hasattr(dm, "to_data_frame"):
        return dm.to_data_frame().copy()
    return pd.DataFrame(dm).copy()


def summarize_distance_metrics_without_replacement(
    *,
    dm: Any,
    group_map: pd.Series,
    boot_number: int,
    alpha: float,
    n_jobs: int,
    random_seed: int,
    n_per_group: int,
    permutations: int,
    omega2_floor: Optional[float] = None,
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Estimate power from a synthetic pool using no-replacement sampling."""

    dm_df = _as_distance_frame(dm)
    common = dm_df.index.intersection(group_map.index)
    dm_df = dm_df.loc[common, common].apply(pd.to_numeric, errors="coerce")
    aligned_group_map = group_map.loc[common]
    if dm_df.empty or aligned_group_map.empty:
        return {"power": np.nan, "mean_boot_omega2": np.nan, "true_omega2": np.nan, "failed_bootstraps": 0}

    values = dm_df.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)
    dm_df.iloc[:, :] = values
    true_omega2 = core.compute_omega2(dm_df, aligned_group_map)
    if omega2_floor is not None and np.isfinite(true_omega2):
        true_omega2 = max(float(true_omega2), float(omega2_floor))

    seeds = core.make_bootstrap_seeds(random_seed, boot_number, n_per_group, len(dm_df))

    def _run_one(seed: int) -> Tuple[float, float, int]:
        context = {**(failure_context or {}), "seed": int(seed)}
        try:
            boot_df, boot_group_map = _sample_groups_without_replacement(
                dm_df,
                aligned_group_map,
                n_per_group=n_per_group,
                seed=int(seed),
            )
            p_value, failed = core.compute_permanova_p_value_with_status(
                boot_df,
                group_map=boot_group_map,
                permutations=permutations,
                failure_log_path=failure_log_path,
                failure_context=context,
                seed=core.make_permanova_seed(seed),
            )
            omega2 = core.compute_omega2(boot_df, boot_group_map)
            if omega2_floor is not None and np.isfinite(omega2):
                omega2 = max(float(omega2), float(omega2_floor))
            return float(p_value), float(omega2), int(failed)
        except Exception as exc:
            core.log_permanova_failure(failure_log_path, context, exc)
            return np.nan, np.nan, 1

    results = Parallel(n_jobs=n_jobs)(delayed(_run_one)(int(seed)) for seed in seeds[:boot_number])
    p_values, omega_values, failed_flags = zip(*results) if results else ([], [], [])
    p_values = np.asarray(p_values, dtype=float)
    omega_values = np.asarray(omega_values, dtype=float)
    valid_p = p_values[np.isfinite(p_values)]
    return {
        "power": float((valid_p <= alpha).mean()) if len(valid_p) else np.nan,
        "mean_boot_omega2": max(float(np.nanmean(omega_values)), 0.0) if len(omega_values) else np.nan,
        "true_omega2": float(true_omega2) if np.isfinite(true_omega2) else np.nan,
        "failed_bootstraps": int(np.sum(np.asarray(failed_flags, dtype=int))) if len(failed_flags) else 0,
    }


def summarize_distance_metrics_with_replacement(
    *,
    dm: Any,
    group_map: pd.Series,
    boot_number: int,
    alpha: float,
    n_jobs: int,
    random_seed: int,
    n_per_group: int,
    permutations: int,
    omega2_floor: Optional[float] = None,
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Estimate power from a synthetic pool using replacement sampling."""

    dm_df = _as_distance_frame(dm)
    common = dm_df.index.intersection(group_map.index)
    dm_df = dm_df.loc[common, common].apply(pd.to_numeric, errors="coerce")
    aligned_group_map = group_map.loc[common]
    if dm_df.empty or aligned_group_map.empty:
        return {"power": np.nan, "mean_boot_omega2": np.nan, "true_omega2": np.nan, "failed_bootstraps": 0}

    values = dm_df.to_numpy(copy=True)
    np.fill_diagonal(values, 0.0)
    dm_df.iloc[:, :] = values
    true_omega2 = core.compute_omega2(dm_df, aligned_group_map)
    if omega2_floor is not None and np.isfinite(true_omega2):
        true_omega2 = max(float(true_omega2), float(omega2_floor))

    group_count = len(sorted(aligned_group_map.dropna().unique()))
    group_pool_sizes = aligned_group_map.value_counts()
    min_pool_size = int(group_pool_sizes.min()) if len(group_pool_sizes) else 0
    plan = core.prepare_bootstrap_sampling_plan(
        dm_df,
        [n_per_group] * group_count,
        aligned_group_map,
    )
    seeds = core.make_bootstrap_seeds(random_seed, boot_number, n_per_group, len(dm_df), 101)

    def _run_one(seed: int) -> Tuple[float, float, int]:
        context = {**(failure_context or {}), "seed": int(seed)}
        try:
            boot_df, boot_group_map = core.bootstrap_distance_matrix_from_plan(plan, seed=int(seed))
            p_value, failed = core.compute_permanova_p_value_with_status(
                boot_df,
                group_map=boot_group_map,
                permutations=permutations,
                failure_log_path=failure_log_path,
                failure_context=context,
                seed=core.make_permanova_seed(seed),
            )
            omega2 = core.compute_omega2(boot_df, boot_group_map)
            if omega2_floor is not None and np.isfinite(omega2):
                omega2 = max(float(omega2), float(omega2_floor))
            return float(p_value), float(omega2), int(failed)
        except Exception as exc:
            core.log_permanova_failure(failure_log_path, context, exc)
            return np.nan, np.nan, 1

    results = Parallel(n_jobs=n_jobs)(delayed(_run_one)(int(seed)) for seed in seeds[:boot_number])
    p_values, omega_values, failed_flags = zip(*results) if results else ([], [], [])
    p_values = np.asarray(p_values, dtype=float)
    omega_values = np.asarray(omega_values, dtype=float)
    valid_p = p_values[np.isfinite(p_values)]
    return {
        "power": float((valid_p <= alpha).mean()) if len(valid_p) else np.nan,
        "mean_boot_omega2": max(float(np.nanmean(omega_values)), 0.0) if len(omega_values) else np.nan,
        "true_omega2": float(true_omega2) if np.isfinite(true_omega2) else np.nan,
        "failed_bootstraps": int(np.sum(np.asarray(failed_flags, dtype=int))) if len(failed_flags) else 0,
        **_expected_duplicate_stats(min_pool_size, int(n_per_group)),
    }


def summarize_distance_metrics_by_resampling(
    *,
    resampling: str,
    dm: Any,
    group_map: pd.Series,
    boot_number: int,
    alpha: float,
    n_jobs: int,
    random_seed: int,
    n_per_group: int,
    permutations: int,
    omega2_floor: Optional[float] = None,
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    kwargs = {
        "dm": dm,
        "group_map": group_map,
        "boot_number": boot_number,
        "alpha": alpha,
        "n_jobs": n_jobs,
        "random_seed": random_seed,
        "n_per_group": n_per_group,
        "permutations": permutations,
        "omega2_floor": omega2_floor,
        "failure_log_path": failure_log_path,
        "failure_context": failure_context,
    }
    if resampling == "with_replacement":
        return summarize_distance_metrics_with_replacement(**kwargs)
    if resampling == "without_replacement":
        return summarize_distance_metrics_without_replacement(**kwargs)
    raise ValueError(f"Unsupported resampling mode: {resampling}")


def _allocate_counts(composition: np.ndarray, library_size: int, rng: np.random.Generator) -> np.ndarray:
    if library_size <= 0 or composition.sum() <= 0:
        return np.zeros_like(composition, dtype=int)
    probs = composition / composition.sum()
    return rng.multinomial(int(library_size), probs).astype(int)


def _clr_from_counts(sample_by_feature: pd.DataFrame, pseudocount: float = 0.5) -> pd.DataFrame:
    x = sample_by_feature.astype(float).to_numpy(copy=True) + pseudocount
    row_sums = x.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0] = 1.0
    logp = np.log(x / row_sums)
    clr = logp - logp.mean(axis=1, keepdims=True)
    return pd.DataFrame(clr, index=sample_by_feature.index, columns=sample_by_feature.columns)


def _softmax_from_clr(vector: np.ndarray) -> np.ndarray:
    z = vector - np.max(vector)
    exp_z = np.exp(z)
    denom = exp_z.sum()
    if not np.isfinite(denom) or denom <= 0:
        return np.full_like(vector, 1.0 / len(vector), dtype=float)
    return exp_z / denom


def _read_taxon_feature_table(path: Path, group_map: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    if path.suffix.lower() == ".qza":
        import biom
        from qiime2 import Artifact

        table = Artifact.load(str(path)).view(biom.Table).to_dataframe()
    else:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        feature_col = "Taxon" if "Taxon" in raw.columns else raw.columns[0]
        table = raw.rename(columns={feature_col: "Taxon"}).drop_duplicates("Taxon").set_index("Taxon")
    table.columns = table.columns.astype(str)
    sample_cols = [sample_id for sample_id in group_map.index if sample_id in table.columns]
    if not sample_cols:
        raise ValueError("No group_map sample IDs matched the taxon table columns.")
    aligned_group_map = group_map.loc[sample_cols]
    table = table[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    table[table < 0] = 0.0
    return table, aligned_group_map


def _compute_gemelli_loading_coordinates(
    table: pd.DataFrame,
    tree_path: Path,
    *,
    n_components: int = 3,
    max_iterations: int = 5,
) -> pd.DataFrame:
    """Return Gemelli/phylo-RPCA sample coordinates that generate its distance matrix.

    Gemelli's phylogenetic-RPCA distance is computed from MatrixCompletion.U,
    not from the final biplot coordinates after the extra centering/refactor
    step in optspace_helper. This helper mirrors the package internals up to
    MatrixCompletion so the simulator is anchored in Gemelli's own metric space.
    """

    import biom
    from gemelli.matrix_completion import MatrixCompletion
    from gemelli.preprocessing import phylogenetic_rclr_transformation
    from gemelli.rpca import rpca_table_processing

    clean = table.apply(pd.to_numeric, errors="coerce").fillna(0.0).copy()
    clean[clean < 0] = 0.0
    biom_table = biom.Table(
        clean.to_numpy(dtype=float),
        observation_ids=clean.index.astype(str).tolist(),
        sample_ids=clean.columns.astype(str).tolist(),
    )
    biom_table = rpca_table_processing(
        biom_table,
        min_sample_count=0,
        min_feature_count=0,
        min_feature_frequency=0,
    )
    _counts_by_node, rclr_table, _phylogeny = phylogenetic_rclr_transformation(
        biom_table,
        str(tree_path),
        min_depth=0,
    )
    rclr = rclr_table.matrix_data.toarray().T
    n_comp = min(int(n_components), max(2, min(rclr.shape) - 1))
    opt = MatrixCompletion(n_components=n_comp, max_iterations=int(max_iterations)).fit(rclr)
    cols = [f"GemelliU{i + 1}" for i in range(opt.sample_weights.shape[1])]
    return pd.DataFrame(opt.sample_weights, index=list(rclr_table.ids("sample")), columns=cols)


def generate_taxon_pool(
    table: pd.DataFrame,
    group_map: pd.Series,
    pool_size_per_group: int,
    random_seed: int,
    between_scale: float = 1.0,
    residual_scale: float = 1.0,
    noise_multiplier: float = 0.10,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    sample_by_feature = table.transpose().astype(float)
    clr = _clr_from_counts(sample_by_feature)
    global_centroid = clr.mean(axis=0).to_numpy(dtype=float)
    synthetic_rows: List[np.ndarray] = []
    synthetic_ids: List[str] = []
    synthetic_groups: Dict[str, str] = {}
    diagnostics: List[Dict[str, Any]] = []

    for group_name in sorted(group_map.unique()):
        members = group_map[group_map == group_name].index.tolist()
        block_counts = sample_by_feature.loc[members]
        block_clr = clr.loc[members]
        group_centroid = block_clr.mean(axis=0).to_numpy(dtype=float)
        centroid = global_centroid + float(between_scale) * (group_centroid - global_centroid)
        residuals = block_clr.sub(block_clr.mean(axis=0), axis=1).to_numpy(dtype=float)
        residual_sd = np.nanstd(residuals, axis=0, ddof=1) if len(members) > 1 else np.zeros(table.shape[0])
        prevalence = ((block_counts > 0).sum(axis=0).to_numpy(dtype=float) + 0.5) / (len(members) + 1.0)
        lib_sizes = block_counts.sum(axis=1).to_numpy(dtype=float)
        log_lib = np.log(np.maximum(lib_sizes, 1.0))
        log_lib_sd = float(np.std(log_lib, ddof=1)) if len(log_lib) > 1 else 0.0

        for i in range(pool_size_per_group):
            template_idx = int(rng.integers(0, max(len(residuals), 1)))
            residual = residuals[template_idx] * float(residual_scale) if len(residuals) else np.zeros_like(centroid)
            noise = rng.normal(0.0, np.maximum(residual_sd, 1e-8) * float(noise_multiplier))
            composition = _softmax_from_clr(centroid + residual + noise)
            presence = rng.binomial(1, prevalence).astype(bool)
            if not presence.any():
                presence[int(np.argmax(prevalence))] = True
            composition = np.where(presence, composition, 0.0)
            if composition.sum() <= 0:
                composition[int(np.argmax(prevalence))] = 1.0
            lib_size = int(round(float(np.exp(rng.normal(float(log_lib.mean()), log_lib_sd)))))
            lib_size = max(lib_size, 1)
            synthetic_rows.append(_allocate_counts(composition, lib_size, rng))
            sample_id = f"syn_{_safe_label(group_name)}_{i + 1:04d}"
            synthetic_ids.append(sample_id)
            synthetic_groups[sample_id] = str(group_name)

        observed_zero_rate = float((block_counts.to_numpy() <= 0).mean())
        synthetic_block = np.asarray(synthetic_rows[-pool_size_per_group:], dtype=float)
        diagnostics.append(
            {
                "group_name": group_name,
                "observed_n": len(members),
                "synthetic_pool_size": pool_size_per_group,
                "between_scale": float(between_scale),
                "residual_scale": float(residual_scale),
                "noise_multiplier": float(noise_multiplier),
                "observed_zero_rate": observed_zero_rate,
                "synthetic_zero_rate": float((synthetic_block <= 0).mean()),
                "zero_rate_drift": float((synthetic_block <= 0).mean() - observed_zero_rate),
                "observed_library_median": float(np.median(lib_sizes)),
                "synthetic_library_median": float(np.median(synthetic_block.sum(axis=1))),
                "library_median_ratio": float(np.median(synthetic_block.sum(axis=1)) / max(float(np.median(lib_sizes)), 1.0)),
            }
        )

    synthetic_sample_by_feature = pd.DataFrame(synthetic_rows, index=synthetic_ids, columns=table.index)
    synthetic_table = synthetic_sample_by_feature.transpose()
    synthetic_group_map = pd.Series(synthetic_groups, name="group")
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df = _append_feature_distance_diagnostics(
        diagnostics_df,
        observed_features=clr,
        observed_groups=group_map,
        synthetic_features=_clr_from_counts(synthetic_sample_by_feature),
        synthetic_groups=synthetic_group_map,
    )
    return synthetic_table, synthetic_group_map, diagnostics_df


def _read_protein_long_table(path: Path, group_map: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, encoding="utf-8-sig")
    if not set(ID_COLS).issubset(df.columns):
        raise ValueError(f"{path} must contain Taxon and Function columns.")
    df = df.drop_duplicates(ID_COLS, keep="first").copy()
    df[ID_COLS] = df[ID_COLS].astype(str)
    sample_cols = [sample_id for sample_id in group_map.index if sample_id in df.columns]
    if not sample_cols:
        raise ValueError("No group_map sample IDs matched the taxon-function table columns.")
    aligned_group_map = group_map.loc[sample_cols]
    df[sample_cols] = df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df[sample_cols] = df[sample_cols].clip(lower=0.0)
    return df[ID_COLS + sample_cols].copy(), aligned_group_map


def _log_positive_matrix(feature_by_sample: pd.DataFrame) -> pd.DataFrame:
    x = feature_by_sample.astype(float).to_numpy(copy=True)
    row_floor = np.zeros(x.shape[0], dtype=float)
    for i in range(x.shape[0]):
        positives = x[i][x[i] > 0]
        row_floor[i] = positives.min() / 100.0 if len(positives) else 1e-9
    log_x = np.log(np.where(x > 0, x, row_floor[:, None]))
    return pd.DataFrame(log_x, index=feature_by_sample.index, columns=feature_by_sample.columns)


def _empirical_ks_distance(left: Sequence[float], right: Sequence[float]) -> float:
    left_arr = np.sort(np.asarray(left, dtype=float))
    right_arr = np.sort(np.asarray(right, dtype=float))
    left_arr = left_arr[np.isfinite(left_arr)]
    right_arr = right_arr[np.isfinite(right_arr)]
    if len(left_arr) == 0 or len(right_arr) == 0:
        return np.nan
    values = np.sort(np.unique(np.concatenate([left_arr, right_arr])))
    left_cdf = np.searchsorted(left_arr, values, side="right") / len(left_arr)
    right_cdf = np.searchsorted(right_arr, values, side="right") / len(right_arr)
    return float(np.max(np.abs(left_cdf - right_cdf)))


def _protein_sample_structure_metrics(
    sample_by_feature: pd.DataFrame,
    taxon_values: Sequence[str],
    function_values: Sequence[str],
) -> pd.DataFrame:
    taxon_arr = np.asarray(taxon_values, dtype=object)
    function_arr = np.asarray(function_values, dtype=object)
    values = sample_by_feature.to_numpy(dtype=float, copy=True)
    rows: List[Dict[str, Any]] = []
    for sample_id, row in zip(sample_by_feature.index.astype(str), values):
        present = row > 0
        rows.append(
            {
                "sample_id": sample_id,
                "edge_count": int(np.sum(present)),
                "taxon_degree": int(len(np.unique(taxon_arr[present]))) if np.any(present) else 0,
                "function_degree": int(len(np.unique(function_arr[present]))) if np.any(present) else 0,
            }
        )
    return pd.DataFrame(rows).set_index("sample_id")


def _add_distribution_diagnostics(
    row: Dict[str, Any],
    observed: pd.Series,
    synthetic: pd.Series,
    prefix: str,
) -> None:
    observed_arr = observed.to_numpy(dtype=float)
    synthetic_arr = synthetic.to_numpy(dtype=float)
    row[f"observed_{prefix}_mean"] = float(np.nanmean(observed_arr)) if len(observed_arr) else np.nan
    row[f"synthetic_{prefix}_mean"] = float(np.nanmean(synthetic_arr)) if len(synthetic_arr) else np.nan
    row[f"{prefix}_drift"] = row[f"synthetic_{prefix}_mean"] - row[f"observed_{prefix}_mean"]
    row[f"{prefix}_ks"] = _empirical_ks_distance(observed_arr, synthetic_arr)


def generate_taxon_function_pool(
    long_df: pd.DataFrame,
    group_map: pd.Series,
    pool_size_per_group: int,
    random_seed: int,
    between_scale: float = 1.0,
    residual_scale: float = 1.0,
    noise_multiplier: float = 0.10,
    detection_slope: float = 1.0,
    protein_generator: str = "template-mask",
    presence_retention: float = 1.0,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    if protein_generator not in {"template-mask", "bernoulli"}:
        raise ValueError("--protein-generator must be either 'template-mask' or 'bernoulli'.")
    rng = np.random.default_rng(random_seed)
    sample_cols = [c for c in long_df.columns if c not in ID_COLS]
    taxon_values = long_df["Taxon"].astype(str).to_numpy()
    function_values = long_df["Function"].astype(str).to_numpy()
    feature_ids = pd.Index([f"{t}||{f}" for t, f in zip(long_df["Taxon"], long_df["Function"])])
    feature_by_sample = long_df[sample_cols].astype(float).copy()
    feature_by_sample.index = feature_ids
    log_values = _log_positive_matrix(feature_by_sample)
    sample_by_feature_log = log_values.transpose()
    sample_by_feature_raw = feature_by_sample.transpose()
    global_centroid = sample_by_feature_log.mean(axis=0).to_numpy(dtype=float)
    raw_values = sample_by_feature_raw.to_numpy(dtype=float)
    row_index = list(sample_by_feature_raw.index)
    # presence_retention r in [0,1] controls how much of the between-group PRESENCE separation is
    # kept: with template-mask, a synthetic sample borrows a cross-group sample's whole presence
    # pattern with probability (1-r)/2, which scales the between-group prevalence difference by r.
    # r=1 reproduces the pilot's presence separation; r<1 dilutes it. Used (with between_scale) to
    # calibrate the pool's effect size down to the pilot's bias-corrected omega^2.
    presence_retention = float(np.clip(presence_retention, 0.0, 1.0))
    cross_group_prob = (1.0 - presence_retention) / 2.0

    synthetic_cols: Dict[str, np.ndarray] = {}
    synthetic_groups: Dict[str, str] = {}
    diagnostics: List[Dict[str, Any]] = []

    for group_name in sorted(group_map.unique()):
        members = group_map[group_map == group_name].index.tolist()
        other_positions = np.asarray(
            [pos for pos, sid in enumerate(row_index) if group_map.get(sid) != group_name], dtype=int
        )
        block_raw = sample_by_feature_raw.loc[members]
        block_log = sample_by_feature_log.loc[members]
        group_centroid = block_log.mean(axis=0).to_numpy(dtype=float)
        centroid = global_centroid + float(between_scale) * (group_centroid - global_centroid)
        residuals = block_log.sub(block_log.mean(axis=0), axis=1).to_numpy(dtype=float)
        residual_sd = np.nanstd(residuals, axis=0, ddof=1) if len(members) > 1 else np.zeros(len(feature_ids))
        prevalence = ((block_raw > 0).sum(axis=0).to_numpy(dtype=float) + 0.5) / (len(members) + 1.0)
        totals = block_raw.sum(axis=1).to_numpy(dtype=float)
        log_total = np.log(np.maximum(totals, 1.0))
        log_total_sd = float(np.std(log_total, ddof=1)) if len(log_total) > 1 else 0.0
        observed_structure = _protein_sample_structure_metrics(block_raw, taxon_values, function_values)
        if len(members):
            repeats = int(np.ceil(pool_size_per_group / len(members)))
            template_order = np.concatenate([rng.permutation(len(members)) for _ in range(repeats)])[:pool_size_per_group]
        else:
            template_order = np.zeros(pool_size_per_group, dtype=int)

        group_values: List[np.ndarray] = []
        template_edge_diffs: List[int] = []
        template_taxon_degree_diffs: List[int] = []
        template_function_degree_diffs: List[int] = []
        for i in range(pool_size_per_group):
            template_idx = int(template_order[i])
            residual = residuals[template_idx] * float(residual_scale) if len(residuals) else np.zeros_like(centroid)
            noise = rng.normal(0.0, np.maximum(residual_sd, 1e-8) * float(noise_multiplier))
            latent_log = centroid + residual + noise
            template_values = block_raw.iloc[template_idx].to_numpy(dtype=float) if len(block_raw) else np.zeros_like(centroid)
            if protein_generator == "template-mask":
                # Borrow a cross-group sample's presence pattern w.p. cross_group_prob to dilute
                # between-group presence separation (abundance stays own-group via centroid/residual).
                if cross_group_prob > 0.0 and other_positions.size and rng.random() < cross_group_prob:
                    present = raw_values[int(rng.choice(other_positions))] > 0
                else:
                    present = template_values > 0
            elif float(detection_slope) > 0:
                intensity_scale = np.maximum(residual_sd, 0.25)
                intensity_z = np.clip((latent_log - group_centroid) / intensity_scale, -6.0, 6.0)
                detect_prob = _sigmoid(_logit(prevalence) + float(detection_slope) * intensity_z)
                present = rng.binomial(1, detect_prob).astype(bool)
            else:
                present = rng.binomial(1, prevalence).astype(bool)
            if not present.any() and protein_generator != "template-mask":
                present[int(np.argmax(prevalence))] = True
            values = np.where(present, np.exp(latent_log), 0.0)
            if protein_generator == "template-mask":
                target_total = float(np.sum(template_values))
            else:
                target_total = float(np.exp(rng.normal(float(log_total.mean()), log_total_sd)))
            if values.sum() > 0 and target_total > 0:
                values = values * (target_total / values.sum())
            sample_id = f"syn_{_safe_label(group_name)}_{i + 1:04d}"
            synthetic_cols[sample_id] = values
            synthetic_groups[sample_id] = str(group_name)
            group_values.append(values)
            if protein_generator == "template-mask":
                synthetic_present = values > 0
                template_present = template_values > 0
                synthetic_taxon_degree = len(np.unique(taxon_values[synthetic_present])) if np.any(synthetic_present) else 0
                template_taxon_degree = len(np.unique(taxon_values[template_present])) if np.any(template_present) else 0
                synthetic_function_degree = (
                    len(np.unique(function_values[synthetic_present])) if np.any(synthetic_present) else 0
                )
                template_function_degree = (
                    len(np.unique(function_values[template_present])) if np.any(template_present) else 0
                )
                template_edge_diffs.append(int(abs(np.sum(synthetic_present) - np.sum(template_present))))
                template_taxon_degree_diffs.append(int(abs(synthetic_taxon_degree - template_taxon_degree)))
                template_function_degree_diffs.append(int(abs(synthetic_function_degree - template_function_degree)))

        synthetic_block = np.asarray(group_values, dtype=float)
        synthetic_block_df = pd.DataFrame(
            synthetic_block,
            index=[f"syn_{_safe_label(group_name)}_{i + 1:04d}" for i in range(pool_size_per_group)],
            columns=feature_ids,
        )
        synthetic_structure = _protein_sample_structure_metrics(synthetic_block_df, taxon_values, function_values)
        observed_zero_rate = float((block_raw.to_numpy() <= 0).mean())
        diagnostic_row = {
            "group_name": group_name,
            "observed_n": len(members),
            "synthetic_pool_size": pool_size_per_group,
            "protein_generator": protein_generator,
            "between_scale": float(between_scale),
            "residual_scale": float(residual_scale),
            "noise_multiplier": float(noise_multiplier),
            "detection_slope": float(detection_slope),
            "observed_zero_rate": observed_zero_rate,
            "synthetic_zero_rate": float((synthetic_block <= 0).mean()),
            "zero_rate_drift": float((synthetic_block <= 0).mean() - observed_zero_rate),
            "observed_library_median": float(np.median(totals)),
            "synthetic_library_median": float(np.median(synthetic_block.sum(axis=1))),
            "library_median_ratio": float(np.median(synthetic_block.sum(axis=1)) / max(float(np.median(totals)), 1.0)),
            "max_template_edge_count_diff": int(max(template_edge_diffs, default=0)),
            "max_template_taxon_degree_diff": int(max(template_taxon_degree_diffs, default=0)),
            "max_template_function_degree_diff": int(max(template_function_degree_diffs, default=0)),
        }
        _add_distribution_diagnostics(diagnostic_row, observed_structure["edge_count"], synthetic_structure["edge_count"], "edge_count")
        _add_distribution_diagnostics(diagnostic_row, observed_structure["taxon_degree"], synthetic_structure["taxon_degree"], "taxon_degree")
        _add_distribution_diagnostics(
            diagnostic_row,
            observed_structure["function_degree"],
            synthetic_structure["function_degree"],
            "function_degree",
        )
        diagnostics.append(diagnostic_row)

    synthetic_values_df = pd.DataFrame(synthetic_cols)
    synthetic_df = pd.concat(
        [long_df[ID_COLS].reset_index(drop=True), synthetic_values_df.reset_index(drop=True)],
        axis=1,
    )
    synthetic_group_map = pd.Series(synthetic_groups, name="group")
    synthetic_feature_by_sample = pd.DataFrame(synthetic_cols, index=feature_ids)
    synthetic_sample_by_feature_log = _log_positive_matrix(synthetic_feature_by_sample).transpose()
    diagnostics_df = pd.DataFrame(diagnostics)
    diagnostics_df = _append_feature_distance_diagnostics(
        diagnostics_df,
        observed_features=sample_by_feature_log,
        observed_groups=group_map,
        synthetic_features=synthetic_sample_by_feature_log,
        synthetic_groups=synthetic_group_map,
    )
    return synthetic_df, synthetic_group_map, diagnostics_df


def _within_group_feature_distance_mean(features: pd.DataFrame, group_map: pd.Series, group_name: str) -> float:
    members = [sample_id for sample_id in group_map[group_map == group_name].index if sample_id in features.index]
    if len(members) < 2:
        return np.nan
    values = features.loc[members].to_numpy(dtype=float)
    return float(np.mean(pdist(values, metric="euclidean")))


def _append_feature_distance_diagnostics(
    diagnostics_df: pd.DataFrame,
    observed_features: pd.DataFrame,
    observed_groups: pd.Series,
    synthetic_features: pd.DataFrame,
    synthetic_groups: pd.Series,
) -> pd.DataFrame:
    rows = []
    for _, row in diagnostics_df.iterrows():
        group_name = row["group_name"]
        observed_mean = _within_group_feature_distance_mean(observed_features, observed_groups, group_name)
        synthetic_mean = _within_group_feature_distance_mean(synthetic_features, synthetic_groups, group_name)
        row = row.to_dict()
        row["observed_within_feature_distance_mean"] = observed_mean
        row["synthetic_within_feature_distance_mean"] = synthetic_mean
        if np.isfinite(observed_mean) and observed_mean != 0 and np.isfinite(synthetic_mean):
            row["within_feature_distance_mean_ratio"] = synthetic_mean / observed_mean
        else:
            row["within_feature_distance_mean_ratio"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _write_taxon_csv(table: pd.DataFrame, path: Path) -> None:
    out = table.copy()
    out.index.name = "Taxon"
    out.reset_index().to_csv(path, index=False)


def _rwct_scenario_diagnostics(scenario: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "presence_diff_after_rwct": scenario.get("presence_diff", scenario.get("presence_diff_after_rwct", np.nan)),
        "max_taxon_degree_diff_after_rwct": scenario.get(
            "max_taxon_degree_diff",
            scenario.get("max_taxon_degree_diff_after_rwct", np.nan),
        ),
        "max_function_degree_diff_after_rwct": scenario.get(
            "max_function_degree_diff",
            scenario.get("max_function_degree_diff_after_rwct", np.nan),
        ),
        "rwct_shift_max": scenario.get("shift_max", scenario.get("rwct_shift_max", np.nan)),
        "rwct_lib_drift_med": scenario.get("lib_drift_med", scenario.get("rwct_lib_drift_med", np.nan)),
    }


def _add_rwct_diagnostics(row: Dict[str, Any], scenario: Dict[str, Any]) -> None:
    for key, value in _rwct_scenario_diagnostics(scenario).items():
        row[key] = value


def _scenario_with_rwct_fields(mode: str, scenario: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "mode": mode,
        "gamma": float(scenario["gamma"]),
        "effect_level": float(scenario["gamma"]),
        "dm": scenario["dm"],
    }
    row.update(_rwct_scenario_diagnostics(scenario))
    return row


def _scenario_rows_from_metrics(
    *,
    n_per_group: int,
    scenario_index: int,
    scenario: Dict[str, Any],
    metrics: Dict[str, float],
    include_gamma: bool,
) -> Dict[str, Any]:
    row = {
        "n_per_group": n_per_group,
        "scenario_index": scenario_index,
        "mode": scenario["mode"],
        "effect_level": scenario["effect_level"],
        "num_swaps": np.nan,
        "power": metrics["power"],
        "true_omega2": metrics["true_omega2"],
        "mean_boot_omega2": metrics["mean_boot_omega2"],
        "failed_bootstraps": metrics["failed_bootstraps"],
    }
    for key in [
        "pool_size_per_group",
        "draw_n_per_group",
        "expected_unique_per_group",
        "expected_duplicate_slots_per_group",
        "expected_duplicate_fraction",
    ]:
        if key in metrics:
            row[key] = metrics[key]
    if include_gamma:
        row["gamma"] = scenario["gamma"]
        _add_rwct_diagnostics(row, scenario)
    return row


def _evaluate_minimum_n(
    *,
    args: argparse.Namespace,
    scenarios: List[Dict[str, Any]],
    group_map: pd.Series,
    min_n: int,
    max_n: int,
    out_dir: Path,
    workflow: str,
    include_gamma: bool,
) -> Tuple[Optional[int], pd.DataFrame, pd.DataFrame]:
    failure_log_path = out_dir / "permanova_failures.jsonl"
    if failure_log_path.exists():
        failure_log_path.unlink()
    fitted_curve_dir = out_dir / "fitted_curves"

    def evaluate_fn(n_per_group: int, search_stage: str) -> pd.DataFrame:
        print(f"  Evaluating semi-synthetic scenarios at n_per_group={n_per_group}")

        def _evaluate(idx: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
            metrics = summarize_distance_metrics_with_replacement(
                dm=scenario["dm"],
                group_map=group_map,
                boot_number=args.boot_number,
                alpha=args.alpha,
                n_jobs=1,
                random_seed=args.random_seed + n_per_group * 10_000 + idx,
                n_per_group=n_per_group,
                permutations=args.permutations,
                omega2_floor=getattr(args, "omega2_floor", None),
                failure_log_path=failure_log_path,
                failure_context={
                    "workflow": workflow,
                    "stage": search_stage,
                    "n_per_group": n_per_group,
                    "scenario_index": idx,
                    "effect_level": float(scenario["effect_level"]),
                },
            )
            return _scenario_rows_from_metrics(
                n_per_group=n_per_group,
                scenario_index=idx,
                scenario=scenario,
                metrics=metrics,
                include_gamma=include_gamma,
            )

        rows = Parallel(n_jobs=args.n_jobs)(
            delayed(_evaluate)(idx, scenario)
            for idx, scenario in enumerate(scenarios)
            if scenario["dm"] is not None
        )
        return pd.DataFrame(rows)

    return core.estimate_minimum_sample_size(
        target_power=args.target_power,
        target_omega2=args.target_omega2,
        alpha=args.alpha,
        min_n=min_n,
        max_n=max_n,
        curve_plot_dir=fitted_curve_dir,
        evaluate_scenarios_for_n_fn=evaluate_fn,
    )


def _write_outputs(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    workflow: str,
    scenarios: List[Dict[str, Any]],
    diagnostics_df: pd.DataFrame,
    group_map: pd.Series,
    min_n: int,
    max_n: int,
    minimum_n: Optional[int],
    power_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    include_gamma: bool,
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_search_rows = []
    for scenario in scenarios:
        row = {
            "mode": scenario["mode"],
            "effect_level": scenario["effect_level"],
            "num_swaps": np.nan,
            "true_omega2": scenario["true_omega2"],
            "mean_boot_omega2": scenario["mean_boot_omega2"],
            "power_full_sample": scenario["power_full_sample"],
            "failed_bootstraps": scenario["failed_bootstraps"],
        }
        if include_gamma:
            row["gamma"] = scenario["gamma"]
            _add_rwct_diagnostics(row, scenario)
        scenario_search_rows.append(row)
    scenario_search_df = pd.DataFrame(scenario_search_rows)
    scenario_search_df.to_csv(out_dir / "scenario_search.csv", index=False)
    power_df.to_csv(out_dir / "power_by_sample_size.csv", index=False)
    metrics_df.to_csv(out_dir / "scenario_metrics_by_sample_size.csv", index=False)
    diagnostics_df.to_csv(out_dir / "synthetic_diagnostics.csv", index=False)

    valid_scenarios = [s for s in scenarios if np.isfinite(s["true_omega2"])]
    selected = min(valid_scenarios, key=lambda s: abs(s["true_omega2"] - args.target_omega2)) if valid_scenarios else None
    observed_n = int(diagnostics_df["observed_n"].min()) if not diagnostics_df.empty else np.nan
    pool_size = int(diagnostics_df["synthetic_pool_size"].min()) if not diagnostics_df.empty else np.nan
    summary = {
        "mode": workflow,
        "estimation_method": "large_pool_semisynthetic_with_replacement_bootstrap",
        "minimum_n_per_group": minimum_n,
        "target_power": args.target_power,
        "target_omega2": args.target_omega2,
        "alpha": args.alpha,
        "boot_number": args.boot_number,
        "min_n": min_n,
        "max_n": max_n,
        "observed_full_sample_n_per_group": observed_n,
        "synthetic_pool_size_per_group": pool_size,
        "sample_generator": (
            f"taxon_function_{args.protein_generator}_semisynthetic_v2"
            if workflow == "taxon-function" and hasattr(args, "protein_generator")
            else "pilot_semisynthetic_lightweight_v1"
        ),
        "nearest_full_sample_mode": None if selected is None else selected["mode"],
        "nearest_full_sample_effect_level": None if selected is None else float(selected["effect_level"]),
        "nearest_full_sample_true_omega2": None if selected is None else float(selected["true_omega2"]),
        "nearest_full_sample_mean_boot_omega2": None if selected is None else float(selected["mean_boot_omega2"]),
        "nearest_full_sample_power": None if selected is None else float(selected["power_full_sample"]),
        "failed_bootstraps_full_sample": int(scenario_search_df["failed_bootstraps"].sum()) if "failed_bootstraps" in scenario_search_df else 0,
        "failed_bootstraps_sample_size_search": int(metrics_df["failed_bootstraps"].sum()) if "failed_bootstraps" in metrics_df else 0,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    core.print_minimum_n_result(workflow, 0.0, 0.0, minimum_n, min_n, max_n)
    return summary


def run_taxon(args: argparse.Namespace) -> Dict[str, Any]:
    core._require_gene_runtime()
    core.load_core_runtime()
    args.out.mkdir(parents=True, exist_ok=True)
    group_map = _read_group_map(args.group)
    table, aligned_group_map = _read_taxon_feature_table(args.table, group_map)
    observed_n = int(aligned_group_map.value_counts().min())
    max_n = _resolve_max_n(observed_n, args.max_n)
    min_n = max(int(args.min_n), 2)
    pool_size = _resolve_pool_size(observed_n, max_n, args.pool_size_per_group)
    synthetic_table, synthetic_group_map, diagnostics_df = generate_taxon_pool(
        table,
        aligned_group_map,
        pool_size_per_group=pool_size,
        random_seed=args.random_seed,
        between_scale=args.between_scale,
        residual_scale=args.residual_scale,
        noise_multiplier=args.noise_multiplier,
    )

    effect_levels = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, args.increase_max, num=max(2, args.increase_num)),
                np.linspace(0.0, args.decrease_min, num=max(2, args.decrease_num)),
            ]
        )
    )
    effect_levels.sort()
    scenarios: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="phylopower_semisyn_taxon_") as tmp:
        work_dir = Path(tmp)
        synthetic_csv = work_dir / "synthetic_table.csv"
        _write_taxon_csv(synthetic_table, synthetic_csv)
        table_qza, tree_qza, tax_qza = core.prepare_qza_inputs(synthetic_csv, args.tree, args.taxonomy, work_dir)
        baseline_dm = core.compute_gemelli_rpca_distance(
            table_qza_path=table_qza,
            tree_qza_path=tree_qza,
            taxonomy_qza_path=tax_qza,
            out_dir=work_dir,
            qiime_env_name=args.qiime_env,
            output_stem="semisynthetic_baseline",
        )
        if baseline_dm is None:
            raise RuntimeError("Failed to compute semi-synthetic baseline Gemelli distance matrix.")
        full_n = int(synthetic_group_map.value_counts().min())
        for idx, level in enumerate(effect_levels):
            dm = baseline_dm if np.isclose(level, 0.0) else core.modulate_taxon_gemelli_geometry(
                baseline_dm,
                synthetic_group_map,
                float(level),
            )
            metrics = summarize_distance_metrics_without_replacement(
                dm=dm,
                group_map=synthetic_group_map,
                boot_number=args.boot_number,
                alpha=args.alpha,
                n_jobs=args.n_jobs,
                random_seed=args.random_seed + 10_000 + idx,
                n_per_group=full_n,
                permutations=args.permutations,
                omega2_floor=args.omega2_floor,
            )
            scenarios.append(
                {
                    "mode": "enhancement" if level >= 0 else "dilution",
                    "effect_level": float(level),
                    "dm": dm,
                    "true_omega2": metrics["true_omega2"],
                    "mean_boot_omega2": metrics["mean_boot_omega2"],
                    "power_full_sample": metrics["power"],
                    "failed_bootstraps": metrics["failed_bootstraps"],
                }
            )

    minimum_n, power_df, metrics_df = _evaluate_minimum_n(
        args=args,
        scenarios=scenarios,
        group_map=synthetic_group_map,
        min_n=min_n,
        max_n=max_n,
        out_dir=args.out,
        workflow="taxon",
        include_gamma=False,
    )
    summary = _write_outputs(
        args=args,
        out_dir=args.out,
        workflow="taxon",
        scenarios=scenarios,
        diagnostics_df=diagnostics_df,
        group_map=synthetic_group_map,
        min_n=min_n,
        max_n=max_n,
        minimum_n=minimum_n,
        power_df=power_df,
        metrics_df=metrics_df,
        include_gamma=False,
    )
    return {"summary": summary, "output_dir": args.out}


def run_taxon_function(args: argparse.Namespace) -> Dict[str, Any]:
    core._require_pro_rwct_runtime()
    core.load_core_runtime()
    args.out.mkdir(parents=True, exist_ok=True)
    group_map = _read_group_map(args.group)
    long_df, aligned_group_map = _read_protein_long_table(args.table, group_map)
    observed_n = int(aligned_group_map.value_counts().min())
    max_n = _resolve_max_n(observed_n, args.max_n)
    min_n = max(int(args.min_n), 2)
    pool_size = _resolve_pool_size(observed_n, max_n, args.pool_size_per_group)
    synthetic_df, synthetic_group_map, diagnostics_df = generate_taxon_function_pool(
        long_df,
        aligned_group_map,
        pool_size_per_group=pool_size,
        random_seed=args.random_seed,
        between_scale=args.between_scale,
        residual_scale=args.residual_scale,
        noise_multiplier=args.noise_multiplier,
        detection_slope=args.detection_slope,
        protein_generator=args.protein_generator,
    )

    increase_levels = np.linspace(0.0, args.increase_max, num=max(2, args.increase_num))
    decrease_levels = np.linspace(0.0, args.decrease_min, num=max(2, args.decrease_num))
    increase_scenarios, decrease_scenarios = core.precompute_rwct_distance_scenarios(
        df_scenario=synthetic_df,
        tree_path_str=str(args.tree),
        group_map=synthetic_group_map,
        enhancement_gammas=increase_levels,
        dilution_gammas=decrease_levels,
        ridge=args.ridge,
        normalize_weights=not args.no_normalize_weights,
    )

    scenarios: List[Dict[str, Any]] = []
    full_n = int(synthetic_group_map.value_counts().min())
    for mode, scenario_list in [("enhancement", increase_scenarios), ("dilution", decrease_scenarios)]:
        for idx, scenario in enumerate(scenario_list):
            metrics = summarize_distance_metrics_without_replacement(
                dm=scenario["dm"],
                group_map=synthetic_group_map,
                boot_number=args.boot_number,
                alpha=args.alpha,
                n_jobs=args.n_jobs,
                random_seed=args.random_seed + (11_000 if mode == "enhancement" else 22_000) + idx,
                n_per_group=full_n,
                permutations=args.permutations,
            )
            scenario_row = _scenario_with_rwct_fields(mode, scenario)
            scenario_row.update(
                {
                    "true_omega2": metrics["true_omega2"],
                    "mean_boot_omega2": metrics["mean_boot_omega2"],
                    "power_full_sample": metrics["power"],
                    "failed_bootstraps": metrics["failed_bootstraps"],
                }
            )
            scenarios.append(scenario_row)

    minimum_n, power_df, metrics_df = _evaluate_minimum_n(
        args=args,
        scenarios=scenarios,
        group_map=synthetic_group_map,
        min_n=min_n,
        max_n=max_n,
        out_dir=args.out,
        workflow="taxon-function",
        include_gamma=True,
    )
    summary = _write_outputs(
        args=args,
        out_dir=args.out,
        workflow="taxon-function",
        scenarios=scenarios,
        diagnostics_df=diagnostics_df,
        group_map=synthetic_group_map,
        min_n=min_n,
        max_n=max_n,
        minimum_n=minimum_n,
        power_df=power_df,
        metrics_df=metrics_df,
        include_gamma=True,
    )
    return {"summary": summary, "output_dir": args.out}


def _plot_resampling_comparison(
    results_df: pd.DataFrame,
    out_path: Path,
    workflow_label: str,
    target_omega2: float,
    target_power: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns = sorted(results_df["n_per_group"].unique())
    fig, axes = plt.subplots(1, len(ns), figsize=(6.2 * len(ns), 4.9), sharey=True)
    if len(ns) == 1:
        axes = [axes]
    colors = {"with_replacement": "#d95f02", "without_replacement": "#1b9e77"}
    labels = {
        "with_replacement": "with replacement",
        "without_replacement": "without replacement",
    }
    for ax, n_per_group in zip(axes, ns):
        subset = results_df[results_df["n_per_group"] == n_per_group]
        for resampling in ["with_replacement", "without_replacement"]:
            line_df = subset[subset["resampling"] == resampling].sort_values("true_omega2")
            if line_df.empty:
                continue
            ax.scatter(
                line_df["true_omega2"],
                line_df["power"],
                s=34,
                color=colors[resampling],
                alpha=0.70,
                edgecolors="white",
                linewidths=0.5,
                label=labels[resampling],
                zorder=3,
            )
            fit_input = line_df[["true_omega2", "power", "mode"]].copy()
            fit_result = core.fit_sigmoid_curve(fit_input, alpha=core.SIGMOID_DECISION_ANCHOR_POWER)
            params = fit_result.get("params")
            if params:
                x_max = max(
                    target_omega2 * 1.25,
                    float(np.nanmax(line_df["true_omega2"])) * 1.08,
                    0.08,
                )
                x_grid = np.linspace(0.0, x_max, 400)
                y_grid = core.anchored_sigmoid_curve(
                    x_grid,
                    float(params["k"]),
                    float(params["x0"]),
                    core.SIGMOID_DECISION_ANCHOR_POWER,
                )
                ax.plot(
                    x_grid,
                    y_grid,
                    color=colors[resampling],
                    linewidth=2.6,
                    alpha=0.95,
                    zorder=2,
                )
                ax.text(
                    0.98,
                    0.08 if resampling == "with_replacement" else 0.16,
                    f"{labels[resampling]} fit: {fit_result.get('status')}",
                    color=colors[resampling],
                    transform=ax.transAxes,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                )
        ax.scatter(
            [0.0],
            [core.SIGMOID_DECISION_ANCHOR_POWER],
            s=46,
            color="black",
            marker="D",
            label="anchor (0, alpha)" if n_per_group == ns[0] else "_nolegend_",
            zorder=4,
        )
        ax.axvline(target_omega2, color="red", linestyle="--", linewidth=1.0)
        ax.axhline(target_power, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{workflow_label}: n={n_per_group}")
        ax.set_xlabel("true omega^2")
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("power")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _compare_resampling_for_scenarios(
    *,
    args: argparse.Namespace,
    workflow: str,
    workflow_label: str,
    scenarios: List[Dict[str, Any]],
    group_map: pd.Series,
    small_n: int,
    large_n: int,
    include_gamma: bool,
    diagnostics_df: pd.DataFrame,
) -> Dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    failure_log_path = args.out / "compare_permanova_failures.jsonl"
    if failure_log_path.exists():
        failure_log_path.unlink()

    rows: List[Dict[str, Any]] = []
    if getattr(args, "resampling_mode", "both") == "both":
        resampling_modes = ["with_replacement", "without_replacement"]
    else:
        resampling_modes = [args.resampling_mode]

    for n_per_group in [small_n, large_n]:
        for scenario_index, scenario in enumerate(scenarios):
            for resampling in resampling_modes:
                print(
                    f"  [{workflow}] n={n_per_group} {resampling} "
                    f"{scenario['mode']} effect={scenario['effect_level']:+.4g}"
                )
                metrics = summarize_distance_metrics_by_resampling(
                    resampling=resampling,
                    dm=scenario["dm"],
                    group_map=group_map,
                    boot_number=args.boot_number,
                    alpha=args.alpha,
                    n_jobs=args.n_jobs,
                    random_seed=args.random_seed
                    + (10_000 if resampling == "with_replacement" else 20_000)
                    + n_per_group * 1_000
                    + scenario_index,
                    n_per_group=n_per_group,
                    permutations=args.permutations,
                    omega2_floor=getattr(args, "omega2_floor", None),
                    failure_log_path=failure_log_path,
                    failure_context={
                        "workflow": workflow,
                        "stage": "resampling_compare",
                        "resampling": resampling,
                        "n_per_group": n_per_group,
                        "scenario_index": scenario_index,
                        "effect_level": float(scenario["effect_level"]),
                    },
                )
                row = {
                    "workflow": workflow,
                    "n_per_group": n_per_group,
                    "resampling": resampling,
                    "scenario_index": scenario_index,
                    "mode": scenario["mode"],
                    "effect_level": scenario["effect_level"],
                    "true_omega2": metrics["true_omega2"],
                    "power": metrics["power"],
                    "mean_boot_omega2": metrics["mean_boot_omega2"],
                    "failed_bootstraps": metrics["failed_bootstraps"],
                }
                for key in [
                    "pool_size_per_group",
                    "draw_n_per_group",
                    "expected_unique_per_group",
                    "expected_duplicate_slots_per_group",
                    "expected_duplicate_fraction",
                ]:
                    if key in metrics:
                        row[key] = metrics[key]
                if include_gamma:
                    row["gamma"] = scenario["gamma"]
                    _add_rwct_diagnostics(row, scenario)
                rows.append(row)

    results_df = pd.DataFrame(rows)
    results_path = args.out / f"{workflow}_resampling_compare.csv"
    diagnostics_path = args.out / f"{workflow}_synthetic_diagnostics.csv"
    plot_path = args.out / f"{workflow}_omega_power_resampling_compare.png"
    results_df.to_csv(results_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)
    _plot_resampling_comparison(
        results_df,
        out_path=plot_path,
        workflow_label=workflow_label,
        target_omega2=args.target_omega2,
        target_power=args.target_power,
    )
    summary = {
        "mode": workflow,
        "estimation_method": "semisynthetic_pool_resampling_compare",
        "small_n": small_n,
        "large_n": large_n,
        "target_power": args.target_power,
        "target_omega2": args.target_omega2,
        "boot_number": args.boot_number,
        "permutations": args.permutations,
        "sigmoid_x0_constrained_nonnegative": True,
        "results_csv": str(results_path),
        "diagnostics_csv": str(diagnostics_path),
        "plot": str(plot_path),
    }
    if hasattr(args, "protein_generator"):
        summary["protein_generator"] = args.protein_generator
    (args.out / f"{workflow}_resampling_compare_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return {"summary": summary, "output_dir": args.out}


def run_compare_taxon(args: argparse.Namespace) -> Dict[str, Any]:
    core._require_gene_runtime()
    core.load_core_runtime()
    args.out.mkdir(parents=True, exist_ok=True)
    group_map = _read_group_map(args.group)
    table, aligned_group_map = _read_taxon_feature_table(args.table, group_map)
    observed_n = int(aligned_group_map.value_counts().min())
    small_n = int(args.small_n if args.small_n is not None else observed_n)
    large_n = int(args.large_n if args.large_n is not None else _resolve_max_n(observed_n, args.max_n))
    max_needed_n = max(small_n, large_n, _resolve_max_n(observed_n, args.max_n))
    pool_size = _resolve_pool_size(observed_n, max_needed_n, args.pool_size_per_group)
    synthetic_table, synthetic_group_map, diagnostics_df = generate_taxon_pool(
        table,
        aligned_group_map,
        pool_size_per_group=pool_size,
        random_seed=args.random_seed,
        between_scale=args.between_scale,
        residual_scale=args.residual_scale,
        noise_multiplier=args.noise_multiplier,
    )
    effect_levels = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, args.increase_max, num=max(2, args.increase_num)),
                np.linspace(0.0, args.decrease_min, num=max(2, args.decrease_num)),
            ]
        )
    )
    effect_levels.sort()
    scenarios: List[Dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="phylopower_semisyn_compare_taxon_") as tmp:
        work_dir = Path(tmp)
        synthetic_csv = work_dir / "synthetic_table.csv"
        _write_taxon_csv(synthetic_table, synthetic_csv)
        table_qza, tree_qza, tax_qza = core.prepare_qza_inputs(synthetic_csv, args.tree, args.taxonomy, work_dir)
        baseline_dm = core.compute_gemelli_rpca_distance(
            table_qza_path=table_qza,
            tree_qza_path=tree_qza,
            taxonomy_qza_path=tax_qza,
            out_dir=work_dir,
            qiime_env_name=args.qiime_env,
            output_stem="semisynthetic_compare_baseline",
        )
        if baseline_dm is None:
            raise RuntimeError("Failed to compute semi-synthetic baseline Gemelli distance matrix.")
        for level in effect_levels:
            dm = baseline_dm if np.isclose(level, 0.0) else core.modulate_taxon_gemelli_geometry(
                baseline_dm,
                synthetic_group_map,
                float(level),
            )
            scenarios.append(
                {
                    "mode": "enhancement" if level >= 0 else "dilution",
                    "effect_level": float(level),
                    "dm": dm,
                }
            )
    return _compare_resampling_for_scenarios(
        args=args,
        workflow="taxon",
        workflow_label="Taxon",
        scenarios=scenarios,
        group_map=synthetic_group_map,
        small_n=small_n,
        large_n=large_n,
        include_gamma=False,
        diagnostics_df=diagnostics_df,
    )


def run_compare_taxon_function(args: argparse.Namespace) -> Dict[str, Any]:
    core._require_pro_rwct_runtime()
    core.load_core_runtime()
    args.out.mkdir(parents=True, exist_ok=True)
    group_map = _read_group_map(args.group)
    long_df, aligned_group_map = _read_protein_long_table(args.table, group_map)
    observed_n = int(aligned_group_map.value_counts().min())
    small_n = int(args.small_n if args.small_n is not None else observed_n)
    large_n = int(args.large_n if args.large_n is not None else _resolve_max_n(observed_n, args.max_n))
    max_needed_n = max(small_n, large_n, _resolve_max_n(observed_n, args.max_n))
    pool_size = _resolve_pool_size(observed_n, max_needed_n, args.pool_size_per_group)
    synthetic_df, synthetic_group_map, diagnostics_df = generate_taxon_function_pool(
        long_df,
        aligned_group_map,
        pool_size_per_group=pool_size,
        random_seed=args.random_seed,
        between_scale=args.between_scale,
        residual_scale=args.residual_scale,
        noise_multiplier=args.noise_multiplier,
        detection_slope=args.detection_slope,
        protein_generator=args.protein_generator,
    )
    increase_levels = np.linspace(0.0, args.increase_max, num=max(2, args.increase_num))
    decrease_levels = np.linspace(0.0, args.decrease_min, num=max(2, args.decrease_num))
    increase_scenarios, decrease_scenarios = core.precompute_rwct_distance_scenarios(
        df_scenario=synthetic_df,
        tree_path_str=str(args.tree),
        group_map=synthetic_group_map,
        enhancement_gammas=increase_levels,
        dilution_gammas=decrease_levels,
        ridge=args.ridge,
        normalize_weights=not args.no_normalize_weights,
    )
    scenarios: List[Dict[str, Any]] = []
    for mode, scenario_list in [("enhancement", increase_scenarios), ("dilution", decrease_scenarios)]:
        for scenario in scenario_list:
            scenarios.append(_scenario_with_rwct_fields(mode, scenario))
    return _compare_resampling_for_scenarios(
        args=args,
        workflow="taxon-function",
        workflow_label="Taxon-function",
        scenarios=scenarios,
        group_map=synthetic_group_map,
        small_n=small_n,
        large_n=large_n,
        include_gamma=True,
        diagnostics_df=diagnostics_df,
    )


def _plot_pilot_sensitivity(
    results_df: pd.DataFrame,
    out_path: Path,
    workflow_label: str,
    target_omega2: float,
    target_power: float,
    alpha: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pilot_ns = sorted(results_df["pilot_n"].unique())
    palette = {
        pilot_n: color
        for pilot_n, color in zip(
            pilot_ns,
            ["#0F766E", "#C2410C", "#1D4ED8", "#7C3AED", "#B45309"],
        )
    }
    fig, ax = plt.subplots(1, 1, figsize=(8.4, 6.2))
    for pilot_n in pilot_ns:
        subset = results_df[results_df["pilot_n"] == pilot_n].sort_values("true_omega2")
        color = palette[pilot_n]
        ax.scatter(
            subset["true_omega2"],
            subset["power"],
            s=38,
            alpha=0.68,
            color=color,
            edgecolors="white",
            linewidths=0.6,
            label=f"pilot n={pilot_n}",
            zorder=3,
        )
        fit_subset = subset
        if "usable_for_fit" in subset:
            candidate = subset[subset["usable_for_fit"].astype(bool)]
            if candidate["true_omega2"].nunique(dropna=True) >= 4:
                fit_subset = candidate
        fit_result = core.fit_sigmoid_curve(
            fit_subset[["true_omega2", "power", "mode"]].copy(),
            alpha=alpha,
        )
        params = fit_result.get("params")
        if params:
            x_max = max(target_omega2 * 1.25, float(np.nanmax(subset["true_omega2"])) * 1.08, 0.08)
            x_grid = np.linspace(0.0, x_max, 500)
            y_grid = core.anchored_sigmoid_curve(
                x_grid,
                float(params["k"]),
                float(params["x0"]),
                alpha,
            )
            ax.plot(x_grid, y_grid, color=color, linewidth=2.6, alpha=0.96, zorder=2)
    ax.scatter([0.0], [alpha], s=52, color="black", marker="D", label="anchor (0, alpha)", zorder=4)
    ax.axvline(target_omega2, color="red", linestyle="--", linewidth=1.1)
    ax.axhline(target_power, color="gray", linestyle=":", linewidth=1.1)
    ax.set_title(f"{workflow_label}: pilot-size sensitivity")
    ax.set_xlabel("true omega^2")
    ax.set_ylabel("power")
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.22)
    ax.legend(title="Pilot subset", frameon=False, fontsize=10, title_fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def _count_transition_points(power_values: pd.Series, alpha: float) -> int:
    values = pd.to_numeric(power_values, errors="coerce").dropna()
    return int(((values > alpha) & (values < 0.95)).sum())


def _as_finite_float(value: Any) -> Optional[float]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None


def _row_passes_realistic_filter(row: Any, args: argparse.Namespace) -> bool:
    """Keep fit inputs in topology-preserving and not-too-distorted RWCT regions."""

    getter = row.get if hasattr(row, "get") else lambda key, default=None: default
    for key in [
        "presence_diff_after_rwct",
        "max_taxon_degree_diff_after_rwct",
        "max_function_degree_diff_after_rwct",
    ]:
        value = _as_finite_float(getter(key, np.nan))
        if value is not None and not np.isclose(value, 0.0):
            return False

    shift_max = _as_finite_float(getter("rwct_shift_max", np.nan))
    max_log_shift = _as_finite_float(getattr(args, "max_log_shift", np.nan))
    if shift_max is not None and max_log_shift is not None and shift_max > max_log_shift:
        return False

    lib_drift = _as_finite_float(getter("rwct_lib_drift_med", np.nan))
    max_lib_drift = _as_finite_float(getattr(args, "max_lib_drift", np.nan))
    if lib_drift is not None and max_lib_drift is not None and abs(lib_drift) > max_lib_drift:
        return False
    return True


def _annotate_fit_usability(results_df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if results_df.empty:
        return results_df
    out = results_df.copy()
    powers = pd.to_numeric(out["power"], errors="coerce")
    omega2 = pd.to_numeric(out["true_omega2"], errors="coerce")
    fit_power_min = float(getattr(args, "fit_power_min", 0.15))
    fit_power_max = float(getattr(args, "fit_power_max", 0.95))
    target_omega2 = float(getattr(args, "target_omega2", np.nan))
    omega_margin = max(target_omega2 * 0.75, 0.02) if np.isfinite(target_omega2) else 0.02
    realistic = out.apply(lambda row: _row_passes_realistic_filter(row, args), axis=1)
    transition = powers.between(fit_power_min, fit_power_max, inclusive="both")
    near_target = np.isfinite(omega2) & (np.abs(omega2 - target_omega2) <= omega_margin)
    fit_filter = str(getattr(args, "fit_filter", "none"))
    out["realistic_for_fit"] = realistic.astype(bool)
    if fit_filter == "none":
        out["usable_for_fit"] = (np.isfinite(powers) & np.isfinite(omega2)).astype(bool)
    elif fit_filter == "realistic":
        out["usable_for_fit"] = (realistic & np.isfinite(powers) & np.isfinite(omega2)).astype(bool)
    else:
        out["usable_for_fit"] = (realistic & (transition | near_target)).astype(bool)
    return out


def _unique_sorted_levels(levels: Sequence[float], *, decreasing: bool = False) -> np.ndarray:
    unique: List[float] = []
    for level in levels:
        value = float(level)
        if not any(np.isclose(value, existing) for existing in unique):
            unique.append(value)
    unique.sort(reverse=decreasing)
    return np.asarray(unique, dtype=float)


def _select_refine_levels_from_rows(
    coarse_rows: Sequence[Dict[str, Any]],
    *,
    args: argparse.Namespace,
    mode: str,
    existing_levels: Sequence[float],
) -> np.ndarray:
    if not coarse_rows:
        return np.asarray([], dtype=float)
    df = pd.DataFrame(coarse_rows)
    if df.empty or "mode" not in df:
        return np.asarray([], dtype=float)
    df = df[df["mode"] == mode].copy()
    if df.empty:
        return np.asarray([], dtype=float)
    df["effect_level"] = pd.to_numeric(df["effect_level"], errors="coerce")
    df["power"] = pd.to_numeric(df["power"], errors="coerce")
    df["true_omega2"] = pd.to_numeric(df["true_omega2"], errors="coerce")
    df = df[np.isfinite(df["effect_level"]) & np.isfinite(df["power"])]
    if len(df) < 2:
        return np.asarray([], dtype=float)

    df["realistic_for_fit"] = df.apply(lambda row: _row_passes_realistic_filter(row, args), axis=1)
    df = df.sort_values("effect_level", ascending=(mode != "dilution"))
    fit_power_min = float(getattr(args, "fit_power_min", 0.15))
    fit_power_max = float(getattr(args, "fit_power_max", 0.95))
    target_power = float(args.target_power)
    target_omega2 = float(args.target_omega2)
    candidates: List[Tuple[float, float]] = []
    records = df.to_dict("records")
    for left, right in zip(records, records[1:]):
        a = float(left["effect_level"])
        b = float(right["effect_level"])
        if np.isclose(a, b):
            continue
        if not (bool(left.get("realistic_for_fit", True)) and bool(right.get("realistic_for_fit", True))):
            continue
        p1 = float(left["power"])
        p2 = float(right["power"])
        o1 = _as_finite_float(left.get("true_omega2"))
        o2 = _as_finite_float(right.get("true_omega2"))
        endpoint_transition = (
            fit_power_min <= p1 <= fit_power_max
            or fit_power_min <= p2 <= fit_power_max
        )
        crosses_power = (p1 - target_power) * (p2 - target_power) <= 0.0
        crosses_omega = (
            o1 is not None
            and o2 is not None
            and (o1 - target_omega2) * (o2 - target_omega2) <= 0.0
        )
        if not (endpoint_transition or crosses_power or crosses_omega):
            continue
        midpoint = (a + b) / 2.0
        if any(np.isclose(midpoint, existing) for existing in existing_levels):
            continue
        score = 0.0
        score += 2.5 if crosses_power else 0.0
        score += 1.8 if crosses_omega else 0.0
        score += 1.0 if endpoint_transition else 0.0
        score -= min(abs(p1 - target_power), abs(p2 - target_power))
        if o1 is not None and o2 is not None:
            score -= 0.25 * min(abs(o1 - target_omega2), abs(o2 - target_omega2)) / max(target_omega2, 1e-6)
        candidates.append((score, midpoint))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [level for _, level in candidates[: max(0, int(getattr(args, "refine_num", 0)))]]
    return _unique_sorted_levels(selected, decreasing=(mode == "dilution"))


def _run_pilot_sensitivity_rows(
    *,
    args: argparse.Namespace,
    workflow: str,
    scenarios: List[Dict[str, Any]],
    synthetic_group_map: pd.Series,
    pilot_n: int,
    eval_n: int,
    include_gamma: bool,
    scenario_index_offset: int = 0,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    failure_log_path = args.out / f"{workflow}_pilot{pilot_n}_permanova_failures.jsonl"
    if scenario_index_offset == 0 and failure_log_path.exists():
        failure_log_path.unlink()
    for scenario_index, scenario in enumerate(scenarios):
        scenario_index_global = int(scenario_index_offset + scenario_index)
        print(
            f"  [{workflow}] pilot_n={pilot_n} eval_n={eval_n} "
            f"with_replacement {scenario['mode']} effect={scenario['effect_level']:+.4g}"
        )
        metrics = summarize_distance_metrics_with_replacement(
            dm=scenario["dm"],
            group_map=synthetic_group_map,
            boot_number=args.boot_number,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            random_seed=args.random_seed + pilot_n * 100_000 + eval_n * 1_000 + scenario_index_global,
            n_per_group=eval_n,
            permutations=args.permutations,
            omega2_floor=getattr(args, "omega2_floor", None),
            failure_log_path=failure_log_path,
            failure_context={
                "workflow": workflow,
                "stage": "pilot_sensitivity",
                "pilot_n": pilot_n,
                "eval_n": eval_n,
                "scenario_index": scenario_index_global,
                "effect_level": float(scenario["effect_level"]),
            },
        )
        row = {
            "workflow": workflow,
            "pilot_n": pilot_n,
            "eval_n": eval_n,
            "resampling": "with_replacement",
            "scenario_index": scenario_index_global,
            "mode": scenario["mode"],
            "effect_level": scenario["effect_level"],
            "true_omega2": metrics["true_omega2"],
            "power": metrics["power"],
            "mean_boot_omega2": metrics["mean_boot_omega2"],
            "failed_bootstraps": metrics["failed_bootstraps"],
        }
        for key in [
            "pool_size_per_group",
            "draw_n_per_group",
            "expected_unique_per_group",
            "expected_duplicate_slots_per_group",
            "expected_duplicate_fraction",
        ]:
            if key in metrics:
                row[key] = metrics[key]
        if include_gamma:
            row["gamma"] = scenario["gamma"]
            _add_rwct_diagnostics(row, scenario)
        rows.append(row)
    return rows


def _write_pilot_sensitivity_outputs(
    *,
    args: argparse.Namespace,
    workflow: str,
    workflow_label: str,
    rows: List[Dict[str, Any]],
    diagnostics_frames: List[pd.DataFrame],
) -> Dict[str, Any]:
    results_df = _annotate_fit_usability(pd.DataFrame(rows), args)
    diagnostics_df = pd.concat(diagnostics_frames, ignore_index=True) if diagnostics_frames else pd.DataFrame()
    results_path = args.out / f"{workflow}_pilot_sensitivity.csv"
    diagnostics_path = args.out / f"{workflow}_pilot_sensitivity_diagnostics.csv"
    plot_path = args.out / f"{workflow}_pilot_sensitivity_omega_power.png"
    results_df.to_csv(results_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)
    _plot_pilot_sensitivity(
        results_df,
        out_path=plot_path,
        workflow_label=workflow_label,
        target_omega2=args.target_omega2,
        target_power=args.target_power,
        alpha=args.alpha,
    )
    summary = {
        "mode": workflow,
        "estimation_method": "pilot_subset_semisynthetic_with_replacement_sensitivity",
        "pilot_ns": sorted(results_df["pilot_n"].unique().tolist()),
        "eval_n": int(results_df["eval_n"].iloc[0]) if not results_df.empty else None,
        "target_power": args.target_power,
        "target_omega2": args.target_omega2,
        "boot_number": args.boot_number,
        "permutations": args.permutations,
        "results_csv": str(results_path),
        "diagnostics_csv": str(diagnostics_path),
        "plot": str(plot_path),
    }
    if not results_df.empty and "pilot_n" in results_df:
        transition_counts = {
            str(pilot_n): _count_transition_points(subset["power"], args.alpha)
            for pilot_n, subset in results_df.groupby("pilot_n")
        }
        summary["transition_point_counts"] = transition_counts
        usable_transition_counts = {
            str(pilot_n): _count_transition_points(subset[subset["usable_for_fit"]]["power"], args.alpha)
            for pilot_n, subset in results_df.groupby("pilot_n")
        }
        summary["usable_transition_point_counts"] = usable_transition_counts
        summary["fit_not_identifiable"] = any(count < 4 for count in transition_counts.values())
    if hasattr(args, "calibrate_to_real"):
        summary["calibrated_to_real"] = bool(args.calibrate_to_real)
        summary["calibration_boot_number"] = int(args.calibration_boot_number)
        summary["calibration_between_scales"] = args.calibration_between_scales
        summary["calibration_residual_scales"] = args.calibration_residual_scales
        summary["calibration_noise_multipliers"] = args.calibration_noise_multipliers
        if hasattr(args, "calibration_detection_slopes"):
            summary["calibration_detection_slopes"] = args.calibration_detection_slopes
    if hasattr(args, "protein_generator"):
        summary["protein_generator"] = args.protein_generator
    if hasattr(args, "engine"):
        summary["engine"] = str(args.engine)
    if hasattr(args, "effect_grid"):
        summary["effect_grid"] = args.effect_grid
        summary["coarse_increase_num"] = int(getattr(args, "coarse_increase_num", 0))
        summary["coarse_decrease_num"] = int(getattr(args, "coarse_decrease_num", 0))
        summary["refine_num"] = int(getattr(args, "refine_num", 0))
        summary["fit_power_min"] = float(getattr(args, "fit_power_min", np.nan))
        summary["fit_power_max"] = float(getattr(args, "fit_power_max", np.nan))
        summary["fit_filter"] = str(getattr(args, "fit_filter", "none"))
        summary["max_lib_drift"] = float(getattr(args, "max_lib_drift", np.nan))
        summary["max_log_shift"] = float(getattr(args, "max_log_shift", np.nan))
    if not diagnostics_df.empty and "reference_true_omega2" in diagnostics_df:
        for source_col, summary_key in [
            ("reference_true_omega2", "reference_true_omega2"),
            ("reference_power", "reference_power"),
            ("reference_mean_boot_omega2", "reference_mean_boot_omega2"),
        ]:
            values = diagnostics_df[source_col].dropna() if source_col in diagnostics_df else pd.Series(dtype=float)
            if not values.empty:
                summary[summary_key] = float(values.iloc[0])
    (args.out / f"{workflow}_pilot_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return {"summary": summary, "output_dir": args.out}


def _compute_rwct_scenario_for_gamma(
    df_scenario: pd.DataFrame,
    tree_path_str: str,
    group_map: pd.Series,
    gamma: float,
    ridge: float,
    normalize_weights: bool,
) -> Dict[str, Any]:
    core.load_core_runtime()
    if np.isclose(gamma, 0.0):
        dm = core.compute_phylofunc_distance_matrix(df_scenario, tree_path_str)
        return {
            "gamma": float(gamma),
            "dm": dm,
            "presence_diff": 0,
            "max_taxon_degree_diff": 0,
            "max_function_degree_diff": 0,
            "shift_max": 0.0,
            "lib_drift_med": 0.0,
        }
    effected, info = core.apply_rwct_effect_scaling(
        df_scenario.copy(),
        float(gamma),
        group_map,
        ridge=ridge,
        normalize_weights=normalize_weights,
    )
    degree = core.summarize_degree_preservation(df_scenario, effected)
    dm = core.compute_phylofunc_distance_matrix(effected, tree_path_str)
    return {"gamma": float(gamma), "dm": dm, **info, **degree}


def precompute_rwct_distance_scenarios_parallel(
    df_scenario: pd.DataFrame,
    tree_path_str: str,
    group_map: pd.Series,
    enhancement_gammas: np.ndarray,
    dilution_gammas: np.ndarray,
    ridge: float,
    normalize_weights: bool,
    n_jobs: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    print("\n" + "=" * 70, flush=True)
    print("Precomputing full-sample PhyloFunc distance matrices (RWCT, parallel)", flush=True)
    print("=" * 70, flush=True)
    base_dm = core.compute_phylofunc_distance_matrix(df_scenario, tree_path_str)
    base_scenario = {
        "gamma": 0.0,
        "dm": base_dm,
        "presence_diff": 0,
        "max_taxon_degree_diff": 0,
        "max_function_degree_diff": 0,
        "shift_max": 0.0,
        "lib_drift_med": 0.0,
    }

    def _build(gamma: float) -> Dict[str, Any]:
        if np.isclose(gamma, 0.0):
            return dict(base_scenario)
        return _compute_rwct_scenario_for_gamma(
            df_scenario=df_scenario,
            tree_path_str=tree_path_str,
            group_map=group_map,
            gamma=float(gamma),
            ridge=ridge,
            normalize_weights=normalize_weights,
        )

    increase_scenarios = Parallel(n_jobs=n_jobs)(
        delayed(_build)(float(gamma)) for gamma in enhancement_gammas
    )
    decrease_scenarios = Parallel(n_jobs=n_jobs)(
        delayed(_build)(float(gamma)) for gamma in dilution_gammas
    )
    return increase_scenarios, decrease_scenarios


def _calibration_score(metrics: Dict[str, float], reference: Dict[str, float]) -> float:
    omega_ref = max(float(reference.get("true_omega2", np.nan)), 0.02)
    boot_omega_ref = max(float(reference.get("mean_boot_omega2", np.nan)), 0.02)
    omega_part = abs(float(metrics.get("true_omega2", np.nan)) - float(reference.get("true_omega2", np.nan))) / omega_ref
    boot_omega_part = (
        abs(float(metrics.get("mean_boot_omega2", np.nan)) - float(reference.get("mean_boot_omega2", np.nan)))
        / boot_omega_ref
    )
    power_part = abs(float(metrics.get("power", np.nan)) - float(reference.get("power", np.nan)))
    parts = [omega_part, 0.35 * boot_omega_part, 2.0 * power_part]
    if any(not np.isfinite(part) for part in parts):
        return float("inf")
    return float(sum(parts))


def _base_generator_params(args: argparse.Namespace, workflow: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {
        "between_scale": float(args.between_scale),
        "residual_scale": float(args.residual_scale),
        "noise_multiplier": float(args.noise_multiplier),
    }
    if workflow == "taxon-function":
        params["detection_slope"] = float(args.detection_slope)
        params["protein_generator"] = str(args.protein_generator)
    return params


def _candidate_generator_params(args: argparse.Namespace, workflow: str) -> List[Dict[str, Any]]:
    if not getattr(args, "calibrate_to_real", False):
        return [_base_generator_params(args, workflow)]
    between_scales = _parse_float_list(args.calibration_between_scales)
    residual_scales = _parse_float_list(args.calibration_residual_scales)
    noise_multipliers = _parse_float_list(args.calibration_noise_multipliers)
    detection_slopes = (
        _parse_float_list(args.calibration_detection_slopes)
        if workflow == "taxon-function"
        else [0.0]
    )
    candidates: List[Dict[str, Any]] = []
    for between_scale in between_scales:
        for residual_scale in residual_scales:
            for noise_multiplier in noise_multipliers:
                for detection_slope in detection_slopes:
                    params: Dict[str, Any] = {
                        "between_scale": float(between_scale),
                        "residual_scale": float(residual_scale),
                        "noise_multiplier": float(noise_multiplier),
                    }
                    if workflow == "taxon-function":
                        params["detection_slope"] = float(detection_slope)
                        params["protein_generator"] = str(args.protein_generator)
                    candidates.append(params)
    return candidates


def _annotate_calibration(
    diagnostics_df: pd.DataFrame,
    *,
    calibrated: bool,
    reference_metrics: Optional[Dict[str, float]],
    selected_metrics: Optional[Dict[str, float]],
    score: Optional[float],
) -> pd.DataFrame:
    diagnostics_df = diagnostics_df.copy()
    diagnostics_df["calibrated_to_real"] = bool(calibrated)
    diagnostics_df["calibration_score"] = np.nan if score is None else float(score)
    if reference_metrics:
        for key, value in reference_metrics.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                diagnostics_df[f"reference_{key}"] = float(value)
    if selected_metrics:
        for key, value in selected_metrics.items():
            if isinstance(value, (int, float, np.floating)) and np.isfinite(value):
                diagnostics_df[f"selected_baseline_{key}"] = float(value)
    return diagnostics_df


def _compute_real_metrics_from_dm(
    *,
    dm: Any,
    group_map: pd.Series,
    args: argparse.Namespace,
    eval_n: int,
    workflow: str,
) -> Dict[str, float]:
    failure_log_path = args.out / f"{workflow}_real_reference_permanova_failures.jsonl"
    if failure_log_path.exists():
        failure_log_path.unlink()
    metrics = summarize_distance_metrics_with_replacement(
        dm=dm,
        group_map=group_map,
        boot_number=args.calibration_boot_number,
        alpha=args.alpha,
        n_jobs=args.n_jobs,
        random_seed=args.random_seed + 777_001,
        n_per_group=eval_n,
        permutations=args.permutations,
        omega2_floor=getattr(args, "omega2_floor", None),
        failure_log_path=failure_log_path,
        failure_context={"workflow": workflow, "stage": "real_reference", "eval_n": eval_n},
    )
    metrics["reference_boot_number"] = int(args.calibration_boot_number)
    return metrics


def _select_taxon_pool_for_sensitivity(
    *,
    args: argparse.Namespace,
    pilot_table: pd.DataFrame,
    pilot_group_map: pd.Series,
    pool_size: int,
    pilot_n: int,
    eval_n: int,
    reference_metrics: Optional[Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, Any]:
    best: Optional[Tuple[float, Dict[str, float], pd.DataFrame, pd.Series, pd.DataFrame, Any, Dict[str, float]]] = None
    candidates = _candidate_generator_params(args, "taxon")
    for candidate_index, params in enumerate(candidates):
        synthetic_table, synthetic_group_map, diagnostics_df = generate_taxon_pool(
            pilot_table,
            pilot_group_map,
            pool_size_per_group=pool_size,
            random_seed=args.random_seed + pilot_n * 10 + candidate_index,
            **params,
        )
        with tempfile.TemporaryDirectory(prefix=f"phylopower_cal_taxon_n{pilot_n}_") as tmp:
            work_dir = Path(tmp)
            synthetic_csv = work_dir / "synthetic_table.csv"
            _write_taxon_csv(synthetic_table, synthetic_csv)
            table_qza, tree_qza, tax_qza = core.prepare_qza_inputs(synthetic_csv, args.tree, args.taxonomy, work_dir)
            baseline_dm = core.compute_gemelli_rpca_distance(
                table_qza_path=table_qza,
                tree_qza_path=tree_qza,
                taxonomy_qza_path=tax_qza,
                out_dir=work_dir,
                qiime_env_name=args.qiime_env,
                output_stem=f"pilot{pilot_n}_candidate{candidate_index}_baseline",
            )
        if baseline_dm is None:
            continue
        if reference_metrics is None:
            metrics = summarize_distance_metrics_with_replacement(
                dm=baseline_dm,
                group_map=synthetic_group_map,
                boot_number=max(10, min(args.calibration_boot_number, args.boot_number)),
                alpha=args.alpha,
                n_jobs=args.n_jobs,
                random_seed=args.random_seed + pilot_n * 500_000 + candidate_index,
                n_per_group=eval_n,
                permutations=args.permutations,
                omega2_floor=getattr(args, "omega2_floor", None),
            )
            score = 0.0
        else:
            metrics = summarize_distance_metrics_with_replacement(
                dm=baseline_dm,
                group_map=synthetic_group_map,
                boot_number=args.calibration_boot_number,
                alpha=args.alpha,
                n_jobs=args.n_jobs,
                random_seed=args.random_seed + pilot_n * 500_000 + candidate_index,
                n_per_group=eval_n,
                permutations=args.permutations,
                omega2_floor=getattr(args, "omega2_floor", None),
            )
            score = _calibration_score(metrics, reference_metrics)
        if best is None or score < best[0]:
            best = (score, params, synthetic_table, synthetic_group_map, diagnostics_df, baseline_dm, metrics)
        if not getattr(args, "calibrate_to_real", False):
            break
    if best is None:
        raise RuntimeError(f"Failed to build calibrated taxon synthetic pool for pilot_n={pilot_n}.")
    score, params, synthetic_table, synthetic_group_map, diagnostics_df, baseline_dm, metrics = best
    print(
        f"  [taxon] selected generator for pilot_n={pilot_n}: "
        f"between={params['between_scale']}, residual={params['residual_scale']}, "
        f"noise={params['noise_multiplier']}, baseline omega2={metrics['true_omega2']:.4g}, "
        f"power={metrics['power']:.3g}, score={score:.3g}"
    )
    diagnostics_df = _annotate_calibration(
        diagnostics_df,
        calibrated=getattr(args, "calibrate_to_real", False),
        reference_metrics=reference_metrics,
        selected_metrics=metrics,
        score=score,
    )
    return synthetic_table, synthetic_group_map, diagnostics_df, baseline_dm


def _select_taxon_function_pool_for_sensitivity(
    *,
    args: argparse.Namespace,
    pilot_df: pd.DataFrame,
    pilot_group_map: pd.Series,
    pool_size: int,
    pilot_n: int,
    eval_n: int,
    reference_metrics: Optional[Dict[str, float]],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    best: Optional[Tuple[float, Dict[str, Any], pd.DataFrame, pd.Series, pd.DataFrame, Dict[str, float]]] = None
    candidates = _candidate_generator_params(args, "taxon-function")
    for candidate_index, params in enumerate(candidates):
        synthetic_df, synthetic_group_map, diagnostics_df = generate_taxon_function_pool(
            pilot_df,
            pilot_group_map,
            pool_size_per_group=pool_size,
            random_seed=args.random_seed + pilot_n * 10 + candidate_index,
            **params,
        )
        baseline_dm = core.compute_phylofunc_distance_matrix(synthetic_df, str(args.tree))
        if reference_metrics is None:
            metrics = summarize_distance_metrics_with_replacement(
                dm=baseline_dm,
                group_map=synthetic_group_map,
                boot_number=max(10, min(args.calibration_boot_number, args.boot_number)),
                alpha=args.alpha,
                n_jobs=args.n_jobs,
                random_seed=args.random_seed + pilot_n * 500_000 + candidate_index,
                n_per_group=eval_n,
                permutations=args.permutations,
            )
            score = 0.0
        else:
            metrics = summarize_distance_metrics_with_replacement(
                dm=baseline_dm,
                group_map=synthetic_group_map,
                boot_number=args.calibration_boot_number,
                alpha=args.alpha,
                n_jobs=args.n_jobs,
                random_seed=args.random_seed + pilot_n * 500_000 + candidate_index,
                n_per_group=eval_n,
                permutations=args.permutations,
            )
            score = _calibration_score(metrics, reference_metrics)
        if best is None or score < best[0]:
            best = (score, params, synthetic_df, synthetic_group_map, diagnostics_df, metrics)
        if not getattr(args, "calibrate_to_real", False):
            break
    if best is None:
        raise RuntimeError(f"Failed to build calibrated taxon-function synthetic pool for pilot_n={pilot_n}.")
    score, params, synthetic_df, synthetic_group_map, diagnostics_df, metrics = best
    print(
        f"  [taxon-function] selected generator for pilot_n={pilot_n}: "
        f"generator={params['protein_generator']}, between={params['between_scale']}, residual={params['residual_scale']}, "
        f"noise={params['noise_multiplier']}, detection={params['detection_slope']}, "
        f"baseline omega2={metrics['true_omega2']:.4g}, power={metrics['power']:.3g}, score={score:.3g}"
    )
    diagnostics_df = _annotate_calibration(
        diagnostics_df,
        calibrated=getattr(args, "calibrate_to_real", False),
        reference_metrics=reference_metrics,
        selected_metrics=metrics,
        score=score,
    )
    return synthetic_df, synthetic_group_map, diagnostics_df


def run_sensitivity_taxon(args: argparse.Namespace) -> Dict[str, Any]:
    core._require_gene_runtime()
    core.load_core_runtime()
    args.out.mkdir(parents=True, exist_ok=True)
    group_map = _read_group_map(args.group)
    table, aligned_group_map = _read_taxon_feature_table(args.table, group_map)
    full_n = int(aligned_group_map.value_counts().min())
    pilot_ns = _parse_int_list(args.pilot_ns) if isinstance(args.pilot_ns, str) else list(args.pilot_ns)
    eval_n = int(args.eval_n if args.eval_n is not None else full_n)
    effect_levels = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, args.increase_max, num=max(2, args.increase_num)),
                np.linspace(0.0, args.decrease_min, num=max(2, args.decrease_num)),
            ]
        )
    )
    effect_levels.sort()
    all_rows: List[Dict[str, Any]] = []
    diagnostics_frames: List[pd.DataFrame] = []
    reference_metrics: Optional[Dict[str, float]] = None
    if args.calibrate_to_real:
        with tempfile.TemporaryDirectory(prefix="phylopower_taxon_real_reference_") as tmp:
            work_dir = Path(tmp)
            real_csv = work_dir / "real_table.csv"
            _write_taxon_csv(table, real_csv)
            table_qza, tree_qza, tax_qza = core.prepare_qza_inputs(real_csv, args.tree, args.taxonomy, work_dir)
            real_dm = core.compute_gemelli_rpca_distance(
                table_qza_path=table_qza,
                tree_qza_path=tree_qza,
                taxonomy_qza_path=tax_qza,
                out_dir=work_dir,
                qiime_env_name=args.qiime_env,
                output_stem="real_reference",
            )
        if real_dm is None:
            raise RuntimeError("Failed to compute Gemelli distance matrix for real taxon reference.")
        reference_metrics = _compute_real_metrics_from_dm(
            dm=real_dm,
            group_map=aligned_group_map,
            args=args,
            eval_n=eval_n,
            workflow="taxon",
        )
        print(
            f"  [taxon] real reference at eval_n={eval_n}: "
            f"omega2={reference_metrics['true_omega2']:.4g}, power={reference_metrics['power']:.3g}"
        )
    if getattr(args, "engine", "ordination") in {"ordination", "gemelli-loading"}:
        engine = getattr(args, "engine", "ordination")
        for pilot_n in pilot_ns:
            pilot_group_map = _subsample_group_map(aligned_group_map, pilot_n, args.random_seed + pilot_n)
            pilot_table = table[pilot_group_map.index].copy()
            pool_size = int(args.pool_size_per_group) if args.pool_size_per_group is not None else max(500, 50 * eval_n)
            if engine == "gemelli-loading":
                coords = _compute_gemelli_loading_coordinates(pilot_table, args.tree)
                model = _build_coordinate_model(
                    coords,
                    pilot_group_map,
                    center_mode=getattr(args, "center_mode", "debiased"),
                    coordinate_source="gemelli_matrix_completion_u",
                )
            else:
                with tempfile.TemporaryDirectory(prefix=f"phylopower_taxon_ordination_n{pilot_n}_") as tmp:
                    work_dir = Path(tmp)
                    pilot_csv = work_dir / "pilot_table.csv"
                    _write_taxon_csv(pilot_table, pilot_csv)
                    table_qza, tree_qza, tax_qza = core.prepare_qza_inputs(pilot_csv, args.tree, args.taxonomy, work_dir)
                    pilot_dm = core.compute_gemelli_rpca_distance(
                        table_qza_path=table_qza,
                        tree_qza_path=tree_qza,
                        taxonomy_qza_path=tax_qza,
                        out_dir=work_dir,
                        qiime_env_name=args.qiime_env,
                        output_stem=f"pilot{pilot_n}_ordination",
                    )
                if pilot_dm is None:
                    raise RuntimeError(f"Failed to compute Gemelli distance matrix for taxon pilot_n={pilot_n}.")
                pilot_dm_df = _as_distance_frame(pilot_dm)
                model = _build_ordination_model(
                    pilot_dm_df,
                    pilot_group_map,
                    center_mode=getattr(args, "center_mode", "debiased"),
                )
            pool_gm = _ordination_pool_group_map(model, pool_size)
            if getattr(args, "effect_grid", "fixed") == "omega-uniform":
                scales = _ordination_scales_by_omega_preview(
                    args,
                    eval_n,
                    model,
                    pool_size,
                    pool_gm,
                    seed=args.random_seed + pilot_n * 10000,
                )
            elif getattr(args, "effect_grid", "fixed") == "power-adaptive":
                scales = _ordination_scales_by_power_preview(
                    args,
                    eval_n,
                    model,
                    pool_size,
                    pool_gm,
                    seed=args.random_seed + pilot_n * 10000,
                    workflow="taxon",
                )
            elif getattr(args, "effect_grid", "fixed") == "power-uniform":
                scales = _ordination_scales_by_power_uniform(
                    args,
                    eval_n,
                    model,
                    pool_size,
                    pool_gm,
                    seed=args.random_seed + pilot_n * 10000,
                    workflow="taxon",
                )
            else:
                scales = _ordination_scales_for_eval_n(args, eval_n, model=model)
            print(
                f"  [taxon] pilot_n={pilot_n} {engine} engine: "
                f"{len(model['groups'])} groups, {model['k']} axes, pool={pool_size}/group, "
                f"scales={len(scales)} ({float(np.min(scales)):.3g}..{float(np.max(scales)):.3g})"
            )
            scenarios: List[Dict[str, Any]] = []
            for sc in scales:
                dm = _ordination_pool_dm(model, float(sc), pool_size, args.random_seed + pilot_n * 1000 + int(sc * 1000))
                scenarios.append(
                    {
                        "mode": "enhancement" if sc >= 1.0 else "dilution",
                        "effect_level": float(sc),
                        "dm": dm,
                    }
                )
            diagnostics_frames.append(pd.DataFrame([{
                "group_name": str(g), "pilot_n": pilot_n, "eval_n": eval_n,
                "engine": engine, "pco_axes": model["k"], "pool_size": pool_size,
                "coordinate_source": model.get("coordinate_source"),
                "center_mode": model.get("center_mode"), "center_shrinkage": model.get("center_shrinkage"),
                "pilot_target_omega2": model.get("pilot_target_omega2"),
                "ledoit_wolf_cov": True, "effect_grid": getattr(args, "effect_grid", "fixed"),
                "ordination_scale_count": len(scales), "ordination_scale_min": float(np.min(scales)),
                "ordination_scale_max": float(np.max(scales)),
            } for g in model["groups"]]))
            all_rows.extend(
                _run_pilot_sensitivity_rows(
                    args=args,
                    workflow="taxon",
                    scenarios=scenarios,
                    synthetic_group_map=pool_gm,
                    pilot_n=pilot_n,
                    eval_n=eval_n,
                    include_gamma=False,
                )
            )
        return _write_pilot_sensitivity_outputs(
            args=args,
            workflow="taxon",
            workflow_label="Taxon",
            rows=all_rows,
            diagnostics_frames=diagnostics_frames,
        )

    for pilot_n in pilot_ns:
        pilot_group_map = _subsample_group_map(aligned_group_map, pilot_n, args.random_seed + pilot_n)
        pilot_table = table[pilot_group_map.index].copy()
        pool_size = _resolve_pool_size(pilot_n, eval_n, args.pool_size_per_group)
        synthetic_table, synthetic_group_map, diagnostics_df, baseline_dm = _select_taxon_pool_for_sensitivity(
            args=args,
            pilot_table=pilot_table,
            pilot_group_map=pilot_group_map,
            pool_size=pool_size,
            pilot_n=pilot_n,
            eval_n=eval_n,
            reference_metrics=reference_metrics,
        )
        diagnostics_df["pilot_n"] = pilot_n
        diagnostics_df["eval_n"] = eval_n
        diagnostics_frames.append(diagnostics_df)
        scenarios: List[Dict[str, Any]] = []
        for level in effect_levels:
            dm = baseline_dm if np.isclose(level, 0.0) else core.modulate_taxon_gemelli_geometry(
                baseline_dm,
                synthetic_group_map,
                float(level),
            )
            scenarios.append(
                {
                    "mode": "enhancement" if level >= 0 else "dilution",
                    "effect_level": float(level),
                    "dm": dm,
                }
            )
        all_rows.extend(
            _run_pilot_sensitivity_rows(
                args=args,
                workflow="taxon",
                scenarios=scenarios,
                synthetic_group_map=synthetic_group_map,
                pilot_n=pilot_n,
                eval_n=eval_n,
                include_gamma=False,
            )
        )
    return _write_pilot_sensitivity_outputs(
        args=args,
        workflow="taxon",
        workflow_label="Taxon",
        rows=all_rows,
        diagnostics_frames=diagnostics_frames,
    )


def _safeguard_lower_omega2(dm: Any, group_map: pd.Series, *, n_boot: int, seed: int, conf: float = 0.60) -> Tuple[float, float]:
    """Pilot effect size and its safeguard lower bound (Perugini et al. 2014).

    Returns (point omega^2, lower-bound omega^2). The lower bound is the (1-conf)/2 quantile of a
    case-resampling bootstrap of the pilot's own omega^2 -- a conservative effect-size target that
    protects against pilots that happen to look more separated than the population.
    """

    point = float(core.compute_omega2(dm, group_map))
    dm_df = _as_distance_frame(dm)
    ids = list(dm_df.index)
    gm = group_map.loc[ids]
    rng = np.random.default_rng(int(seed))
    groups = sorted(gm.dropna().unique())
    boots: List[float] = []
    for _ in range(int(n_boot)):
        picked: List[str] = []
        for g in groups:
            members = gm[gm == g].index.to_numpy()
            picked.extend(rng.choice(members, size=len(members), replace=True).tolist())
        sub = dm_df.loc[picked, picked]
        boots.append(float(core.compute_omega2(sub, gm.loc[picked])))
    boots = np.asarray([b for b in boots if np.isfinite(b)], dtype=float)
    lower = float(np.quantile(boots, (1.0 - conf) / 2.0)) if boots.size else point
    return point, max(lower, 0.0)


def _calibrate_protein_pool_to_pilot(
    *,
    args: argparse.Namespace,
    pilot_df: pd.DataFrame,
    pilot_group_map: pd.Series,
    pool_size: int,
    pilot_n: int,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, float, float]:
    """Build a synthetic pool whose effect size matches the pilot's own (bias-corrected) omega^2.

    The generator otherwise reproduces the pilot's *raw* separation, which -- placed in a large pool
    where omega^2's small-sample correction no longer applies -- systematically inflates the effect
    size (verified: pilot omega^2 ~0.04 -> pool omega^2 ~0.12 at n=5). We remove that artifact by
    grid-searching a single separation scalar ``s`` (driving both between_scale and presence_retention)
    so the pool's omega^2 lands on the pilot target. The target defaults to the pilot's point omega^2,
    or its safeguard lower bound when --safeguard-power is set. Self-contained: uses only the pilot.
    """

    pilot_dm = core.compute_phylofunc_distance_matrix(pilot_df, str(args.tree))
    point, lower = _safeguard_lower_omega2(
        pilot_dm, pilot_group_map, n_boot=int(getattr(args, "calibration_boot_number", 50)),
        seed=args.random_seed + pilot_n,
    )
    target = lower if getattr(args, "safeguard_power", False) else point
    target = max(float(target), 0.0)

    s_grid = np.linspace(0.0, 1.0, int(getattr(args, "calibration_grid", 6)))
    best = None
    for s in s_grid:
        syn_df, sgm, diag = generate_taxon_function_pool(
            pilot_df, pilot_group_map, pool_size_per_group=pool_size,
            random_seed=args.random_seed + pilot_n * 10,
            between_scale=float(s), residual_scale=args.residual_scale,
            noise_multiplier=args.noise_multiplier, detection_slope=args.detection_slope,
            protein_generator=args.protein_generator, presence_retention=float(s),
        )
        dm = core.compute_phylofunc_distance_matrix(syn_df, str(args.tree))
        pool_omega = float(core.compute_omega2(dm, sgm))
        score = abs(pool_omega - target)
        if best is None or score < best[0]:
            best = (score, float(s), syn_df, sgm, diag, pool_omega)
    score, s_best, syn_df, sgm, diag, pool_omega = best
    print(
        f"  [taxon-function] pilot_n={pilot_n} calibrated to pilot omega2: "
        f"target={target:.4g} (point={point:.4g}, lower={lower:.4g}), "
        f"selected s={s_best:.2f} -> pool omega2={pool_omega:.4g}"
    )
    diag = diag.copy()
    diag["calibrated_to_pilot"] = True
    diag["pilot_target_omega2"] = target
    diag["pilot_point_omega2"] = point
    diag["pilot_lower_omega2"] = lower
    diag["calibration_separation_scale"] = s_best
    diag["calibrated_pool_omega2"] = pool_omega
    return syn_df, sgm, diag, target, s_best


def _pcoa_coords(dm_df: pd.DataFrame, n_axes: Optional[int] = None) -> pd.DataFrame:
    """Classical PCoA embedding: Euclidean coordinates whose distances reproduce dm_df
    (positive-eigenvalue axes only). Lets us resample in a space where Euclidean distance
    equals the original phylogeny-aware distance.

    If `n_axes` is given, retain only the top-`n_axes` positive-eigenvalue axes (largest variance).
    This standardises the embedding dimension across pilots of different sizes: a small pilot
    yields only ~2n-1 PCoA axes, and fewer axes concentrate the effect -> inflated power-per-omega^2.
    Fixing the dimension removes that pilot-size artefact so the power curves align across pilots.
    """
    ids = list(dm_df.index)
    D = dm_df.to_numpy(dtype=float)
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    order = np.argsort(w)[::-1]
    w = w[order]
    V = V[:, order]
    keep = w > 1e-8
    coords = V[:, keep] * np.sqrt(w[keep])
    if n_axes is not None and n_axes > 0:
        coords = coords[:, :int(n_axes)]      # top axes are first (eigenvalues sorted descending)
    return pd.DataFrame(coords, index=ids)


def _analytical_nonlinear_shrinkage_cov(resid: np.ndarray) -> np.ndarray:
    """Analytical nonlinear shrinkage covariance (Ledoit & Wolf 2020, Annals of Statistics).

    Unlike linear (Ledoit-Wolf 2004) shrinkage, which pulls every eigenvalue toward a common
    mean (isotropic shrink toward scaled identity), this individually corrects each sample
    eigenvalue -- raising the underestimated small ones and lowering the inflated large ones --
    via a kernel estimate of the sample spectral density and its Hilbert transform. That is the
    right correction in the n ~ p and p > n regime where pilot covariances live, and it is what
    fixes the small-pilot optimism (under-estimated within-group spread). Closed form, no tuning.

    `resid`: (n, p) de-meaned within-group coordinates. Returns a (p, p) PSD covariance.
    """
    X = np.asarray(resid, dtype=float)
    n, p = X.shape
    if n < 2:
        return np.eye(p) * 1e-9
    sample = (X.T @ X) / n
    sample = (sample + sample.T) / 2.0
    lam_all, u = np.linalg.eigh(sample)              # ascending, p eigenpairs
    lam = lam_all[max(0, p - n):]                    # trailing m = min(p, n) eigenvalues
    sqrt5 = np.sqrt(5.0)
    tol = max(float(lam.max()), 1e-12) * 1e-6
    lam = np.maximum(lam, tol)                       # guard zeros (rank deficiency at small n)
    m = lam.shape[0]
    L = np.tile(lam.reshape(-1, 1), (1, m))
    h = n ** (-1.0 / 3.0)
    H = h * L.T
    x = (L - L.T) / H
    ftilde = (3.0 / 4.0 / sqrt5) * np.mean(np.maximum(1.0 - x ** 2 / 5.0, 0.0) / H, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        Hftemp = ((-3.0 / 10.0 / np.pi) * x
                  + (3.0 / 4.0 / sqrt5 / np.pi) * (1.0 - x ** 2 / 5.0)
                  * np.log(np.abs((sqrt5 - x) / (sqrt5 + x))))
    edge = np.abs(x) == sqrt5
    Hftemp[edge] = (-3.0 / 10.0 / np.pi) * x[edge]
    Hftilde = np.mean(Hftemp / H, axis=1)
    if p <= n:
        dtilde = lam / ((np.pi * (p / n) * lam * ftilde) ** 2
                        + (1.0 - (p / n) - np.pi * (p / n) * lam * Hftilde) ** 2)
    else:
        # abs() is the analytic continuation used consistently with the per-eigenvalue Hilbert
        # term above; it keeps the formula finite for small n (where sqrt(5)*h can exceed 1, i.e.
        # outside the large-dimensional asymptotic regime the closed form was derived for).
        Hftilde0 = ((1.0 / np.pi) * (3.0 / 10.0 / h ** 2 + 3.0 / 4.0 / sqrt5 / h
                    * (1.0 - 1.0 / 5.0 / h ** 2)
                    * np.log(np.abs((1.0 + sqrt5 * h) / (1.0 - sqrt5 * h)))) * np.mean(1.0 / lam))
        dtilde0 = 1.0 / (np.pi * (p - n) / n * Hftilde0)
        dtilde1 = lam / (np.pi ** 2 * lam ** 2 * (ftilde ** 2 + Hftilde ** 2))
        dtilde = np.concatenate([dtilde0 * np.ones(p - n), dtilde1])
    if not np.all(np.isfinite(dtilde)):
        raise FloatingPointError("nonlinear shrinkage produced non-finite eigenvalues")
    sigma = (u * dtilde) @ u.T
    return (sigma + sigma.T) / 2.0


def _estimate_group_cov(resid: np.ndarray, cov_estimator: str):
    """Per-group covariance with the requested shrinkage estimator.

    - ledoit-wolf:        linear Ledoit-Wolf (sklearn, /n MLE normalisation).
    - ledoit-wolf-ddof:   linear Ledoit-Wolf with an n/(n-1) trace de-bias. sklearn normalises by
                          n (MLE), which under-estimates the within-group spread by (n-1)/n -- a
                          per-pilot-size bias that systematically inflates power for small pilots
                          (n=5 -> x0.8). The n/(n-1) factor restores an unbiased trace at every n,
                          equalising the power curve across pilot sizes (validated: trace ratio
                          0.80/0.90/0.94 -> 1.00 at n=5/10/17).
    - nonlinear-shrinkage: analytical nonlinear shrinkage (Ledoit-Wolf 2020). NOT recommended for
                          small n / p>n / spiked spectra (our regime): controlled and real-data
                          tests show it under-estimates the within-group spread even more than
                          linear LW and worsens cross-pilot consistency. Kept for completeness.
    """
    from sklearn.covariance import ledoit_wolf

    n = int(np.asarray(resid).shape[0])
    if cov_estimator == "isotropic":
        # Diagnostic: isotropic sigma^2 * I gives EVERY pilot the same effective within-group
        # dimensionality (full rank k), removing the rank = n-1 deficiency that makes small pilots
        # over-concentrated and optimistic. sigma^2 = mean shrunk variance.
        cov, _ = ledoit_wolf(resid)
        p = cov.shape[0]
        return np.eye(p) * (float(np.trace(cov)) / max(p, 1))
    if cov_estimator == "nonlinear-shrinkage":
        try:
            return _analytical_nonlinear_shrinkage_cov(resid)
        except (FloatingPointError, np.linalg.LinAlgError, ValueError):
            pass
        cov, _ = ledoit_wolf(resid)
        return cov
    cov, _ = ledoit_wolf(resid)
    if cov_estimator == "ledoit-wolf-ddof" and n > 1:
        cov = cov * (n / (n - 1.0))
    return cov


def _estimate_pool_df(coords: pd.DataFrame, group_map: pd.Series, blocks: Dict[Any, Dict[str, Any]],
                      groups: list, *, n_draw: int = 500, seed: int = 0) -> Optional[float]:
    """Data-driven tail-heaviness: choose the multivariate-t degrees of freedom whose synthetic
    WITHIN-group pairwise-distance SD best matches the real one (Gaussian pools under-disperse the
    distances of real, heavy-tailed omics data). Returns None (Gaussian) if the data is light-tailed.
    Within-group pairwise distances depend only on the covariance, so mu cancels -> cheap & robust."""
    X = coords.to_numpy(dtype=float)
    labels = group_map.loc[coords.index].astype(str).to_numpy()
    iu = np.triu_indices(X.shape[0], 1)
    rd = squareform(pdist(X))[iu]
    same = labels[iu[0]] == labels[iu[1]]
    if same.sum() < 3:
        return None
    target = float(rd[same].std())
    if not np.isfinite(target) or target <= 0:
        return None
    rng = np.random.default_rng(seed)
    best_df, best_err = None, np.inf
    for df in [25, 20, 15, 12, 10, 8, 6]:
        sds = []
        for g in groups:
            cov = blocks[g]["cov"]
            if blocks[g]["n"] <= 1:
                continue
            p = cov.shape[0]
            if df is None:
                Z = rng.multivariate_normal(np.zeros(p), cov, n_draw)
            else:
                Zc = rng.multivariate_normal(np.zeros(p), cov * (df - 2) / df, n_draw)
                w = rng.chisquare(df, n_draw) / df
                Z = Zc / np.sqrt(w)[:, None]
            du = np.triu_indices(n_draw, 1)
            sds.append(squareform(pdist(Z))[du])
        if not sds:
            continue
        err = abs(float(np.concatenate(sds).std()) - target)
        if err < best_err:
            best_err, best_df = err, df
    return None if best_df is None else float(best_df)


def _build_coordinate_model(
    coords: pd.DataFrame,
    group_map: pd.Series,
    *,
    center_mode: str = "debiased",
    coordinate_source: str = "ordination",
    cov_estimator: str = "ledoit-wolf",
    pool_dist: str = "gaussian",
    pool_df: Optional[float] = None,
    pool_cov: bool = False,
    cov_eb_pool: bool = False,
) -> Dict[str, Any]:
    """Coordinate-space generative model from real sample coordinates.

    Per group estimates a centroid and a Ledoit-Wolf shrunk covariance
    (analytic intensity -> full rank even when n < dimensions, no tuning). The between-group
    centroid separation can be used as observed, or de-biased by a single GLOBAL positive-part factor
    s = sqrt(max(0, 1 - sum_a(var1_a/n1 + var2_a/n2) / sum_a delta_a^2)), which removes the
    small-sample inflation of the observed centroid distance while shrinking the offset uniformly.
    (A per-axis correction was tried but preferentially keeps low-within-variance axes -- the
    var/n threshold is easier to exceed there -- which pushes the effect onto low-variance
    directions and destabilises the power-per-omega^2 curve, badly so for near-null pilots; the
    global factor keeps the observed direction and is far more stable across pilot draws.)
    No dataset-specific constants -> generalises across datasets.
    """

    if center_mode not in {"observed", "debiased", "empirical-bayes", "omega-calibrated"}:
        raise ValueError(f"Unknown ordination center_mode={center_mode!r}")
    if cov_estimator not in {"ledoit-wolf", "ledoit-wolf-ddof", "nonlinear-shrinkage", "isotropic"}:
        raise ValueError(f"Unknown cov_estimator={cov_estimator!r}")

    X = coords.to_numpy(dtype=float)
    idx = {s: i for i, s in enumerate(coords.index)}
    groups = sorted(group_map.loc[coords.index].dropna().unique())
    gc = X.mean(axis=0)
    blocks: Dict[Any, Dict[str, Any]] = {}
    resid_all: List[np.ndarray] = []
    for g in groups:
        members = [s for s in group_map[group_map == g].index if s in idx]
        Xg = X[[idx[s] for s in members]]
        mu = Xg.mean(axis=0)
        resid = Xg - mu
        if len(members) > 1:
            cov = _estimate_group_cov(resid, cov_estimator)
            var = resid.var(axis=0, ddof=1)
            resid_all.append(resid)
        else:
            cov = np.eye(X.shape[1]) * 1e-9
            var = np.zeros(X.shape[1])
        blocks[g] = {"mu": mu, "n": len(members), "var": var, "cov": cov}
    if (pool_cov or cov_eb_pool) and resid_all:
        # Pooled within-group covariance (group-centered residuals stacked).
        #  - pool_cov: replace each group's Sigma with the common pooled Sigma (full pooling).
        #  - cov_eb_pool: shrink each group's Sigma TOWARD the pooled one by lambda = k/(k+n_g)
        #    (empirical-Bayes / partial pooling). lambda is data-driven (strong when n is small or
        #    dimension is high, i.e. when the per-group dispersion difference is least trustworthy),
        #    so it removes the noise-driven between-group dispersion difference without assuming it
        #    is exactly zero. No hardcoded constant.
        pooled = np.vstack(resid_all)
        cov_pooled = _estimate_group_cov(pooled, cov_estimator)
        var_pooled = pooled.var(axis=0, ddof=1)
        kdim = int(X.shape[1])
        for g in groups:
            if blocks[g]["n"] <= 1:
                continue
            if cov_eb_pool and not pool_cov:
                ng = float(blocks[g]["n"])
                lam = kdim / (kdim + ng)
                blocks[g]["cov"] = lam * cov_pooled + (1.0 - lam) * blocks[g]["cov"]
                blocks[g]["var"] = lam * var_pooled + (1.0 - lam) * blocks[g]["var"]
            else:
                blocks[g]["cov"] = cov_pooled
                blocks[g]["var"] = var_pooled
    def _global_shrink(delta: np.ndarray, bias: np.ndarray) -> float:
        return float(np.sqrt(max(0.0, 1.0 - float(np.sum(bias)) / max(float(np.sum(delta ** 2)), 1e-12))))

    def _eb_shrink(delta: np.ndarray, bias: np.ndarray) -> float:
        signal = max(float(np.sum(delta ** 2)), 0.0)
        noise = max(float(np.sum(bias)), 0.0)
        return float(np.sqrt(signal / max(signal + noise, 1e-12)))

    def _shrink(delta: np.ndarray, bias: np.ndarray) -> float:
        if center_mode == "observed":
            return 1.0
        if center_mode == "omega-calibrated":
            return 1.0
        if center_mode == "empirical-bayes":
            return _eb_shrink(delta, bias)
        return _global_shrink(delta, bias)

    pilot_target_omega2 = np.nan
    if len(groups) == 2:
        g0, g1 = groups
        delta = blocks[g0]["mu"] - blocks[g1]["mu"]
        bias = blocks[g0]["var"] / max(blocks[g0]["n"], 1) + blocks[g1]["var"] / max(blocks[g1]["n"], 1)
        s = _shrink(delta, bias)
        if center_mode == "omega-calibrated":
            pilot_dm = pd.DataFrame(squareform(pdist(X, metric="euclidean")), index=coords.index, columns=coords.index)
            pilot_target_omega2 = max(0.0, float(core.compute_omega2(pilot_dm, group_map.loc[coords.index])))
            d2 = max(float(np.sum(delta ** 2)), 1e-12)
            trace_sum = max(float(np.trace(blocks[g0]["cov"]) + np.trace(blocks[g1]["cov"])), 1e-12)
            target = min(max(pilot_target_omega2, 0.0), 0.999)
            s = float(np.sqrt((2.0 * trace_sum * target) / max((1.0 - target) * d2, 1e-12)))
            # Near-null pilot: the bias-corrected omega^2 floors to ~0 (common at very small n,
            # e.g. 5/group), which collapses the centroid offset to exactly zero -- the scale sweep
            # then has no direction to amplify and every scale yields the null (power ~ alpha),
            # leaving the power curve with no transition points. Fall back to the global positive-
            # part de-biased separation, and finally to the raw observed direction, so a usable
            # offset direction always survives. The plotted (omega^2, power) pairs are measured from
            # the pool, so this only reparametrises where scale=1 lands; it does not bias the curve.
            if not (s > 1e-6):
                s = _global_shrink(delta, bias)
            if not (s > 1e-6):
                s = 1.0
        n0 = float(blocks[g0]["n"])
        n1 = float(blocks[g1]["n"])
        total = max(n0 + n1, 1.0)
        debiased = {
            g0: gc + (n1 / total) * s * delta,
            g1: gc - (n0 / total) * s * delta,
        }
    else:
        # Multi-group: shrink each centroid offset toward the global centroid by one global factor.
        debiased = {}
        for g in groups:
            delta = blocks[g]["mu"] - gc
            bias = blocks[g]["var"] / max(blocks[g]["n"], 1)
            debiased[g] = gc + _shrink(delta, bias) * delta
        s = float("nan")
    if pool_dist == "student-t" and (pool_df is None or (isinstance(pool_df, str) and pool_df == "auto")):
        pool_df = _estimate_pool_df(coords, group_map, blocks, groups)   # data-driven tail-heaviness
    elif isinstance(pool_df, str):
        pool_df = None
    return {
        "groups": groups,
        "gc": gc,
        "blocks": blocks,
        "debiased": debiased,
        "k": int(X.shape[1]),
        "center_mode": center_mode,
        "center_shrinkage": float(s),
        "pilot_target_omega2": float(pilot_target_omega2) if np.isfinite(pilot_target_omega2) else np.nan,
        "coordinate_source": coordinate_source,
        "cov_estimator": cov_estimator,
        "pool_dist": pool_dist,
        "pool_df": float(pool_df) if pool_df is not None else None,
        "pool_cov": bool(pool_cov),
        "cov_eb_pool": bool(cov_eb_pool),
    }


def _build_ordination_model(
    dm_df: pd.DataFrame,
    group_map: pd.Series,
    *,
    center_mode: str = "debiased",
    cov_estimator: str = "ledoit-wolf",
    n_axes: Optional[int] = None,
    pool_dist: str = "gaussian",
    pool_df: Optional[float] = None,
    pool_cov: bool = False,
    cov_eb_pool: bool = False,
) -> Dict[str, Any]:
    """Ordination-space generative model from a REAL distance matrix (fully data-driven).

    Embeds via PCoA, then delegates to the coordinate-space MVN model.
    """

    coords = _pcoa_coords(dm_df, n_axes=n_axes)
    return _build_coordinate_model(
        coords,
        group_map,
        center_mode=center_mode,
        coordinate_source="pcoa_distance",
        cov_estimator=cov_estimator,
        pool_dist=pool_dist,
        pool_df=pool_df,
        pool_cov=pool_cov,
        cov_eb_pool=cov_eb_pool,
    )


def _effective_pool_size(base: int, eval_n: int, factor: int = 50) -> int:
    """Pool size must stay >> eval_n so the with-replacement bootstrap draws negligible duplicate
    (zero-distance) pairs -- a fixed pool inflates type-I / power at large eval_n (the duplicate-zero
    artifact). Scale the pool with the target sample size: pool = max(base, factor*eval_n)."""
    return int(max(int(base), int(factor) * int(eval_n)))


def _ordination_pool_group_map(model: Dict[str, Any], pool_size: int) -> pd.Series:
    labels: List[str] = []
    ids: List[str] = []
    for g in model["groups"]:
        labels.extend([str(g)] * pool_size)
        ids.extend([f"syn_{_safe_label(g)}_{i:05d}" for i in range(pool_size)])
    return pd.Series(labels, index=ids, name="group")


def _ordination_pool_dm(model: Dict[str, Any], scale: float, pool_size: int, seed: int) -> pd.DataFrame:
    """Generate a large synthetic pool by drawing fresh MVN points per group, with the between-group
    centroid offset scaled by `scale` (scale<1 dilutes, >1 enhances -- bidirectional), then return
    their Euclidean distance matrix. A large pool keeps later with-replacement bootstrap collisions
    (duplicate zero-distances) negligible."""
    rng = np.random.default_rng(int(seed))
    pool_dist = model.get("pool_dist", "gaussian")
    pool_df = model.get("pool_df", None)
    pts: List[np.ndarray] = []
    ids: List[str] = []
    for g in model["groups"]:
        mu = model["gc"] + float(scale) * (model["debiased"][g] - model["gc"])
        cov = model["blocks"][g]["cov"]
        if pool_dist == "student-t" and pool_df is not None and float(pool_df) > 2.0:
            # Multivariate-t with df=pool_df: mu + Z/sqrt(w), Z~N(0, cov*(df-2)/df), w~chi2(df)/df.
            # Scaling cov by (df-2)/df makes the t-pool's covariance equal `cov`, so only the TAIL
            # weight changes (heavier within-group dispersion), not the scale -- omega^2 stays
            # comparable while the heavier tails widen the PERMANOVA null and lower power toward the
            # real (non-Gaussian, over-dispersed) data. One knob (df); df -> inf recovers Gaussian.
            df = float(pool_df)
            Z = rng.multivariate_normal(np.zeros_like(mu), cov * (df - 2.0) / df, size=pool_size)
            w = rng.chisquare(df, size=pool_size) / df
            pts.append(mu + Z / np.sqrt(w)[:, None])
        else:
            pts.append(rng.multivariate_normal(mu, cov, size=pool_size))
        ids.extend([f"syn_{_safe_label(g)}_{i:05d}" for i in range(pool_size)])
    P = np.vstack(pts)
    return pd.DataFrame(squareform(pdist(P, metric="euclidean")), index=ids, columns=ids)


def _omega_calibrated_internal_scale_max(
    args: argparse.Namespace,
    model: Optional[Dict[str, Any]],
    observed_scale_max: float,
) -> float:
    if not model or model.get("center_mode") != "omega-calibrated":
        return float(observed_scale_max)
    shrink = float(model.get("center_shrinkage", 1.0))
    if not np.isfinite(shrink) or shrink <= 1e-8:
        shrink = 1e-8
    max_internal = float(observed_scale_max) / shrink
    cap = float(getattr(args, "omega_calibrated_max_scale", 20.0))
    return float(min(max_internal, cap))


def _ordination_scales(args: argparse.Namespace, model: Optional[Dict[str, Any]] = None) -> np.ndarray:
    dilution = np.linspace(0.0, 1.0, num=max(2, int(args.decrease_num)))
    enhance_max = _omega_calibrated_internal_scale_max(
        args,
        model,
        float(getattr(args, "ordination_enhance_max", 3.0)),
    )
    enhancement = np.linspace(1.0, enhance_max, num=max(2, int(args.increase_num)))
    return _unique_sorted_levels(np.concatenate([dilution, enhancement]))


def _ordination_scales_for_eval_n(
    args: argparse.Namespace,
    eval_n: int,
    model: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if getattr(args, "effect_grid", "fixed") != "adaptive":
        return _ordination_scales(args, model=model)

    reference_n = max(2.0, float(getattr(args, "adaptive_reference_n", 17)))
    eval_n_float = max(2.0, float(eval_n))
    enhance_max = float(getattr(args, "ordination_enhance_max", 3.0))
    shrink = min(1.0, np.sqrt(reference_n / eval_n_float))
    adaptive_observed_max = 1.0 + (enhance_max - 1.0) * shrink
    adaptive_observed_max = max(1.05, min(enhance_max, adaptive_observed_max))
    adaptive_max = _omega_calibrated_internal_scale_max(args, model, adaptive_observed_max)

    decrease_n = max(4, int(args.decrease_num))
    increase_n = max(4, int(args.increase_num))
    low_n = max(6, int(getattr(args, "adaptive_low_num", 12)))

    # Large target n shifts the informative power transition toward small omega^2.
    # Use squared spacing to put more effect points at low centroid scales, then
    # keep a short plateau past the expected transition.
    dilution = np.linspace(0.0, 1.0, num=decrease_n) ** 2
    low_cap = min(adaptive_max, 1.2)
    low_transition = low_cap * (np.linspace(0.0, 1.0, num=low_n) ** 2)
    enhance_power = 0.75 if model and model.get("center_mode") == "omega-calibrated" else 1.5
    enhancement = 1.0 + (adaptive_max - 1.0) * (np.linspace(0.0, 1.0, num=increase_n) ** enhance_power)
    return _unique_sorted_levels(np.concatenate([dilution, low_transition, enhancement]))


def _ordination_scales_by_omega_preview(
    args: argparse.Namespace,
    eval_n: int,
    model: Dict[str, Any],
    pool_size: int,
    pool_group_map: pd.Series,
    seed: int,
) -> np.ndarray:
    """Choose effect scales whose resulting omega^2 values are roughly evenly spaced.

    Scale-space grids over-sample the low-omega region when small scales collapse to
    omega^2=0 after PERMANOVA's correction. This preview pass is cheap relative to
    bootstrapping: it computes omega^2 for a denser candidate scale grid, then picks
    scales nearest to evenly spaced omega^2 targets.
    """

    reference_n = max(2.0, float(getattr(args, "adaptive_reference_n", 17)))
    eval_n_float = max(2.0, float(eval_n))
    observed_max = float(getattr(args, "ordination_enhance_max", 3.0))
    if getattr(args, "effect_grid", "fixed") != "fixed":
        shrink_n = min(1.0, np.sqrt(reference_n / eval_n_float))
        observed_max = 1.0 + (observed_max - 1.0) * shrink_n
    max_scale = _omega_calibrated_internal_scale_max(args, model, observed_max)

    candidate_n = max(40, int(getattr(args, "omega_grid_candidates", 80)))
    below_n = max(10, int(args.decrease_num) + 4)
    below = np.linspace(0.0, 1.0, below_n)
    above_linear = 1.0 + (max_scale - 1.0) * np.linspace(0.0, 1.0, candidate_n)
    if max_scale > 1.0:
        above_geom = np.geomspace(1.0, max_scale, candidate_n)
    else:
        above_geom = np.array([1.0])
    candidates = _unique_sorted_levels(np.concatenate([below, above_linear, above_geom, [0.0, 1.0, max_scale]]))

    records: List[Tuple[float, float]] = []
    for i, scale in enumerate(candidates):
        dm = _ordination_pool_dm(model, float(scale), pool_size, int(seed) + i * 9973)
        omega = float(core.compute_omega2(dm, pool_group_map))
        if np.isfinite(omega):
            records.append((float(scale), max(0.0, omega)))
    if len(records) <= 3:
        return candidates

    preview = pd.DataFrame(records, columns=["scale", "omega2"]).sort_values(["omega2", "scale"])
    desired = max(10, int(args.decrease_num) + int(args.increase_num) + int(getattr(args, "adaptive_low_num", 12)) - 4)
    forced = [0.0, 1.0, max_scale]
    selected: List[float] = []
    for scale in forced:
        nearest = preview.iloc[(preview["scale"] - float(scale)).abs().argsort()[:1]]
        if not nearest.empty:
            selected.append(float(nearest.iloc[0]["scale"]))

    positive = preview[preview["omega2"] > 1e-8]
    if positive.empty:
        return _unique_sorted_levels(selected)

    omega_min = float(positive["omega2"].min())
    omega_max = float(preview["omega2"].max())
    remaining = max(0, desired - len(_unique_sorted_levels(selected)))
    targets = np.linspace(omega_min, omega_max, remaining) if remaining else np.array([])
    for target in targets:
        nearest = preview.iloc[(preview["omega2"] - float(target)).abs().argsort()[:1]]
        if not nearest.empty:
            selected.append(float(nearest.iloc[0]["scale"]))

    return _unique_sorted_levels(selected)


def _ordination_scales_by_power_preview(
    args: argparse.Namespace,
    eval_n: int,
    model: Dict[str, Any],
    pool_size: int,
    pool_group_map: pd.Series,
    seed: int,
    workflow: str,
) -> np.ndarray:
    """Choose effect scales by previewing the power transition.

    This is the ordination analogue of the protein adaptive coarse/refine
    strategy: run a cheap preview over candidate scales, identify intervals
    that cross the target power or fall in the transition band, and select the
    final scales around those intervals plus anchors. It avoids hard-coding an
    omega^2 range for high eval_n cases where power saturates very early.
    """

    reference_n = max(2.0, float(getattr(args, "adaptive_reference_n", 17)))
    eval_n_float = max(2.0, float(eval_n))
    observed_max = float(getattr(args, "ordination_enhance_max", 3.0))
    shrink_n = min(1.0, np.sqrt(reference_n / eval_n_float))
    observed_max = 1.0 + (observed_max - 1.0) * shrink_n
    max_scale = _omega_calibrated_internal_scale_max(args, model, observed_max)

    candidate_n = max(28, int(getattr(args, "power_grid_candidates", 48)))
    below_n = max(8, int(args.decrease_num))
    below = np.linspace(0.0, 1.0, below_n)
    if max_scale > 1.0:
        above_geom = np.geomspace(1.0, max_scale, candidate_n)
        above_linear = 1.0 + (max_scale - 1.0) * np.linspace(0.0, 1.0, max(10, candidate_n // 2))
    else:
        above_geom = np.array([1.0])
        above_linear = np.array([1.0])
    candidates = _unique_sorted_levels(np.concatenate([below, above_geom, above_linear, [0.0, 1.0, max_scale]]))

    preview_boot = max(8, int(getattr(args, "power_preview_boot_number", 20)))
    preview_perms = max(9, int(getattr(args, "power_preview_permutations", 49)))
    preview_rows: List[Dict[str, float]] = []
    for i, scale in enumerate(candidates):
        dm = _ordination_pool_dm(model, float(scale), pool_size, int(seed) + i * 9973)
        metrics = summarize_distance_metrics_with_replacement(
            dm=dm,
            group_map=pool_group_map,
            boot_number=preview_boot,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            random_seed=int(seed) + i * 7919,
            n_per_group=eval_n,
            permutations=preview_perms,
            omega2_floor=getattr(args, "omega2_floor", None),
            failure_log_path=args.out / f"{workflow}_power_preview_failures.jsonl",
            failure_context={
                "workflow": workflow,
                "stage": "power_preview",
                "eval_n": eval_n,
                "scale": float(scale),
            },
        )
        preview_rows.append({
            "scale": float(scale),
            "omega2": float(metrics["true_omega2"]),
            "power": float(metrics["power"]),
        })

    preview = pd.DataFrame(preview_rows).replace([np.inf, -np.inf], np.nan).dropna()
    if len(preview) <= 3:
        return candidates
    preview = preview.sort_values("scale")

    fit_power_min = float(getattr(args, "fit_power_min", 0.15))
    fit_power_max = float(getattr(args, "fit_power_max", 0.95))
    target_power = float(args.target_power)
    selected: List[float] = []
    for anchor in [0.0, 1.0, max_scale]:
        nearest = preview.iloc[(preview["scale"] - anchor).abs().argsort()[:1]]
        if not nearest.empty:
            selected.append(float(nearest.iloc[0]["scale"]))

    records = preview.to_dict("records")
    intervals: List[Tuple[float, float, float]] = []
    for left, right in zip(records, records[1:]):
        a = float(left["scale"])
        b = float(right["scale"])
        if np.isclose(a, b):
            continue
        p1 = float(left["power"])
        p2 = float(right["power"])
        endpoint_transition = (fit_power_min <= p1 <= fit_power_max) or (fit_power_min <= p2 <= fit_power_max)
        crosses_target = (p1 - target_power) * (p2 - target_power) <= 0.0
        if endpoint_transition or crosses_target:
            score = 0.0
            score += 3.0 if crosses_target else 0.0
            score += 1.5 if endpoint_transition else 0.0
            score -= min(abs(p1 - target_power), abs(p2 - target_power))
            intervals.append((score, a, b))
    intervals.sort(key=lambda item: item[0], reverse=True)

    final_n = max(12, int(args.decrease_num) + int(args.increase_num) + int(getattr(args, "adaptive_low_num", 12)) - 4)
    per_interval = max(4, int(getattr(args, "power_refine_per_interval", 7)))
    for _score, a, b in intervals:
        if len(_unique_sorted_levels(selected)) >= final_n:
            break
        if a <= 0 < b:
            grid = np.linspace(a, b, per_interval)
        elif a > 0 and b / max(a, 1e-12) > 1.5:
            grid = np.geomspace(a, b, per_interval)
        else:
            grid = np.linspace(a, b, per_interval)
        selected.extend([float(x) for x in grid])

    if len(_unique_sorted_levels(selected)) < max(8, final_n // 2):
        return _ordination_scales_by_omega_preview(args, eval_n, model, pool_size, pool_group_map, seed)

    return _unique_sorted_levels(selected)


def _ordination_scales_by_power_uniform(
    args: argparse.Namespace,
    eval_n: int,
    model: Dict[str, Any],
    pool_size: int,
    pool_group_map: pd.Series,
    seed: int,
    workflow: str = "taxon",
) -> np.ndarray:
    """Choose effect scales so the resulting POWER values are evenly spaced over [~alpha, ~1].

    omega-uniform spacing piles points at power=1 when eval_n is large and the power-vs-omega^2
    transition is a narrow window (most omega^2 values are already saturated). This two-stage
    routine instead targets uniform POWER: (1) a coarse preview locates the transition window in
    scale-space (power ~0.1 -> ~0.97); (2) a dense preview inside that window builds a monotone
    power(scale) curve; then it INVERTS that curve at evenly spaced target power levels to pick the
    final scales. Robust to any eval_n / steepness, for both gene (Gemelli) and protein (PhyloFunc).
    """

    observed_max = float(getattr(args, "ordination_enhance_max", 3.0))
    if getattr(args, "effect_grid", "fixed") != "fixed":
        reference_n = max(2.0, float(getattr(args, "adaptive_reference_n", 17)))
        observed_max = 1.0 + (observed_max - 1.0) * min(1.0, np.sqrt(reference_n / max(2.0, float(eval_n))))
    max_scale = _omega_calibrated_internal_scale_max(args, model, observed_max)

    preview_boot = max(12, int(getattr(args, "power_preview_boot_number", 20)))
    preview_perms = max(9, int(getattr(args, "power_preview_permutations", 49)))
    fit_power_min = float(getattr(args, "fit_power_min", 0.15))
    n_targets = max(10, int(args.decrease_num) + int(args.increase_num) - 2)

    def _preview(scales: np.ndarray, tag: str) -> pd.DataFrame:
        rows: List[Tuple[float, float]] = []
        for i, sc in enumerate(scales):
            dm = _ordination_pool_dm(model, float(sc), pool_size, int(seed) + hash(tag) % 9973 + i * 7919)
            metrics = summarize_distance_metrics_with_replacement(
                dm=dm, group_map=pool_group_map, boot_number=preview_boot, alpha=args.alpha,
                n_jobs=args.n_jobs, random_seed=int(seed) + i * 6131, n_per_group=eval_n,
                permutations=preview_perms, omega2_floor=getattr(args, "omega2_floor", None),
            )
            rows.append((float(sc), float(metrics["power"])))
        df = pd.DataFrame(rows, columns=["scale", "power"]).dropna().sort_values("scale")
        df["pmono"] = np.maximum.accumulate(df["power"].values)
        return df

    # Stage 1: coarse preview to locate the transition window in scale-space. For a weak-effect
    # pilot the eval-reduced max_scale may not reach saturation; expand toward the hard cap until
    # the preview power approaches 1, so the transition window is always captured.
    n_coarse = max(24, int(getattr(args, "omega_grid_candidates", 30)) // 3 + 12)
    hard_cap = float(getattr(args, "omega_calibrated_max_scale", 20.0))
    coarse = _preview(np.linspace(0.0, max_scale, n_coarse), "coarse")
    tries = 0
    while len(coarse) > 3 and float(coarse["pmono"].max()) < 0.9 and max_scale < hard_cap and tries < 4:
        max_scale = min(hard_cap, max_scale * 2.5)
        tries += 1
        coarse = _preview(np.linspace(0.0, max_scale, n_coarse), f"coarse{tries}")
    if len(coarse) <= 3:
        return _ordination_scales(args, model)
    lo = coarse[coarse["pmono"] <= 0.12]["scale"]
    hi = coarse[coarse["pmono"] >= 0.97]["scale"]
    s_lo = max(0.0, (float(lo.max()) if len(lo) else 0.0) - max_scale * 0.03)
    s_hi = (float(hi.min()) if len(hi) else float(coarse["scale"].max())) + max_scale * 0.03

    # Stage 2: dense preview inside the window, then invert power(scale) at uniform power targets.
    dense = _preview(np.linspace(s_lo, s_hi, 40), "dense")
    if len(dense) <= 3 or float(dense["pmono"].max()) - float(dense["pmono"].min()) < 1e-3:
        return _ordination_scales(args, model)
    targets = np.linspace(max(0.05, fit_power_min - 0.05), 0.96, n_targets)
    inverted = np.interp(targets, dense["pmono"].values, dense["scale"].values)
    scales = np.concatenate([[0.0], inverted, [float(dense["scale"].max())]])
    return _unique_sorted_levels(scales)


def run_sensitivity_taxon_function(args: argparse.Namespace) -> Dict[str, Any]:
    core._require_pro_rwct_runtime()
    core.load_core_runtime()
    args.out.mkdir(parents=True, exist_ok=True)
    group_map = _read_group_map(args.group)
    long_df, aligned_group_map = _read_protein_long_table(args.table, group_map)
    full_n = int(aligned_group_map.value_counts().min())
    pilot_ns = _parse_int_list(args.pilot_ns) if isinstance(args.pilot_ns, str) else list(args.pilot_ns)
    eval_n = int(args.eval_n if args.eval_n is not None else full_n)
    if getattr(args, "effect_grid", "fixed") == "adaptive":
        increase_levels = np.linspace(0.0, args.increase_max, num=max(2, args.coarse_increase_num))
        decrease_levels = np.linspace(0.0, args.decrease_min, num=max(2, args.coarse_decrease_num))
    else:
        increase_levels = np.linspace(0.0, args.increase_max, num=max(2, args.increase_num))
        decrease_levels = np.linspace(0.0, args.decrease_min, num=max(2, args.decrease_num))
    all_rows: List[Dict[str, Any]] = []
    diagnostics_frames: List[pd.DataFrame] = []
    reference_metrics: Optional[Dict[str, float]] = None
    if args.calibrate_to_real:
        real_dm = core.compute_phylofunc_distance_matrix(long_df, str(args.tree))
        reference_metrics = _compute_real_metrics_from_dm(
            dm=real_dm,
            group_map=aligned_group_map,
            args=args,
            eval_n=eval_n,
            workflow="taxon-function",
        )
        print(
            f"  [taxon-function] real reference at eval_n={eval_n}: "
            f"omega2={reference_metrics['true_omega2']:.4g}, power={reference_metrics['power']:.3g}"
        )
    if getattr(args, "engine", "ordination") == "ordination":
        for pilot_n in pilot_ns:
            pilot_group_map = _subsample_group_map(aligned_group_map, pilot_n, args.random_seed + pilot_n)
            pilot_df = long_df[ID_COLS + list(pilot_group_map.index)].copy()
            pool_size = int(args.pool_size_per_group) if args.pool_size_per_group is not None else max(500, 50 * eval_n)
            real_pilot_dm = core.compute_phylofunc_distance_matrix(pilot_df, str(args.tree))
            real_pilot_dm_df = _as_distance_frame(real_pilot_dm)
            model = _build_ordination_model(
                real_pilot_dm_df,
                pilot_group_map,
                center_mode=getattr(args, "center_mode", "debiased"),
            )
            pool_gm = _ordination_pool_group_map(model, pool_size)
            print(
                f"  [taxon-function] pilot_n={pilot_n} ordination engine: "
                f"{len(model['groups'])} groups, {model['k']} PCoA axes, pool={pool_size}/group"
            )
            if getattr(args, "effect_grid", "fixed") == "omega-uniform":
                scales = _ordination_scales_by_omega_preview(
                    args,
                    eval_n,
                    model,
                    pool_size,
                    pool_gm,
                    seed=args.random_seed + pilot_n * 10000,
                )
            elif getattr(args, "effect_grid", "fixed") == "power-adaptive":
                scales = _ordination_scales_by_power_preview(
                    args,
                    eval_n,
                    model,
                    pool_size,
                    pool_gm,
                    seed=args.random_seed + pilot_n * 10000,
                    workflow="taxon-function",
                )
            elif getattr(args, "effect_grid", "fixed") == "power-uniform":
                scales = _ordination_scales_by_power_uniform(
                    args,
                    eval_n,
                    model,
                    pool_size,
                    pool_gm,
                    seed=args.random_seed + pilot_n * 10000,
                    workflow="taxon-function",
                )
            else:
                scales = _ordination_scales_for_eval_n(args, eval_n, model=model)
            scenarios = []
            for sc in scales:
                dm = _ordination_pool_dm(model, float(sc), pool_size, args.random_seed + pilot_n * 1000 + int(sc * 100))
                scenarios.append({"mode": "enhancement" if sc >= 1.0 else "dilution", "effect_level": float(sc), "dm": dm})
            all_rows.extend(
                _run_pilot_sensitivity_rows(
                    args=args,
                    workflow="taxon-function",
                    scenarios=scenarios,
                    synthetic_group_map=pool_gm,
                    pilot_n=pilot_n,
                    eval_n=eval_n,
                    include_gamma=False,
                )
            )
            diagnostics_frames.append(pd.DataFrame([{
                "group_name": str(g), "pilot_n": pilot_n, "eval_n": eval_n,
                "engine": "ordination", "pco_axes": model["k"], "pool_size": pool_size,
                "center_mode": model.get("center_mode"), "center_shrinkage": model.get("center_shrinkage"),
                "pilot_target_omega2": model.get("pilot_target_omega2"),
                "ledoit_wolf_cov": True, "effect_grid": getattr(args, "effect_grid", "fixed"),
                "ordination_scale_count": len(scales), "ordination_scale_min": float(np.min(scales)),
                "ordination_scale_max": float(np.max(scales)),
            } for g in model["groups"]]))
        return _write_pilot_sensitivity_outputs(
            args=args, workflow="taxon-function", workflow_label="Taxon-function",
            rows=all_rows, diagnostics_frames=diagnostics_frames,
        )

    for pilot_n in pilot_ns:
        pilot_group_map = _subsample_group_map(aligned_group_map, pilot_n, args.random_seed + pilot_n)
        pilot_df = long_df[ID_COLS + list(pilot_group_map.index)].copy()
        pool_size = _resolve_pool_size(pilot_n, eval_n, args.pool_size_per_group)
        if getattr(args, "calibrate_to_pilot", False):
            synthetic_df, synthetic_group_map, diagnostics_df, _target, _s = _calibrate_protein_pool_to_pilot(
                args=args,
                pilot_df=pilot_df,
                pilot_group_map=pilot_group_map,
                pool_size=pool_size,
                pilot_n=pilot_n,
            )
        else:
            synthetic_df, synthetic_group_map, diagnostics_df = _select_taxon_function_pool_for_sensitivity(
                args=args,
                pilot_df=pilot_df,
                pilot_group_map=pilot_group_map,
                pool_size=pool_size,
                pilot_n=pilot_n,
                eval_n=eval_n,
                reference_metrics=reference_metrics,
            )
        diagnostics_df["pilot_n"] = pilot_n
        diagnostics_df["eval_n"] = eval_n
        diagnostics_frames.append(diagnostics_df)
        increase_scenarios, decrease_scenarios = precompute_rwct_distance_scenarios_parallel(
            df_scenario=synthetic_df,
            tree_path_str=str(args.tree),
            group_map=synthetic_group_map,
            enhancement_gammas=increase_levels,
            dilution_gammas=decrease_levels,
            ridge=args.ridge,
            normalize_weights=not args.no_normalize_weights,
            n_jobs=args.n_jobs,
        )
        scenarios: List[Dict[str, Any]] = []
        for mode, scenario_list in [("enhancement", increase_scenarios), ("dilution", decrease_scenarios)]:
            for scenario in scenario_list:
                scenarios.append(_scenario_with_rwct_fields(mode, scenario))
        coarse_rows = _run_pilot_sensitivity_rows(
            args=args,
            workflow="taxon-function",
            scenarios=scenarios,
            synthetic_group_map=synthetic_group_map,
            pilot_n=pilot_n,
            eval_n=eval_n,
            include_gamma=True,
        )
        all_rows.extend(coarse_rows)

        if getattr(args, "effect_grid", "fixed") == "adaptive":
            refine_increase_levels = _select_refine_levels_from_rows(
                coarse_rows,
                args=args,
                mode="enhancement",
                existing_levels=increase_levels,
            )
            refine_decrease_levels = _select_refine_levels_from_rows(
                coarse_rows,
                args=args,
                mode="dilution",
                existing_levels=decrease_levels,
            )
            if len(refine_increase_levels) or len(refine_decrease_levels):
                print(
                    f"  [taxon-function] pilot_n={pilot_n} adaptive refine: "
                    f"increase={refine_increase_levels.tolist()} "
                    f"decrease={refine_decrease_levels.tolist()}"
                )
                refine_increase_scenarios, refine_decrease_scenarios = precompute_rwct_distance_scenarios_parallel(
                    df_scenario=synthetic_df,
                    tree_path_str=str(args.tree),
                    group_map=synthetic_group_map,
                    enhancement_gammas=refine_increase_levels,
                    dilution_gammas=refine_decrease_levels,
                    ridge=args.ridge,
                    normalize_weights=not args.no_normalize_weights,
                    n_jobs=args.n_jobs,
                )
                refine_scenarios: List[Dict[str, Any]] = []
                for mode, scenario_list in [
                    ("enhancement", refine_increase_scenarios),
                    ("dilution", refine_decrease_scenarios),
                ]:
                    for scenario in scenario_list:
                        refine_scenarios.append(_scenario_with_rwct_fields(mode, scenario))
                all_rows.extend(
                    _run_pilot_sensitivity_rows(
                        args=args,
                        workflow="taxon-function",
                        scenarios=refine_scenarios,
                        synthetic_group_map=synthetic_group_map,
                        pilot_n=pilot_n,
                        eval_n=eval_n,
                        include_gamma=True,
                        scenario_index_offset=len(scenarios),
                    )
                )
    return _write_pilot_sensitivity_outputs(
        args=args,
        workflow="taxon-function",
        workflow_label="Taxon-function",
        rows=all_rows,
        diagnostics_frames=diagnostics_frames,
    )


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-power", type=float, required=True)
    parser.add_argument("--target-omega2", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--boot-number", type=int, default=core.DEFAULT_BOOT_NUMBER)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--pool-size-per-group", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--out", type=Path, required=True)


def _add_compare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--small-n", type=int, default=None)
    parser.add_argument("--large-n", type=int, default=None)
    parser.add_argument(
        "--resampling-mode",
        choices=["both", "with_replacement", "without_replacement"],
        default="both",
    )


def _add_sensitivity_arguments(parser: argparse.ArgumentParser, default_pilot_ns: str, *, include_detection: bool) -> None:
    parser.add_argument("--pilot-ns", type=str, default=default_pilot_ns)
    parser.add_argument("--eval-n", type=int, default=None)
    if include_detection:
        parser.add_argument(
            "--engine",
            choices=["ordination", "table"],
            default="ordination",
            help="Sample-size extrapolation engine. 'ordination' (default): embed the real pilot "
            "distance matrix via PCoA, de-bias centroids, Ledoit-Wolf covariance, draw a large MVN "
            "pool, with-replacement bootstrap -- preserves the real distance geometry, no table "
            "regeneration. 'table': legacy template-mask feature-table regeneration.",
        )
        parser.add_argument(
            "--ordination-enhance-max",
            type=float,
            default=3.0,
            help="Max centroid-separation scale for the ordination effect sweep (>1 = enhancement).",
        )
        parser.add_argument(
            "--adaptive-reference-n",
            type=int,
            default=17,
            help="Reference target n for ordination adaptive effect range; larger eval_n shrink the scale range.",
        )
        parser.add_argument(
            "--adaptive-low-num",
            type=int,
            default=12,
            help="Number of extra low-scale points for ordination adaptive effect range.",
        )
    parser.add_argument("--effect-grid", choices=["fixed", "adaptive", "omega-uniform", "power-adaptive", "power-uniform"], default="fixed")
    parser.add_argument(
        "--center-mode",
        choices=["observed", "debiased", "empirical-bayes", "omega-calibrated"],
        default="debiased",
        help="Centroid target for ordination/MVN simulations. 'observed' preserves the pilot's "
        "raw centroid separation at scale=1; 'debiased' uses the positive-part small-sample "
        "correction; 'empirical-bayes' uses a continuous signal/(signal+noise) shrinkage; "
        "'omega-calibrated' chooses the scale=1 centroid separation to match the pilot's own "
        "PERMANOVA omega^2.",
    )
    parser.add_argument(
        "--omega-calibrated-max-scale",
        type=float,
        default=20.0,
        help="Safety cap for the internal centroid scale when --center-mode omega-calibrated "
        "expands the effect range after shrinking scale=1 to the pilot omega^2.",
    )
    parser.add_argument(
        "--omega-grid-candidates",
        type=int,
        default=80,
        help="Candidate scale count for --effect-grid omega-uniform preview selection.",
    )
    parser.add_argument(
        "--power-grid-candidates",
        type=int,
        default=48,
        help="Candidate scale count for --effect-grid power-adaptive preview.",
    )
    parser.add_argument(
        "--power-preview-boot-number",
        type=int,
        default=20,
        help="Bootstrap count for --effect-grid power-adaptive preview.",
    )
    parser.add_argument(
        "--power-preview-permutations",
        type=int,
        default=49,
        help="PERMANOVA permutations for --effect-grid power-adaptive preview.",
    )
    parser.add_argument(
        "--power-refine-per-interval",
        type=int,
        default=7,
        help="Final scale points placed in each preview-detected transition interval.",
    )
    parser.add_argument("--coarse-increase-num", type=int, default=6)
    parser.add_argument("--coarse-decrease-num", type=int, default=6)
    parser.add_argument("--refine-num", type=int, default=6)
    parser.add_argument("--fit-power-min", type=float, default=0.15)
    parser.add_argument("--fit-power-max", type=float, default=0.95)
    parser.add_argument("--fit-filter", choices=["none", "realistic", "transition"], default="none")
    parser.add_argument("--max-lib-drift", type=float, default=1.0)
    parser.add_argument("--max-log-shift", type=float, default=4.0)
    parser.add_argument("--calibrate-to-real", action="store_true")
    parser.add_argument(
        "--calibrate-to-pilot",
        action="store_true",
        help="Calibrate the synthetic pool's effect size to the pilot's own bias-corrected omega^2 "
        "(removes the generator's small-pilot inflation). Self-contained; needs no larger reference.",
    )
    parser.add_argument(
        "--safeguard-power",
        action="store_true",
        help="With --calibrate-to-pilot, target the safeguard lower bound of the pilot omega^2 "
        "(conservative; protects against an over-separated pilot draw).",
    )
    parser.add_argument("--calibration-grid", type=int, default=6, help="Grid points for --calibrate-to-pilot separation search.")
    parser.add_argument("--calibration-boot-number", type=int, default=50)
    parser.add_argument("--calibration-between-scales", type=str, default="0.35,0.55,0.75,1.0")
    parser.add_argument("--calibration-residual-scales", type=str, default="1.0,1.75")
    parser.add_argument("--calibration-noise-multipliers", type=str, default="0.10")
    if include_detection:
        parser.add_argument("--calibration-detection-slopes", type=str, default="0.0,1.5")


def _add_generator_arguments(parser: argparse.ArgumentParser, *, include_detection: bool) -> None:
    parser.add_argument("--between-scale", type=float, default=1.0)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--noise-multiplier", type=float, default=0.10)
    if include_detection:
        parser.add_argument("--protein-generator", choices=["template-mask", "bernoulli"], default="template-mask")
        parser.add_argument("--detection-slope", type=float, default=1.0)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="semisynthetic_power.py",
        description="Experimental semi-synthetic Phylopower runner that estimates prospective power by with-replacement bootstrapping from generated large pools.",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    taxon = subparsers.add_parser("taxon", help="Semi-synthetic power analysis for taxonomic abundance data.")
    _add_shared_arguments(taxon)
    _add_generator_arguments(taxon, include_detection=False)
    taxon.add_argument("--table", type=Path, default=core.DATAGENE_DIR / "table.csv")
    taxon.add_argument("--tree", type=Path, default=core.DATAGENE_DIR / "rooted-tree.nwk")
    taxon.add_argument("--taxonomy", type=Path, default=core.DATAGENE_DIR / "taxonomy.csv")
    taxon.add_argument("--group", type=Path, default=core.DATAGENE_DIR / "group.csv")
    taxon.add_argument("--qiime-env", type=str, default="qiime2-metagenome-2024.10")
    taxon.add_argument("--increase-max", type=float, default=core.DEFAULT_TAXON_INCREASE_MAX)
    taxon.add_argument("--increase-num", type=int, default=30)
    taxon.add_argument("--decrease-min", type=float, default=core.DEFAULT_TAXON_DECREASE_MIN)
    taxon.add_argument("--decrease-num", type=int, default=75)
    taxon.add_argument("--omega2-floor", type=float, default=0.0)
    taxon.set_defaults(func=run_taxon)

    taxon_function = subparsers.add_parser(
        "taxon-function",
        help="Semi-synthetic power analysis for Taxon-Function abundance data.",
    )
    _add_shared_arguments(taxon_function)
    _add_generator_arguments(taxon_function, include_detection=True)
    taxon_function.add_argument("--table", type=Path, default=core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv")
    taxon_function.add_argument("--tree", type=Path, default=core.DATAPRO_DIR / "rooted-tree.nwk")
    taxon_function.add_argument("--group", type=Path, default=core.DATAPRO_DIR / "group.csv")
    taxon_function.add_argument("--increase-max", type=float, default=core.DEFAULT_RWCT_INCREASE_MAX)
    taxon_function.add_argument("--increase-num", type=int, default=core.DEFAULT_RWCT_LEVELS)
    taxon_function.add_argument("--decrease-min", type=float, default=core.DEFAULT_RWCT_DECREASE_MIN)
    taxon_function.add_argument("--decrease-num", type=int, default=core.DEFAULT_RWCT_LEVELS)
    taxon_function.add_argument("--ridge", type=float, default=1.0)
    taxon_function.add_argument("--no-normalize-weights", action="store_true")
    taxon_function.set_defaults(func=run_taxon_function)

    compare_taxon = subparsers.add_parser(
        "compare-taxon",
        help="Plot with- vs without-replacement omega^2-power curves for taxonomic abundance data.",
    )
    _add_shared_arguments(compare_taxon)
    _add_generator_arguments(compare_taxon, include_detection=False)
    _add_compare_arguments(compare_taxon)
    compare_taxon.add_argument("--table", type=Path, default=core.DATAGENE_DIR / "table.csv")
    compare_taxon.add_argument("--tree", type=Path, default=core.DATAGENE_DIR / "rooted-tree.nwk")
    compare_taxon.add_argument("--taxonomy", type=Path, default=core.DATAGENE_DIR / "taxonomy.csv")
    compare_taxon.add_argument("--group", type=Path, default=core.DATAGENE_DIR / "group.csv")
    compare_taxon.add_argument("--qiime-env", type=str, default="qiime2-metagenome-2024.10")
    compare_taxon.add_argument("--increase-max", type=float, default=core.DEFAULT_TAXON_INCREASE_MAX)
    compare_taxon.add_argument("--increase-num", type=int, default=30)
    compare_taxon.add_argument("--decrease-min", type=float, default=core.DEFAULT_TAXON_DECREASE_MIN)
    compare_taxon.add_argument("--decrease-num", type=int, default=75)
    compare_taxon.add_argument("--omega2-floor", type=float, default=0.0)
    compare_taxon.set_defaults(func=run_compare_taxon)

    compare_taxon_function = subparsers.add_parser(
        "compare-taxon-function",
        help="Plot with- vs without-replacement omega^2-power curves for Taxon-Function data.",
    )
    _add_shared_arguments(compare_taxon_function)
    _add_generator_arguments(compare_taxon_function, include_detection=True)
    _add_compare_arguments(compare_taxon_function)
    compare_taxon_function.add_argument("--table", type=Path, default=core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv")
    compare_taxon_function.add_argument("--tree", type=Path, default=core.DATAPRO_DIR / "rooted-tree.nwk")
    compare_taxon_function.add_argument("--group", type=Path, default=core.DATAPRO_DIR / "group.csv")
    compare_taxon_function.add_argument("--increase-max", type=float, default=core.DEFAULT_RWCT_INCREASE_MAX)
    compare_taxon_function.add_argument("--increase-num", type=int, default=core.DEFAULT_RWCT_LEVELS)
    compare_taxon_function.add_argument("--decrease-min", type=float, default=core.DEFAULT_RWCT_DECREASE_MIN)
    compare_taxon_function.add_argument("--decrease-num", type=int, default=core.DEFAULT_RWCT_LEVELS)
    compare_taxon_function.add_argument("--ridge", type=float, default=1.0)
    compare_taxon_function.add_argument("--no-normalize-weights", action="store_true")
    compare_taxon_function.set_defaults(func=run_compare_taxon_function)

    sensitivity_taxon = subparsers.add_parser(
        "sensitivity-taxon",
        help="Pilot-size sensitivity curves for taxonomic abundance data using with-replacement bootstrap.",
    )
    _add_shared_arguments(sensitivity_taxon)
    _add_generator_arguments(sensitivity_taxon, include_detection=False)
    _add_sensitivity_arguments(sensitivity_taxon, default_pilot_ns="5,7,10", include_detection=False)
    sensitivity_taxon.add_argument(
        "--engine",
        choices=["ordination", "gemelli-loading", "table"],
        default="ordination",
        help="Sample-size extrapolation engine. 'ordination' (default): compute the pilot Gemelli/RPCA "
        "distance matrix, embed it with PCoA, estimate Ledoit-Wolf covariance, draw a large MVN pool, "
        "and bootstrap with replacement. 'gemelli-loading': build the MVN pool directly in Gemelli's "
        "MatrixCompletion U-loading space, matching Gemelli's own distance definition. 'table': legacy "
        "CLR/prevalence/library-size table regeneration.",
    )
    sensitivity_taxon.add_argument(
        "--ordination-enhance-max",
        type=float,
        default=3.0,
        help="Max centroid-separation scale for the ordination effect sweep (>1 = enhancement).",
    )
    sensitivity_taxon.add_argument(
        "--adaptive-reference-n",
        type=int,
        default=17,
        help="Reference target n for ordination adaptive effect range; larger eval_n shrink the scale range.",
    )
    sensitivity_taxon.add_argument(
        "--adaptive-low-num",
        type=int,
        default=12,
        help="Number of extra low-scale points for ordination adaptive effect range.",
    )
    sensitivity_taxon.add_argument("--table", type=Path, default=core.DATAGENE_DIR / "table.csv")
    sensitivity_taxon.add_argument("--tree", type=Path, default=core.DATAGENE_DIR / "rooted-tree.nwk")
    sensitivity_taxon.add_argument("--taxonomy", type=Path, default=core.DATAGENE_DIR / "taxonomy.csv")
    sensitivity_taxon.add_argument("--group", type=Path, default=core.DATAGENE_DIR / "group.csv")
    sensitivity_taxon.add_argument("--qiime-env", type=str, default="qiime2-metagenome-2024.10")
    sensitivity_taxon.add_argument("--increase-max", type=float, default=core.DEFAULT_TAXON_INCREASE_MAX)
    sensitivity_taxon.add_argument("--increase-num", type=int, default=30)
    sensitivity_taxon.add_argument("--decrease-min", type=float, default=core.DEFAULT_TAXON_DECREASE_MIN)
    sensitivity_taxon.add_argument("--decrease-num", type=int, default=75)
    sensitivity_taxon.add_argument("--omega2-floor", type=float, default=0.0)
    sensitivity_taxon.set_defaults(func=run_sensitivity_taxon)

    sensitivity_taxon_function = subparsers.add_parser(
        "sensitivity-taxon-function",
        help="Pilot-size sensitivity curves for Taxon-Function data using with-replacement bootstrap.",
    )
    _add_shared_arguments(sensitivity_taxon_function)
    _add_generator_arguments(sensitivity_taxon_function, include_detection=True)
    _add_sensitivity_arguments(sensitivity_taxon_function, default_pilot_ns="5,10,17", include_detection=True)
    sensitivity_taxon_function.add_argument("--table", type=Path, default=core.DATAPRO_DIR / "protein_taxon_function_cleaned.csv")
    sensitivity_taxon_function.add_argument("--tree", type=Path, default=core.DATAPRO_DIR / "rooted-tree.nwk")
    sensitivity_taxon_function.add_argument("--group", type=Path, default=core.DATAPRO_DIR / "group.csv")
    sensitivity_taxon_function.add_argument("--increase-max", type=float, default=core.DEFAULT_RWCT_INCREASE_MAX)
    sensitivity_taxon_function.add_argument("--increase-num", type=int, default=core.DEFAULT_RWCT_LEVELS)
    sensitivity_taxon_function.add_argument("--decrease-min", type=float, default=core.DEFAULT_RWCT_DECREASE_MIN)
    sensitivity_taxon_function.add_argument("--decrease-num", type=int, default=core.DEFAULT_RWCT_LEVELS)
    sensitivity_taxon_function.add_argument("--ridge", type=float, default=1.0)
    sensitivity_taxon_function.add_argument("--no-normalize-weights", action="store_true")
    sensitivity_taxon_function.set_defaults(func=run_sensitivity_taxon_function)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    _maybe_reexec_taxon_in_qiime_env(args)
    if args.min_n < 2:
        raise ValueError("--min-n must be at least 2.")
    if args.boot_number <= 0:
        raise ValueError("--boot-number must be positive.")
    result = args.func(args)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
