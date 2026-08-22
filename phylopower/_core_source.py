"""Maintainable source embedded into the generated single-file runner."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import pickle
import platform
import shutil
import tempfile
import traceback
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from ._data import DATAGENE_DIR, DATAPRO_DIR

__all__ = [
    "compute_taxon",
    "compute_taxon_function",
    "create_argument_parser",
    "main",
]

TAXON_AUTO_CENTROID_SCALE_RANGE = (0.1, 4.0)
RWCT_AUTO_GAMMA_RANGE = (0.5, 6.0)
RWCT_AUTO_MAX_LOG_SHIFT = 2.0
SIGMOID_DECISION_ANCHOR_POWER = 0.05
DEFAULT_BOOT_NUMBER = 200
DEFAULT_TAXON_INCREASE_MAX = math.log(TAXON_AUTO_CENTROID_SCALE_RANGE[1])
DEFAULT_TAXON_DECREASE_MIN = math.log(TAXON_AUTO_CENTROID_SCALE_RANGE[0])
DEFAULT_RWCT_INCREASE_MAX = 6.0
DEFAULT_RWCT_DECREASE_MIN = -6.0
DEFAULT_RWCT_LEVELS = 11
MINIMUM_N_STABILITY_WINDOW = 2


def create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phylopower",
        description="Unified minimum balanced sample size estimator for taxon and taxon-function workflows."
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    gene_parser = subparsers.add_parser(
        "taxon",
        help="Estimate minimum sample size for taxonomic abundance data using the bidirectional centroid effect.",
    )
    add_shared_arguments(gene_parser)
    gene_parser.add_argument("--table", type=Path, default=DATAGENE_DIR / "table.csv")
    gene_parser.add_argument("--tree", type=Path, default=DATAGENE_DIR / "rooted-tree.nwk")
    gene_parser.add_argument("--taxonomy", type=Path, default=DATAGENE_DIR / "taxonomy.csv")
    gene_parser.add_argument("--group", type=Path, default=DATAGENE_DIR / "group.csv")
    gene_parser.add_argument("--qiime-env", type=str, default="qiime2-metagenome-2024.10")
    gene_parser.add_argument("--out", type=Path, default=Path("gene_min_sample_size_output"))
    gene_parser.add_argument(
        "--increase-max",
        type=float,
        default=DEFAULT_TAXON_INCREASE_MAX,
        help="Upper bound of the enhancement range in log centroid-scale units. Default: log(4.0).",
    )
    gene_parser.add_argument("--increase-num", type=int, default=30)
    gene_parser.add_argument(
        "--decrease-min",
        type=float,
        default=DEFAULT_TAXON_DECREASE_MIN,
        help="Lower bound of the dilution range in log centroid-scale units. Default: log(0.1).",
    )
    gene_parser.add_argument("--decrease-num", type=int, default=75)
    gene_parser.add_argument("--omega2-floor", type=float, default=0.0)
    gene_parser.set_defaults(workflow="taxon", pipeline=run_gene_workflow)

    pro_parser = subparsers.add_parser(
        "taxon-function",
        help="Estimate minimum sample size for Taxon-Function data using the RWCT effect operator.",
    )
    add_shared_arguments(pro_parser)
    pro_parser.add_argument("--table", dest="table", type=Path, default=DATAPRO_DIR / "protein_taxon_function_cleaned.csv")
    pro_parser.add_argument("--tree", type=Path, default=DATAPRO_DIR / "rooted-tree.nwk")
    pro_parser.add_argument("--group", type=Path, default=DATAPRO_DIR / "group.csv")
    pro_parser.add_argument("--out", type=Path, default=Path("pro_min_sample_size_output"))
    pro_parser.add_argument("--permutations", type=int, default=999)
    pro_parser.add_argument("--ridge", type=float, default=1.0)
    pro_parser.add_argument(
        "--increase-max",
        type=float,
        default=DEFAULT_RWCT_INCREASE_MAX,
        help="Upper bound of the enhancement range in RWCT gamma units. Default: 6.0.",
    )
    pro_parser.add_argument("--increase-num", type=int, default=DEFAULT_RWCT_LEVELS)
    pro_parser.add_argument(
        "--decrease-min",
        type=float,
        default=DEFAULT_RWCT_DECREASE_MIN,
        help="Lower bound of the dilution range in RWCT gamma units. Default: -6.0.",
    )
    pro_parser.add_argument("--decrease-num", type=int, default=DEFAULT_RWCT_LEVELS)
    pro_parser.add_argument(
        "--rwct-max-log-shift",
        type=float,
        default=RWCT_AUTO_MAX_LOG_SHIFT,
        help="When RWCT bounds are auto-resolved, cap the largest induced per-feature log shift at this value.",
    )
    pro_parser.add_argument("--no-normalize-weights", action="store_true")
    pro_parser.set_defaults(workflow="taxon-function", pipeline=run_protein_rwct_workflow)

    return parser


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target-power", type=float, required=True)
    parser.add_argument("--target-omega2", type=float, required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--boot-number", type=int, default=DEFAULT_BOOT_NUMBER)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-n", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=None)
    parser.add_argument("--tree-noise", nargs="+", default=["0,0"])
    parser.add_argument("--nni-support-threshold", type=float, default=None)
    parser.add_argument("--force-recompute", action="store_true")


def parse_tree_noise_arg(values: List[str]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for value in values:
        parts = value.split(",")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"--tree-noise level must be 'sigma,nni_prob' (got {value!r})"
            )
        out.append((float(parts[0]), float(parts[1])))
    return out


def make_tree_rng(random_seed: int, tree_label: str):
    # ``hash()`` on str is salted per process (PYTHONHASHSEED), which made tree
    # perturbation non-reproducible across runs.  Use SHA-256 instead so the
    # same (random_seed, tree_label) always yields the same perturbed tree.
    label_hash = int.from_bytes(hashlib.sha256(tree_label.encode("utf-8")).digest()[:8], "little")
    return np.random.default_rng(random_seed + label_hash % 100_000)


def load_core_runtime() -> None:
    if all(name in globals() for name in ("np", "pd", "Parallel", "delayed", "curve_fit")):
        return

    global np, pd, Parallel, delayed, curve_fit

    import numpy as np  # type: ignore[no-redef]
    import pandas as pd  # type: ignore[no-redef]
    from joblib import Parallel, delayed  # type: ignore[no-redef]
    from scipy.optimize import curve_fit  # type: ignore[no-redef]


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_bootstrap_seeds(random_seed: int, count: int, *stream_keys: int) -> "np.ndarray":
    load_core_runtime()
    if count <= 0:
        return np.asarray([], dtype=np.uint64)
    entropy = [int(random_seed) & 0xFFFFFFFF]
    entropy.extend(int(key) & 0xFFFFFFFF for key in stream_keys)
    seed_sequence = np.random.SeedSequence(entropy)
    child_sequences = seed_sequence.spawn(count)
    return np.asarray(
        [child.generate_state(1, dtype=np.uint64)[0] for child in child_sequences],
        dtype=np.uint64,
    )


#: Stream key separating PERMANOVA permutation seeds from bootstrap draws.
PERMANOVA_SEED_STREAM_KEY = 917


def make_permanova_seed(boot_seed: int, *stream_keys: int) -> int:
    """Derive a deterministic PERMANOVA permutation seed from a bootstrap seed.

    Uses the same ``SeedSequence`` scheme as :func:`make_bootstrap_seeds` so
    each bootstrap replicate gets an independent but fully reproducible
    permutation stream.
    """
    return int(make_bootstrap_seeds(int(boot_seed), 1, PERMANOVA_SEED_STREAM_KEY, *stream_keys)[0])


def make_bootstrap_seed_matrix(random_seed: int, rows: int, cols: int, *stream_keys: int) -> "np.ndarray":
    load_core_runtime()
    if rows <= 0:
        return np.empty((0, max(cols, 0)), dtype=np.uint64)
    return np.vstack([
        make_bootstrap_seeds(random_seed, cols, *stream_keys, row_idx)
        for row_idx in range(rows)
    ])


def log_permanova_failure(
    log_path: Optional[Path],
    context: Optional[Dict[str, Any]],
    exc: BaseException,
) -> None:
    if log_path is None:
        return
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "context": context or {},
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


def resolve_taxon_modulation_bounds(
    increase_max: Optional[float],
    decrease_min: Optional[float],
) -> tuple[float, float]:
    auto_decrease, auto_increase = (
        math.log(TAXON_AUTO_CENTROID_SCALE_RANGE[0]),
        math.log(TAXON_AUTO_CENTROID_SCALE_RANGE[1]),
    )
    resolved_increase = auto_increase if increase_max is None else float(increase_max)
    resolved_decrease = auto_decrease if decrease_min is None else float(decrease_min)
    if resolved_increase < 0:
        raise ValueError("--increase-max must be >= 0 for taxon modulation.")
    if resolved_decrease > 0:
        raise ValueError("--decrease-min must be <= 0 for taxon modulation.")
    return resolved_increase, resolved_decrease


def resolve_rwct_modulation_bounds(
    df: pd.DataFrame,
    group_map: pd.Series,
    ridge: float,
    normalize_weights: bool,
    increase_max: Optional[float],
    decrease_min: Optional[float],
    max_abs_log_shift: float,
) -> tuple[float, float, Dict[str, float]]:
    sample_cols = [c for c in df.columns if c not in ID_COLS]
    if max_abs_log_shift <= 0:
        raise ValueError("--rwct-max-log-shift must be > 0.")

    groups: Dict[str, List[str]] = {}
    for sample_id, group_name in group_map.items():
        if sample_id in sample_cols:
            groups.setdefault(str(group_name), []).append(sample_id)
    valid_groups = sorted(group_name for group_name in groups if len(groups[group_name]) >= 2)
    if len(valid_groups) < 2:
        fallback = 1.0
        return (
            fallback if increase_max is None else float(increase_max),
            -fallback if decrease_min is None else float(decrease_min),
            {
                "auto_gamma_cap": fallback,
                "auto_direction_max": 0.0,
                "rwct_max_log_shift": float(max_abs_log_shift),
            },
        )

    x = df[sample_cols].astype(float).to_numpy(dtype=float)
    presence = x > 0
    n_rows = x.shape[0]
    row_min_pos = np.zeros(n_rows, dtype=float)
    for i in range(n_rows):
        row_pos = x[i][x[i] > 0]
        row_min_pos[i] = (row_pos.min() / 100.0) if len(row_pos) else 1e-9
    log_x = np.log(np.where(presence, x, row_min_pos[:, None]))

    col_pos = {
        group_name: np.asarray([sample_cols.index(s) for s in groups[group_name]], dtype=int)
        for group_name in valid_groups
    }
    means = np.zeros((len(valid_groups), n_rows), dtype=float)
    for k, group_name in enumerate(valid_groups):
        means[k] = log_x[:, col_pos[group_name]].mean(axis=1)
    grand_mean = means.mean(axis=0)
    centered_means = means - grand_mean[None, :]

    pooled_ss = np.zeros(n_rows, dtype=float)
    df_within = 0
    for group_name in valid_groups:
        cols = col_pos[group_name]
        if len(cols) >= 2:
            pooled_ss += (len(cols) - 1) * log_x[:, cols].var(axis=1, ddof=1)
            df_within += len(cols) - 1
    sigma2 = pooled_ss / max(df_within, 1)
    q = 1.0 / (sigma2 + ridge)
    if normalize_weights:
        finite_q = q[np.isfinite(q) & (q > 0)]
        if len(finite_q):
            q = q / np.median(finite_q)
    directions = centered_means * q[None, :]
    directions = directions - directions.mean(axis=0, keepdims=True)
    direction_max = float(np.nanmax(np.abs(directions))) if directions.size else 0.0

    if not np.isfinite(direction_max) or direction_max <= 1e-12:
        auto_gamma_cap = 1.0
    else:
        auto_gamma_cap = max_abs_log_shift / direction_max
        auto_gamma_cap = min(max(auto_gamma_cap, RWCT_AUTO_GAMMA_RANGE[0]), RWCT_AUTO_GAMMA_RANGE[1])

    resolved_increase = auto_gamma_cap if increase_max is None else float(increase_max)
    resolved_decrease = -auto_gamma_cap if decrease_min is None else float(decrease_min)
    if resolved_increase < 0:
        raise ValueError("--increase-max must be >= 0 for taxon-function modulation.")
    if resolved_decrease > 0:
        raise ValueError("--decrease-min must be <= 0 for taxon-function modulation.")
    return resolved_increase, resolved_decrease, {
        "auto_gamma_cap": float(auto_gamma_cap),
        "auto_direction_max": direction_max,
        "rwct_max_log_shift": float(max_abs_log_shift),
    }


def _missing_modules(module_names: Sequence[str]) -> list[str]:
    missing = []
    for module_name in module_names:
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    return missing


def _require_base_runtime() -> None:
    base_missing = _missing_modules(["numpy", "pandas", "joblib", "scipy"])
    if base_missing:
        raise RuntimeError(
            "Missing required Python packages: "
            + ", ".join(base_missing)
            + ". Install them in the current environment before running this script."
        )

def _require_pro_rwct_runtime() -> None:
    _require_base_runtime()
    pro_missing = _missing_modules(["phylofunc", "skbio"])
    if pro_missing:
        raise RuntimeError(
            "Taxon-Function mode only needs the proteomics stack, but these packages are missing: "
            + ", ".join(pro_missing)
            + ". You do not need qiime2 for this mode."
        )


def _require_gene_runtime() -> None:
    _require_base_runtime()
    gene_missing = _missing_modules(["biom", "qiime2", "skbio"])
    if gene_missing:
        raise RuntimeError(
            "Taxon mode requires additional packages: "
            + ", ".join(gene_missing)
            + "."
        )
    if platform.system().lower().startswith("win"):
        raise RuntimeError(
            "Taxon mode requires a bash-compatible shell because it invokes `qiime gemelli`. "
            "Use macOS/Linux or WSL for this mode."
        )
    if shutil.which("bash") is None:
        raise RuntimeError(
            "Taxon mode requires `bash` because it invokes `qiime gemelli`."
        )
    if shutil.which("conda") is None:
        raise RuntimeError(
            "Taxon mode requires `conda` to activate the qiime2 environment."
        )


def validate_mode_runtime(args: argparse.Namespace) -> None:
    workflow = getattr(args, "workflow", args.mode)

    if workflow == "taxon-function":
        _require_pro_rwct_runtime()
        return

    if workflow == "taxon":
        _require_gene_runtime()
        return

    raise ValueError(f"Unsupported mode: {workflow}")


def anchored_sigmoid_curve(
    x: np.ndarray | float,
    k: float,
    x0: float,
    alpha: float,
) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    raw = 1.0 / (1.0 + np.exp(-k * (x_arr - x0)))
    raw0 = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    denom = max(1e-12, 1.0 - raw0)
    y = alpha + (1.0 - alpha) * ((raw - raw0) / denom)
    return np.clip(y, alpha, 1.0)


def fit_sigmoid_curve(
    scenario_metrics_df: pd.DataFrame,
    alpha: float,
) -> Dict[str, object]:
    valid = (
        scenario_metrics_df
        .dropna(subset=["true_omega2", "power"])
        .sort_values("true_omega2")
        .reset_index(drop=True)
    )
    if valid.empty:
        return {
            "status": "no_valid_points",
            "x": np.array([], dtype=float),
            "y": np.array([], dtype=float),
            "params": None,
        }

    x = valid["true_omega2"].to_numpy(dtype=float)
    y = valid["power"].to_numpy(dtype=float)
    x = np.clip(x, 0.0, None)
    y = np.clip(y, alpha, 1.0)

    x_aug = np.concatenate([[0.0], x])
    y_aug = np.concatenate([[alpha], y])
    order = np.argsort(x_aug)
    x_aug = x_aug[order]
    y_aug = y_aug[order]

    unique_x = np.unique(x_aug)
    if len(unique_x) < 2:
        return {
            "status": "insufficient_unique_x",
            "x": x_aug,
            "y": y_aug,
            "params": None,
        }

    x_scale = max(float(np.nanmax(x_aug)), 1.0)
    pos_x = x_aug[x_aug > 0]
    p0 = np.array([
        max(0.5, 5.0 / x_scale),
        float(np.median(pos_x)) if len(pos_x) else 0.5 * x_scale,
    ])

    try:
        params, _ = curve_fit(
            lambda xv, k, x0: anchored_sigmoid_curve(xv, k, x0, alpha),
            x_aug,
            y_aug,
            p0=p0,
            bounds=([1e-6, -5.0 * x_scale], [1e3, 5.0 * x_scale]),
            maxfev=20000,
        )
        status = "ok"
    except Exception:
        params = p0
        status = "fallback_initial_guess"

    return {
        "status": status,
        "x": x_aug,
        "y": y_aug,
        "params": {"k": float(params[0]), "x0": float(params[1])},
    }


def evaluate_sigmoid_curve(
    fit_result: Dict[str, object],
    alpha: float,
    target_omega2: float,
    target_power: float,
) -> Dict[str, float]:
    params = fit_result.get("params")
    if not params:
        return {
            "curve_fit_status": str(fit_result.get("status")),
            "fitted_power_at_target_omega2": np.nan,
            "required_omega2_for_target_power": np.nan,
            "curve_reaches_target_power": False,
        }

    k = float(params["k"])
    x0 = float(params["x0"])
    fitted_power = float(anchored_sigmoid_curve(np.array([target_omega2]), k, x0, alpha)[0])

    if target_power <= alpha:
        required_omega2 = 0.0
        reaches_target = True
    else:
        hi = max(1.0, target_omega2 * 2.0, float(np.nanmax(fit_result["x"])) * 2.0 + 1e-6)
        reaches_target = False
        for _ in range(40):
            if float(anchored_sigmoid_curve(np.array([hi]), k, x0, alpha)[0]) >= target_power:
                reaches_target = True
                break
            hi *= 2.0
        if not reaches_target:
            required_omega2 = np.nan
        else:
            lo = 0.0
            for _ in range(80):
                mid = 0.5 * (lo + hi)
                if float(anchored_sigmoid_curve(np.array([mid]), k, x0, alpha)[0]) >= target_power:
                    hi = mid
                else:
                    lo = mid
            required_omega2 = hi

    return {
        "curve_fit_status": str(fit_result.get("status")),
        "fitted_power_at_target_omega2": fitted_power,
        "required_omega2_for_target_power": float(required_omega2) if not np.isnan(required_omega2) else np.nan,
        "curve_reaches_target_power": bool(reaches_target),
    }


def save_sigmoid_curve_plot(
    scenario_metrics_df: pd.DataFrame,
    fit_result: Dict[str, object],
    alpha: float,
    target_omega2: float,
    target_power: float,
    n_per_group: int,
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 5))

    valid = scenario_metrics_df.dropna(subset=["true_omega2", "power"]).copy()
    if not valid.empty:
        valid["true_omega2"] = valid["true_omega2"].clip(lower=0.0)
        valid["power"] = valid["power"].clip(lower=alpha, upper=1.0)
        for mode, color in [("dilution", "#d95f02"), ("enhancement", "#1b9e77")]:
            subset = valid[valid["mode"] == mode]
            if not subset.empty:
                ax.scatter(subset["true_omega2"], subset["power"], s=36, alpha=0.85, color=color, label=mode)

    ax.scatter([0.0], [alpha], s=50, color="black", label="anchor")
    data_x_max = float(np.nanmax(valid["true_omega2"])) if not valid.empty else 0.0
    x_max = max(target_omega2 * 1.25, data_x_max * 1.15, 0.12)

    params = fit_result.get("params")
    if params:
        k = float(params["k"])
        x0 = float(params["x0"])
        x_grid = np.linspace(0.0, x_max, 400)
        y_grid = anchored_sigmoid_curve(x_grid, k, x0, alpha)
        ax.plot(x_grid, y_grid, color="#386cb0", lw=2.0, label="anchored sigmoid")
        y_target = float(anchored_sigmoid_curve(np.array([target_omega2]), k, x0, alpha)[0])
        ax.scatter([target_omega2], [y_target], s=45, color="#386cb0")

    ax.axhline(alpha, color="gray", ls=":", lw=1.0)
    ax.axhline(target_power, color="red", ls="--", lw=1.0)
    ax.axvline(target_omega2, color="red", ls="--", lw=1.0)
    ax.set_xlim(-0.005, x_max)
    ax.set_ylim(alpha - 0.02, 1.02)
    ax.set_xlabel("True omega^2")
    ax.set_ylabel("Power")
    ax.set_title(f"n_per_group = {n_per_group}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _lower_tri_dm(dm: np.ndarray) -> np.ndarray:
    if dm.ndim != 2 or dm.shape[0] != dm.shape[1]:
        return np.array([])
    rows, cols = np.tril_indices_from(dm, k=-1)
    return dm[rows, cols]


def _summarize_ss(dm: pd.DataFrame, grouping: pd.Series) -> tuple[float, float, float]:
    grouping = grouping.loc[dm.columns]
    groups = grouping.dropna().unique()
    vals = dm.to_numpy()
    n = vals.shape[0]
    if n == 0:
        return 0.0, 0.0, 0.0
    sst = np.sum(_lower_tri_dm(vals) ** 2) / n
    ssw = 0.0
    for group_name in groups:
        members = grouping[grouping == group_name].index
        n_group = len(members)
        if n_group > 1:
            ssw += np.sum(_lower_tri_dm(dm.loc[members, members].to_numpy()) ** 2) / n_group
    ssa = sst - ssw
    return sst, ssw, ssa


def compute_omega2(dm: pd.DataFrame, group_map: pd.Series) -> float:
    load_core_runtime()
    k = len(group_map.dropna().unique())
    n = dm.shape[0]
    if k < 2 or n <= k:
        return 0.0
    sst, ssw, ssa = _summarize_ss(dm, group_map)
    df_b = k - 1
    df_w = n - k
    if df_w <= 0:
        return 0.0
    ms_w = ssw / df_w
    denom = sst + ms_w
    if denom == 0:
        return 0.0
    return (ssa - df_b * ms_w) / denom


def compute_permanova_p_value(
    dm: pd.DataFrame,
    group_map: pd.Series,
    permutations: int = 999,
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> float:
    p_value, _ = compute_permanova_p_value_with_status(
        dm=dm,
        group_map=group_map,
        permutations=permutations,
        failure_log_path=failure_log_path,
        failure_context=failure_context,
        seed=seed,
    )
    return p_value


def compute_permanova_p_value_with_status(
    dm: pd.DataFrame,
    group_map: pd.Series,
    permutations: int = 999,
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
) -> tuple[float, bool]:
    load_core_runtime()
    if group_map.isnull().any() or len(group_map.unique()) < 2:
        exc = ValueError("PERMANOVA requires at least two non-null groups.")
        log_permanova_failure(failure_log_path, failure_context, exc)
        return 1.0, True
    from skbio import DistanceMatrix as SkbioDistanceMatrix
    from skbio.stats.distance import permanova

    sk_dm = SkbioDistanceMatrix(np.ascontiguousarray(dm.values), ids=dm.columns)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="divide by zero encountered in scalar divide",
                category=RuntimeWarning,
            )
            try:
                result = permanova(sk_dm, grouping=group_map, permutations=permutations, seed=seed)
            except TypeError:
                # scikit-bio < 0.7 (e.g. QIIME 2 2024.10's skbio 0.6.0)
                # does not accept a ``seed`` keyword.  Reseed the global
                # NumPy RNG for the duration of the call so the result is
                # deterministic, matching the seeded behaviour of skbio ≥ 0.7.
                if seed is not None:
                    _stashed_state = np.random.get_state()
                    try:
                        np.random.seed(int(seed) % (2 ** 32))
                        result = permanova(sk_dm, grouping=group_map,
                                           permutations=permutations)
                    finally:
                        np.random.set_state(_stashed_state)
                else:
                    result = permanova(sk_dm, grouping=group_map,
                                       permutations=permutations)
        return float(result["p-value"]), False
    except ValueError as exc:
        log_permanova_failure(failure_log_path, failure_context, exc)
        return 1.0, True


def bootstrap_distance_matrix(
    dm: pd.DataFrame,
    subject_group_vector: Sequence[int],
    group_map: pd.Series,
    rng: Optional["np.random.Generator"] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    load_core_runtime()
    if rng is None:
        rng = np.random.default_rng()
    if dm is None or dm.empty:
        raise ValueError("Input distance matrix is empty.")
    group_map = group_map.loc[dm.columns]
    unique_groups = sorted(group_map.dropna().unique())
    if len(unique_groups) != len(subject_group_vector):
        raise ValueError("subject_group_vector length mismatch.")

    selected: list[str] = []
    for group_name, n_subjects in zip(unique_groups, subject_group_vector):
        members = group_map[group_map == group_name].index.tolist()
        if not members:
            raise ValueError(f"Group '{group_name}' has no samples.")
        selected.extend(rng.choice(members, size=n_subjects, replace=True))

    boot_df = dm.loc[selected, selected].copy()
    counts: Dict[str, int] = {}
    new_ids: list[str] = []
    new_groups: Dict[str, str] = {}
    for sample_id in selected:
        counts[sample_id] = counts.get(sample_id, 0) + 1
        new_id = f"{sample_id}_{counts[sample_id]}"
        new_ids.append(new_id)
        new_groups[new_id] = str(group_map[sample_id])

    boot_df.columns = new_ids
    boot_df.index = new_ids
    return boot_df, pd.Series(new_groups, name="group")


def prepare_bootstrap_sampling_plan(
    dm: pd.DataFrame,
    subject_group_vector: Sequence[int],
    group_map: pd.Series,
) -> Dict[str, Any]:
    load_core_runtime()
    if dm is None or dm.empty:
        raise ValueError("Input distance matrix is empty.")
    group_map = group_map.loc[dm.columns]
    unique_groups = sorted(group_map.dropna().unique())
    if len(unique_groups) != len(subject_group_vector):
        raise ValueError("subject_group_vector length mismatch.")

    sample_ids = dm.columns.to_numpy()
    positions = pd.Series(np.arange(len(sample_ids)), index=dm.columns)
    pools = []
    for group_name in unique_groups:
        members = group_map[group_map == group_name].index.tolist()
        if not members:
            raise ValueError(f"Group '{group_name}' has no samples.")
        pools.append(positions.loc[members].to_numpy(dtype=int))

    return {
        "dm_values": np.ascontiguousarray(dm.to_numpy()),
        "sample_ids": sample_ids,
        "group_names": unique_groups,
        "group_sizes": list(subject_group_vector),
        "group_pools": pools,
    }


def bootstrap_distance_matrix_from_plan(plan: Dict[str, Any], seed: int) -> tuple[pd.DataFrame, pd.Series]:
    load_core_runtime()
    rng = np.random.default_rng(seed)
    selected_positions = []
    selected_groups = []
    for group_name, n_subjects, pool in zip(
        plan["group_names"],
        plan["group_sizes"],
        plan["group_pools"],
    ):
        sampled = rng.choice(pool, size=n_subjects, replace=True)
        selected_positions.append(sampled)
        selected_groups.extend([group_name] * n_subjects)

    selected_positions = np.concatenate(selected_positions)
    boot_values = plan["dm_values"][np.ix_(selected_positions, selected_positions)].copy()
    original_names = plan["sample_ids"][selected_positions]
    counts: Dict[str, int] = {}
    new_ids: list[str] = []
    new_groups: Dict[str, str] = {}
    for sample_id, group_name in zip(original_names, selected_groups):
        counts[sample_id] = counts.get(sample_id, 0) + 1
        new_id = f"{sample_id}_{counts[sample_id]}"
        new_ids.append(new_id)
        new_groups[new_id] = group_name

    return pd.DataFrame(boot_values, index=new_ids, columns=new_ids), pd.Series(new_groups, name="group")


def convert_csv_to_feature_table_qza(csv_path: Path, out_qza_path: Path) -> Path:
    import biom
    from qiime2 import Artifact

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError(f"CSV '{csv_path}' must have at least 2 columns.")
    feature_col = df.columns[0]
    df = df.rename(columns={feature_col: "Taxon"}).drop_duplicates(subset=["Taxon"], keep="first").set_index("Taxon")
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    biom_table = biom.Table(
        df.values,
        observation_ids=df.index.astype(str).tolist(),
        sample_ids=df.columns.astype(str).tolist(),
    )
    artifact = Artifact.import_data("FeatureTable[Frequency]", biom_table)
    out_qza_path.parent.mkdir(parents=True, exist_ok=True)
    artifact.save(str(out_qza_path))
    print(f"Converted CSV -> QZA: {csv_path}  ->  {out_qza_path}")
    return out_qza_path


def convert_nwk_to_rooted_tree_qza(nwk_path: Path, out_qza_path: Path) -> Path:
    from qiime2 import Artifact
    from skbio import TreeNode

    tree = TreeNode.read(str(nwk_path), format="newick")
    try:
        is_rooted = bool(tree.root().children and len(tree.root().children) >= 2)
    except Exception:
        is_rooted = False
    if not is_rooted:
        try:
            tree = tree.root_at_midpoint()
        except Exception:
            pass
    out_qza_path.parent.mkdir(parents=True, exist_ok=True)
    Artifact.import_data("Phylogeny[Rooted]", tree).save(str(out_qza_path))
    print(f"Converted NWK -> QZA: {nwk_path}  ->  {out_qza_path}")
    return out_qza_path


def convert_csv_to_taxonomy_qza(csv_path: Path, out_qza_path: Path) -> Path:
    import os
    from qiime2 import Artifact

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    if df.shape[1] < 2:
        raise ValueError(f"Taxonomy CSV '{csv_path}' must have at least 2 columns.")
    lower_map = {c.lower().strip(): c for c in df.columns}
    id_col = lower_map.get("feature id") or lower_map.get("featureid") or lower_map.get("#otu id") or lower_map.get("id") or df.columns[0]
    tax_col = None
    for key in ("taxonomy", "taxon"):
        if key in lower_map and lower_map[key] != id_col:
            tax_col = lower_map[key]
            break
    if tax_col is None:
        tax_col = df.columns[1] if df.columns[1] != id_col else df.columns[-1]

    out_df = pd.DataFrame({"Feature ID": df[id_col].astype(str), "Taxon": df[tax_col].astype(str)})
    if "confidence" in lower_map:
        out_df["Confidence"] = df[lower_map["confidence"]]

    out_qza_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_tsv = out_qza_path.with_suffix(".tmp.tsv")
    out_df.to_csv(tmp_tsv, sep="\t", index=False)
    try:
        Artifact.import_data(
            "FeatureData[Taxonomy]",
            str(tmp_tsv),
            view_type="HeaderlessTSVTaxonomyFormat" if "Confidence" not in out_df.columns else "TSVTaxonomyFormat",
        ).save(str(out_qza_path))
    except Exception:
        with open(tmp_tsv, "w", encoding="utf-8") as handle:
            handle.write("Feature ID\tTaxon")
            if "Confidence" in out_df.columns:
                handle.write("\tConfidence")
            handle.write("\n")
            for _, row in out_df.iterrows():
                line = f"{row['Feature ID']}\t{row['Taxon']}"
                if "Confidence" in out_df.columns:
                    line += f"\t{row['Confidence']}"
                handle.write(line + "\n")
        Artifact.import_data("FeatureData[Taxonomy]", str(tmp_tsv), view_type="TSVTaxonomyFormat").save(str(out_qza_path))
    finally:
        if tmp_tsv.exists():
            try:
                os.remove(tmp_tsv)
            except OSError:
                pass
    return out_qza_path


def relabel_internal_tree_nodes(tree) -> Any:
    relabeled_tree = tree.copy()
    counter = 0
    for node in relabeled_tree.postorder():
        if not node.is_tip():
            counter += 1
            node.name = f"inode_{counter}"
    return relabeled_tree


def prepare_gene_tree_artifact(base_tree_qza_path: Path, out_dir: Path) -> tuple[Path, Any]:
    from qiime2 import Artifact
    from skbio import TreeNode

    tree_artifact = Artifact.load(str(base_tree_qza_path))
    relabeled = relabel_internal_tree_nodes(tree_artifact.view(TreeNode))
    out_path = out_dir / f"{base_tree_qza_path.stem}_internal_unique.qza"
    Artifact.import_data("Phylogeny[Rooted]", relabeled).save(str(out_path))
    return out_path, relabeled


def apply_branch_length_jittering(tree, sigma: float, rng) -> Any:
    perturbed = tree.copy()
    if sigma <= 0:
        return perturbed
    for node in perturbed.traverse():
        if node.length is not None and node.length > 0:
            node.length = node.length * np.exp(sigma * rng.standard_normal())
    return perturbed


def apply_nni_on_low_support_nodes(
    tree,
    flip_probability: float,
    rng,
    support_threshold: Optional[float] = None,
) -> Any:
    perturbed = tree.copy()
    if flip_probability <= 0:
        return perturbed

    candidates = []
    for node in perturbed.non_tips():
        if node.parent is None or node.parent.parent is None:
            continue
        if support_threshold is not None:
            support = None
            try:
                support = float(node.name) if node.name else None
            except (ValueError, TypeError):
                support = None
            if support is not None and support >= support_threshold:
                continue
        candidates.append(node)

    for node in candidates:
        if rng.random() > flip_probability:
            continue
        parent = node.parent
        children = list(node.children)
        siblings = [child for child in parent.children if child is not node]
        if len(children) < 2 or len(siblings) < 1:
            continue
        swap_child = children[rng.integers(0, len(children))]
        swap_sibling = siblings[rng.integers(0, len(siblings))]
        node.remove(swap_child)
        parent.remove(swap_sibling)
        node.append(swap_sibling)
        parent.append(swap_child)

    return perturbed


def generate_perturbed_tree(
    base_tree,
    sigma: float,
    nni_prob: float,
    rng,
    support_threshold: Optional[float] = None,
) -> Any:
    perturbed = apply_nni_on_low_support_nodes(
        base_tree,
        flip_probability=nni_prob,
        rng=rng,
        support_threshold=support_threshold,
    )
    return apply_branch_length_jittering(perturbed, sigma=sigma, rng=rng)


def save_perturbed_tree_as_qza(perturbed_tree, out_dir: Path, output_stem: str) -> Path:
    from qiime2 import Artifact

    labeled = relabel_internal_tree_nodes(perturbed_tree)
    out_path = out_dir / f"temp_tree_{output_stem}.qza"
    Artifact.import_data("Phylogeny[Rooted]", labeled).save(str(out_path))
    return out_path


def save_perturbed_tree_as_newick(perturbed_tree, out_dir: Path, output_stem: str) -> Path:
    out_path = out_dir / f"tree_perturbed_{output_stem}.nwk"
    with open(out_path, "w", encoding="utf-8") as handle:
        try:
            perturbed_tree.write(handle)
        except TypeError:
            perturbed_tree.write(handle, format="newick")
    return out_path


def materialize_perturbed_tree(
    base_tree,
    sigma: float,
    nni_prob: float,
    random_seed: int,
    out_dir: Path,
    output_stem: str,
    output_format: str,
    support_threshold: Optional[float] = None,
) -> Optional[Path]:
    if sigma == 0.0 and nni_prob == 0.0:
        return None

    tree_rng = make_tree_rng(random_seed, output_stem)
    perturbed_tree = generate_perturbed_tree(
        base_tree,
        sigma=sigma,
        nni_prob=nni_prob,
        rng=tree_rng,
        support_threshold=support_threshold,
    )
    if output_format == "qza":
        return save_perturbed_tree_as_qza(perturbed_tree, out_dir, output_stem)
    if output_format == "newick":
        return save_perturbed_tree_as_newick(perturbed_tree, out_dir, output_stem)
    raise ValueError(f"Unsupported perturbed-tree output format: {output_format}")


def prepare_group_filtered_table(base_table_qza_path: Path, group_map: pd.Series, out_dir: Path, output_name: str) -> tuple[Path, pd.Series]:
    import biom
    from qiime2 import Artifact

    artifact = Artifact.load(str(base_table_qza_path))
    biom_table = artifact.view(biom.Table)
    df = biom_table.to_dataframe()
    valid_samples = [sample_id for sample_id in group_map.index if sample_id in df.columns]
    if not valid_samples:
        raise ValueError("No samples in group_map matched the input feature table.")
    filtered_group_map = group_map.loc[valid_samples]
    subset_df = df[valid_samples].copy()
    new_biom_table = biom.Table(subset_df.values, observation_ids=subset_df.index, sample_ids=subset_df.columns)
    out_path = out_dir / output_name
    Artifact.import_data("FeatureTable[Frequency]", new_biom_table).save(str(out_path))
    return out_path, filtered_group_map


def export_group_pool_feature_table(base_table_qza_path: Path, group_map: pd.Series, out_dir: Path, output_name: str) -> Path:
    import biom
    from qiime2 import Artifact

    artifact = Artifact.load(str(base_table_qza_path))
    biom_table = artifact.view(biom.Table)
    df = biom_table.to_dataframe()
    valid_samples = [sample_id for sample_id in group_map.index if sample_id in df.columns]
    if not valid_samples:
        raise ValueError("No overlapping samples.")
    subset_df = df[valid_samples].copy()
    new_biom_table = biom.Table(subset_df.values, observation_ids=subset_df.index, sample_ids=subset_df.columns)
    out_path = out_dir / output_name
    Artifact.import_data("FeatureTable[Frequency]", new_biom_table).save(str(out_path))
    return out_path


def apply_centroid_convergence_effect(
    base_table_qza_path: Path,
    group_map: pd.Series,
    out_dir: Path,
    effect_level: float,
    random_seed: int,
    preserve_library_size: bool = True,
    pseudocount: float = 0.5,
) -> Path:
    import biom
    from qiime2 import Artifact

    _ = random_seed
    artifact = Artifact.load(str(base_table_qza_path))
    biom_table = artifact.view(biom.Table)
    df = biom_table.to_dataframe()
    df_t = df.transpose().astype(float)
    aligned_group_map = group_map.loc[group_map.index.intersection(df_t.index)]
    df_t_aligned = df_t.loc[aligned_group_map.index]
    if len(df_t_aligned) == 0:
        raise ValueError("No overlapping samples between group_map and table.")

    def clr_transform(sample_by_feature_df: pd.DataFrame, pseudo: float) -> pd.DataFrame:
        x = sample_by_feature_df.to_numpy(dtype=float, copy=True)
        x += pseudo
        row_sums = x.sum(axis=1, keepdims=True)
        row_sums[row_sums <= 0] = 1.0
        proportions = x / row_sums
        logp = np.log(proportions)
        clr = logp - logp.mean(axis=1, keepdims=True)
        return pd.DataFrame(clr, index=sample_by_feature_df.index, columns=sample_by_feature_df.columns)

    def inverse_clr_to_composition(clr_df: pd.DataFrame) -> pd.DataFrame:
        z = clr_df.to_numpy(dtype=float, copy=True)
        z -= z.max(axis=1, keepdims=True)
        exp_z = np.exp(z)
        denom = exp_z.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        comp = exp_z / denom
        return pd.DataFrame(comp, index=clr_df.index, columns=clr_df.columns)

    def allocate_integer_counts(
        composition_df: pd.DataFrame,
        target_library_sizes: pd.Series,
    ) -> pd.DataFrame:
        comp = composition_df.to_numpy(dtype=float, copy=True)
        targets = np.rint(target_library_sizes.to_numpy(dtype=float)).astype(int)
        out = np.zeros_like(comp, dtype=int)
        for i in range(comp.shape[0]):
            target = max(int(targets[i]), 0)
            if target == 0:
                continue
            expected = comp[i] * target
            base = np.floor(expected).astype(int)
            deficit = target - int(base.sum())
            if deficit > 0:
                frac = expected - base
                promote = np.argsort(-frac)[:deficit]
                base[promote] += 1
            elif deficit < 0:
                frac = expected - base
                demote = np.argsort(frac)[:(-deficit)]
                for idx in demote:
                    if base[idx] > 0:
                        base[idx] -= 1
            out[i] = base
        return pd.DataFrame(out, index=composition_df.index, columns=composition_df.columns)

    original_lib_sizes = df_t_aligned.sum(axis=1)
    clr_df = clr_transform(df_t_aligned, pseudo=pseudocount)
    global_centroid = clr_df.mean(axis=0)
    between_scale = float(np.exp(effect_level))

    effected_clr_parts = []
    for _, sample_names in aligned_group_map.groupby(aligned_group_map).groups.items():
        sample_names = list(sample_names)
        group_block = clr_df.loc[sample_names]
        group_centroid = group_block.mean(axis=0)
        within_residual = group_block.sub(group_centroid, axis=1)
        shifted_centroid = global_centroid + between_scale * (group_centroid - global_centroid)
        shifted_clr = within_residual.add(shifted_centroid, axis=1)
        effected_clr_parts.append(shifted_clr)
    effected_clr_df = pd.concat(effected_clr_parts).loc[df_t_aligned.index]

    effected_comp_df = inverse_clr_to_composition(effected_clr_df)
    if preserve_library_size:
        effected_df_t = allocate_integer_counts(effected_comp_df, original_lib_sizes)
    else:
        effected_df_t = allocate_integer_counts(
            effected_comp_df,
            pd.Series(np.ones(len(effected_comp_df)), index=effected_comp_df.index),
        )

    effected_df = effected_df_t.transpose()
    effected_df[effected_df < 0] = 0

    direction = "enhance" if effect_level > 0 else ("dilute" if effect_level < 0 else "identity")
    final_lib = effected_df.sum(axis=0)
    lib_drift = (final_lib.values - original_lib_sizes.values) / np.where(
        original_lib_sizes.values > 0,
        original_lib_sizes.values,
        1.0,
    )
    mean_drift_pct = 100.0 * float(np.mean(np.abs(lib_drift)))
    baseline_comp = (df_t_aligned.add(pseudocount)).div(
        (df_t_aligned.add(pseudocount)).sum(axis=1),
        axis=0,
    )
    mean_l1_shift = float(np.mean(np.abs(effected_comp_df.to_numpy() - baseline_comp.to_numpy())))
    print(
        f"  [compositional-centroid {direction}] alpha={effect_level:+.3f}  "
        f"mean_L1_comp_shift={mean_l1_shift:.6f}  "
        f"mean|lib-size drift|={mean_drift_pct:.2f}%"
    )

    new_biom_table = biom.Table(effected_df.values, observation_ids=effected_df.index, sample_ids=effected_df.columns)
    new_artifact = Artifact.import_data("FeatureTable[Frequency]", new_biom_table)
    out_path = out_dir / f"temp_table_effect_{effect_level:+.3f}.qza"
    new_artifact.save(str(out_path))
    return out_path


def compute_gemelli_rpca_distance(
    table_qza_path: Path,
    tree_qza_path: Path,
    taxonomy_qza_path: Path,
    out_dir: Path,
    qiime_env_name: str,
    output_stem: str,
):
    import os
    import subprocess
    from qiime2 import Artifact
    from skbio import DistanceMatrix

    _ = taxonomy_qza_path
    out_path = out_dir / f"temp_dist_{output_stem}.qza"
    dummy_files = {
        "biplot": out_dir / f"dummy_biplot_{output_stem}.qza",
        "node_tree": out_dir / f"dummy_tree_{output_stem}.qza",
        "node_table": out_dir / f"dummy_node_table_{output_stem}.qza",
    }
    command = [
        "qiime", "gemelli", "phylogenetic-rpca-without-taxonomy",
        "--i-table", str(table_qza_path),
        "--i-phylogeny", str(tree_qza_path),
        "--o-distance-matrix", str(out_path),
        "--o-biplot", str(dummy_files["biplot"]),
        "--o-counts-by-node-tree", str(dummy_files["node_tree"]),
        "--o-counts-by-node", str(dummy_files["node_table"]),
    ]
    conda_command = (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {qiime_env_name} && {' '.join(command)}"
    )
    result = subprocess.run(conda_command, shell=True, capture_output=True, text=True, executable="/bin/bash")
    for path in dummy_files.values():
        if path.exists():
            os.remove(path)
    if result.returncode != 0:
        print(f"\nGemelli failed for {output_stem}.")
        print(result.stderr)
        if out_path.exists():
            os.remove(out_path)
        return None
    dist_matrix = Artifact.load(str(out_path)).view(DistanceMatrix)
    if out_path.exists():
        os.remove(out_path)
    return dist_matrix


def compute_classical_mds_coordinates(dm, tol: float = 1e-10) -> "pd.DataFrame":
    """Embed a distance matrix into Euclidean coordinates via classical MDS."""
    load_core_runtime()
    dm_df = dm.to_data_frame()
    ids = dm_df.index.to_list()
    d = dm_df.to_numpy(dtype=float)
    n = d.shape[0]
    if n == 0:
        return pd.DataFrame(index=ids)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d ** 2) @ j
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep = eigvals > tol
    if not np.any(keep):
        return pd.DataFrame(np.zeros((n, 1)), index=ids, columns=["pc1"])
    coords = eigvecs[:, keep] * np.sqrt(eigvals[keep])
    columns = [f"pc{i + 1}" for i in range(coords.shape[1])]
    return pd.DataFrame(coords, index=ids, columns=columns)


def modulate_taxon_gemelli_geometry(
    base_dm,
    group_map: "pd.Series",
    effect_level: float,
):
    """Apply the figure/taxon.py centroid-scaling operator in Gemelli latent space."""
    load_core_runtime()
    from scipy.spatial.distance import pdist, squareform
    from skbio import DistanceMatrix

    base_df = base_dm.to_data_frame()
    common = base_df.index.intersection(group_map.index)
    if len(common) == 0:
        raise ValueError("No overlapping samples between base distance matrix and group map.")
    aligned_dm = DistanceMatrix(base_df.loc[common, common].to_numpy(), ids=list(common))
    aligned_group_map = group_map.loc[common]
    coords = compute_classical_mds_coordinates(aligned_dm)
    global_centroid = coords.mean(axis=0)
    between_scale = float(np.exp(effect_level))

    transformed_parts = []
    for group_name in aligned_group_map.unique():
        members = aligned_group_map[aligned_group_map == group_name].index
        group_coords = coords.loc[members]
        centroid = group_coords.mean(axis=0)
        between_component = centroid - global_centroid
        within_component = group_coords - centroid
        transformed = global_centroid + between_scale * between_component + within_component
        transformed_parts.append(transformed)
    transformed_coords = pd.concat(transformed_parts).loc[coords.index]

    dist_values = squareform(pdist(transformed_coords.to_numpy(dtype=float)))
    np.fill_diagonal(dist_values, 0.0)
    return DistanceMatrix(dist_values, ids=list(transformed_coords.index))


def run_gene_bootstrap_permanova(
    boot_seed: int,
    original_dm,
    original_group_map: pd.Series,
    subject_group_vector: Sequence[int],
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
    permutations: int = 999,
) -> tuple[float, float, int]:
    load_core_runtime()
    if original_dm is None:
        return np.nan, np.nan, 1
    original_dm_df = original_dm.to_data_frame()
    common_samples = original_dm_df.index.intersection(original_group_map.index)
    original_dm_df = original_dm_df.loc[common_samples, common_samples]
    aligned_group_map = original_group_map.loc[common_samples]
    context = {**(failure_context or {}), "seed": int(boot_seed)}
    try:
        rng = np.random.default_rng(int(boot_seed))
        boot_df, boot_group_map = bootstrap_distance_matrix(
            original_dm_df,
            subject_group_vector,
            group_map=aligned_group_map,
            rng=rng,
        )
        aligned_boot_group_map = boot_group_map.loc[boot_df.index]
        p_value, permanova_failed = compute_permanova_p_value_with_status(
            boot_df,
            group_map=aligned_boot_group_map,
            permutations=int(permutations),
            failure_log_path=failure_log_path,
            failure_context=context,
            seed=make_permanova_seed(boot_seed),
        )
        omega2 = compute_omega2(boot_df, group_map=aligned_boot_group_map)
        return p_value, omega2, int(permanova_failed)
    except Exception as exc:
        log_permanova_failure(failure_log_path, context, exc)
        return np.nan, np.nan, 1


ID_COLS = ["Taxon", "Function"]


def detect_input_data_type(data_path: Path) -> str:
    suffix = data_path.suffix.lower()
    if suffix == ".qza":
        return "taxon"
    if suffix in (".csv", ".tsv", ".txt"):
        sep = "\t" if suffix in (".tsv", ".txt") else ","
        try:
            header = pd.read_csv(data_path, sep=sep, nrows=0, encoding="utf-8-sig").columns
        except Exception:
            try:
                header = pd.read_csv(data_path, sep=sep, nrows=0, encoding="utf-8").columns
            except Exception:
                return "unknown"
        cols_lower = {c.strip().lower() for c in header}
        if "taxon" in cols_lower and "function" in cols_lower:
            return "taxon-function"
        if "taxon" in cols_lower:
            return "taxon"
    return "unknown"


def load_protein_long_table(base_df_path: Path, group_map: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    df_template = pd.read_csv(base_df_path, encoding="utf-8")
    sample_cols = [sample_id for sample_id in group_map.index if sample_id in df_template.columns]
    if not sample_cols:
        raise ValueError("No samples matched.")
    df_scenario = df_template[ID_COLS + sample_cols].copy()
    group_map = group_map.loc[sample_cols]
    df_scenario[sample_cols] = df_scenario[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)
    return df_scenario, group_map


def compute_phylofunc_distance_matrix(long_df: pd.DataFrame, tree_path_str: str) -> pd.DataFrame:
    import contextlib
    import os
    import warnings
    from phylofunc import PhyloFunc_matrix

    local_df = long_df.copy()
    sample_cols = [col for col in local_df.columns if col not in ID_COLS]
    if sample_cols:
        local_df[sample_cols] = local_df[sample_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    with open(os.devnull, "w") as sink, warnings.catch_warnings(), contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        warnings.simplefilter("ignore")
        # PhyloFunc_matrix keeps its own tree_file-keyed cache for parsed trees and branch tables.
        phylofunc_dm_true = PhyloFunc_matrix(tree_file=tree_path_str, sample_data=local_df)
    return phylofunc_dm_true


def apply_rwct_effect_scaling(
    df: pd.DataFrame,
    gamma: float,
    group_map: pd.Series,
    ridge: float = 1.0,
    normalize_weights: bool = True,
) -> tuple[pd.DataFrame, dict]:
    if gamma == 0 or df.empty:
        return df.copy(), {
            "presence_diff": 0,
            "shift_max": 0.0,
            "lib_drift_med": 0.0,
            "weight_min": np.nan,
            "weight_max": np.nan,
        }

    sample_cols = [c for c in df.columns if c not in ID_COLS]
    if not sample_cols:
        return df.copy(), {
            "presence_diff": 0,
            "shift_max": 0.0,
            "lib_drift_med": 0.0,
            "weight_min": np.nan,
            "weight_max": np.nan,
        }

    groups: Dict[str, List[str]] = {}
    for sample_id, group_name in group_map.items():
        if sample_id in sample_cols:
            groups.setdefault(str(group_name), []).append(sample_id)
    valid_groups = sorted(group_name for group_name in groups if len(groups[group_name]) >= 2)
    if len(valid_groups) < 2:
        print("Warning: fewer than two valid groups; skipping RWCT translation.")
        return df.copy(), {
            "presence_diff": 0,
            "shift_max": 0.0,
            "lib_drift_med": 0.0,
            "weight_min": np.nan,
            "weight_max": np.nan,
        }

    out = df.copy()
    out[sample_cols] = out[sample_cols].astype(float)
    x = out[sample_cols].to_numpy(dtype=float)
    presence = x > 0
    n_rows = x.shape[0]
    eps = 1e-12

    col_pos = {
        group_name: np.asarray([sample_cols.index(s) for s in groups[group_name]], dtype=int)
        for group_name in valid_groups
    }
    row_min_pos = np.zeros(n_rows, dtype=float)
    for i in range(n_rows):
        row_pos = x[i][x[i] > 0]
        row_min_pos[i] = (row_pos.min() / 100.0) if len(row_pos) else 1e-9
    log_x = np.log(np.where(presence, x, row_min_pos[:, None]))

    means = np.zeros((len(valid_groups), n_rows), dtype=float)
    for k, group_name in enumerate(valid_groups):
        means[k] = log_x[:, col_pos[group_name]].mean(axis=1)
    grand_mean = means.mean(axis=0)
    centered_means = means - grand_mean[None, :]

    pooled_ss = np.zeros(n_rows, dtype=float)
    df_within = 0
    for group_name in valid_groups:
        cols = col_pos[group_name]
        if len(cols) >= 2:
            pooled_ss += (len(cols) - 1) * log_x[:, cols].var(axis=1, ddof=1)
            df_within += len(cols) - 1
    sigma2 = pooled_ss / max(df_within, 1)
    q = 1.0 / (sigma2 + ridge)
    if normalize_weights:
        finite_q = q[np.isfinite(q) & (q > 0)]
        if len(finite_q):
            q = q / np.median(finite_q)

    directions = centered_means * q[None, :]
    directions = directions - directions.mean(axis=0, keepdims=True)
    add_shift = np.zeros_like(x, dtype=float)
    for k, group_name in enumerate(valid_groups):
        add_shift[:, col_pos[group_name]] = (gamma * directions[k])[:, None]

    new_vals = np.where(presence, x * np.exp(add_shift), 0.0)
    underflow = presence & (new_vals <= 0)
    if underflow.any():
        floor_mat = np.broadcast_to(row_min_pos[:, None], new_vals.shape)
        new_vals = np.where(underflow, floor_mat, new_vals)

    out[sample_cols] = new_vals
    new_presence = new_vals > 0
    orig_lib = x.sum(axis=0)
    new_lib = new_vals.sum(axis=0)
    info = {
        "presence_diff": int((presence != new_presence).sum()),
        "shift_max": float(np.max(np.abs(add_shift))) if add_shift.size else 0.0,
        "lib_drift_med": float(np.median(np.abs(new_lib - orig_lib) / (orig_lib + eps))),
        "weight_min": float(np.nanmin(q)),
        "weight_max": float(np.nanmax(q)),
    }
    direction = "ENHANCE" if gamma > 0 else "DILUTE"
    print(
        f"  [RWCT] gamma={gamma:+.3f} {direction}  "
        f"presence_diff={info['presence_diff']} (must=0)  "
        f"|log-shift|_max={info['shift_max']:.2f}  "
        f"median|libsize drift|={100 * info['lib_drift_med']:.1f}%"
    )
    return out, info


def summarize_degree_preservation(original: pd.DataFrame, changed: pd.DataFrame) -> dict[str, int]:
    sample_cols = [c for c in original.columns if c not in ID_COLS]
    before = original[sample_cols].to_numpy(dtype=float) > 0
    after = changed[sample_cols].to_numpy(dtype=float) > 0
    taxon_codes = pd.Categorical(original["Taxon"]).codes
    function_codes = pd.Categorical(original["Function"]).codes

    def sample_degrees(mask: np.ndarray, codes: np.ndarray) -> np.ndarray:
        degrees = np.zeros(mask.shape[1], dtype=int)
        for sample_idx in range(mask.shape[1]):
            present_codes = codes[mask[:, sample_idx]]
            degrees[sample_idx] = len(np.unique(present_codes)) if len(present_codes) else 0
        return degrees

    return {
        "presence_diff": int((before != after).sum()),
        "max_taxon_degree_diff": int(np.max(np.abs(sample_degrees(before, taxon_codes) - sample_degrees(after, taxon_codes)))) if len(sample_cols) else 0,
        "max_function_degree_diff": int(np.max(np.abs(sample_degrees(before, function_codes) - sample_degrees(after, function_codes)))) if len(sample_cols) else 0,
    }


def has_finite_distance_matrix(dm: Optional[pd.DataFrame]) -> bool:
    if dm is None or dm.empty:
        return False
    return np.isfinite(pd.DataFrame(dm).apply(pd.to_numeric, errors="coerce").to_numpy()).all()


def summarize_rwct_distance_metrics(
    phylofunc_dm_true: pd.DataFrame,
    group_map: pd.Series,
    boot_number: int,
    subject_group_vector: Sequence[int],
    alpha: float,
    n_jobs: int,
    rng_seeds: np.ndarray,
    permutations: int,
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
) -> dict[str, float]:
    load_core_runtime()
    if phylofunc_dm_true is None or phylofunc_dm_true.empty:
        return {
            "power_phylofunc": np.nan,
            "mean_boot_omega2_phylofunc": np.nan,
            "true_omega2_phylofunc": np.nan,
            "failed_bootstraps": 0,
        }
    common = phylofunc_dm_true.index.intersection(group_map.index)
    aligned_dm = phylofunc_dm_true.loc[common, common].copy().apply(pd.to_numeric, errors="coerce")
    aligned_group_map = group_map.loc[common]
    if aligned_dm.empty:
        return {
            "power_phylofunc": np.nan,
            "mean_boot_omega2_phylofunc": np.nan,
            "true_omega2_phylofunc": np.nan,
            "failed_bootstraps": 0,
        }
    aligned_dm_values = aligned_dm.to_numpy(copy=True)
    np.fill_diagonal(aligned_dm_values, 0.0)
    aligned_dm.iloc[:, :] = aligned_dm_values
    finite_mask = np.isfinite(aligned_dm_values)
    if not finite_mask.all():
        valid_index = aligned_dm.index[finite_mask.all(axis=1)]
        aligned_dm = aligned_dm.loc[valid_index, valid_index].copy()
        aligned_group_map = aligned_group_map.loc[valid_index]
    true_omega2 = max(float(compute_omega2(aligned_dm, group_map=aligned_group_map)), 0.0)
    if boot_number <= 0:
        return {
            "power_phylofunc": np.nan,
            "mean_boot_omega2_phylofunc": np.nan,
            "true_omega2_phylofunc": true_omega2,
            "failed_bootstraps": 0,
        }

    plan = prepare_bootstrap_sampling_plan(aligned_dm, subject_group_vector, aligned_group_map)

    def _run_one(seed: int) -> tuple[float, float, int]:
        context = {**(failure_context or {}), "seed": int(seed)}
        try:
            boot_df, boot_group_map = bootstrap_distance_matrix_from_plan(plan, seed=seed)
            p_value, permanova_failed = compute_permanova_p_value_with_status(
                boot_df,
                group_map=boot_group_map,
                permutations=permutations,
                failure_log_path=failure_log_path,
                failure_context=context,
                seed=make_permanova_seed(seed),
            )
            omega2 = compute_omega2(boot_df, group_map=boot_group_map)
            return p_value, omega2, int(permanova_failed)
        except Exception as exc:
            log_permanova_failure(failure_log_path, context, exc)
            return np.nan, np.nan, 1

    results = Parallel(n_jobs=n_jobs)(delayed(_run_one)(int(seed)) for seed in rng_seeds[:boot_number])
    p_values, omega_values, failed_flags = zip(*results) if results else ([], [], [])
    p_values = np.asarray(p_values, dtype=float)
    omega_values = np.asarray(omega_values, dtype=float)
    failed_bootstraps = int(np.sum(np.asarray(failed_flags, dtype=int))) if len(failed_flags) else 0
    valid_p = p_values[np.isfinite(p_values)]
    return {
        "power_phylofunc": float((valid_p <= alpha).mean()) if len(valid_p) else np.nan,
        "mean_boot_omega2_phylofunc": max(float(np.nanmean(omega_values)), 0.0),
        "true_omega2_phylofunc": true_omega2,
        "failed_bootstraps": failed_bootstraps,
    }


def precompute_rwct_distance_scenarios(
    df_scenario: pd.DataFrame,
    tree_path_str: str,
    group_map: pd.Series,
    enhancement_gammas: np.ndarray,
    dilution_gammas: np.ndarray,
    ridge: float,
    normalize_weights: bool,
) -> tuple[list[dict], list[dict]]:
    print("\n" + "=" * 70)
    print("Precomputing full-sample PhyloFunc distance matrices (RWCT)")
    print("=" * 70)
    base_dm = compute_phylofunc_distance_matrix(df_scenario, tree_path_str)

    increase_scenarios = []
    for gamma in enhancement_gammas:
        if np.isclose(gamma, 0):
            increase_scenarios.append({
                "gamma": float(gamma),
                "dm": base_dm,
                "presence_diff": 0,
                "max_taxon_degree_diff": 0,
                "max_function_degree_diff": 0,
                "shift_max": 0.0,
                "lib_drift_med": 0.0,
            })
            continue
        effected, info = apply_rwct_effect_scaling(
            df_scenario.copy(),
            float(gamma),
            group_map,
            ridge=ridge,
            normalize_weights=normalize_weights,
        )
        degree = summarize_degree_preservation(df_scenario, effected)
        scenario_dm = compute_phylofunc_distance_matrix(effected, tree_path_str)
        increase_scenarios.append({"gamma": float(gamma), "dm": scenario_dm, **info, **degree})

    decrease_scenarios = []
    for gamma in dilution_gammas:
        if np.isclose(gamma, 0):
            decrease_scenarios.append({
                "gamma": float(gamma),
                "dm": base_dm,
                "presence_diff": 0,
                "max_taxon_degree_diff": 0,
                "max_function_degree_diff": 0,
                "shift_max": 0.0,
                "lib_drift_med": 0.0,
            })
            continue
        diluted, info = apply_rwct_effect_scaling(
            df_scenario.copy(),
            float(gamma),
            group_map,
            ridge=ridge,
            normalize_weights=normalize_weights,
        )
        degree = summarize_degree_preservation(df_scenario, diluted)
        scenario_dm = compute_phylofunc_distance_matrix(diluted, tree_path_str)
        decrease_scenarios.append({"gamma": float(gamma), "dm": scenario_dm, **info, **degree})
    return increase_scenarios, decrease_scenarios


def build_taxonfunction_cache_metadata(
    args: argparse.Namespace,
    current_tree_path: str,
) -> Dict[str, Any]:
    return {
        "cache_version": 2,
        "table_sha256": file_sha256(args.table),
        "group_sha256": file_sha256(args.group),
        "tree_sha256": file_sha256(current_tree_path),
        "params": {
            "ridge": float(args.ridge),
            "increase_num": int(args.increase_num),
            "decrease_num": int(args.decrease_num),
            "increase_max": float(args.increase_max),
            "decrease_min": float(args.decrease_min),
            "normalize_weights": not args.no_normalize_weights,
        },
    }


def load_taxonfunction_scenario_cache(
    cache_file: Path,
    expected_metadata: Dict[str, Any],
) -> Optional[tuple[list[dict], list[dict]]]:
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    if payload.get("metadata") != expected_metadata:
        return None
    try:
        return payload["increase_scenarios"], payload["decrease_scenarios"]
    except KeyError:
        return None


def save_taxonfunction_scenario_cache(
    cache_file: Path,
    metadata: Dict[str, Any],
    increase_scenarios: list[dict],
    decrease_scenarios: list[dict],
) -> None:
    with open(cache_file, "wb") as handle:
        pickle.dump(
            {
                "metadata": metadata,
                "increase_scenarios": increase_scenarios,
                "decrease_scenarios": decrease_scenarios,
            },
            handle,
        )


def build_taxon_distance_cache_metadata(
    args: argparse.Namespace,
    filtered_group_map: "pd.Series",
    sigma: float,
    nni_prob: float,
    effect_level: float,
    effect_seed: int,
) -> Dict[str, Any]:
    load_core_runtime()
    group_payload = pd.DataFrame(
        {
            "sample_id": filtered_group_map.index.astype(str),
            "group_name": filtered_group_map.astype(str).to_numpy(),
        }
    ).sort_values("sample_id")
    group_csv = group_payload.to_csv(index=False).encode("utf-8")
    return {
        "cache_version": 2,
        "table_sha256": file_sha256(args.table),
        "group_sha256": file_sha256(args.group),
        "tree_sha256": file_sha256(args.tree),
        "taxonomy_sha256": file_sha256(args.taxonomy),
        "group_map_sha256": hashlib.sha256(group_csv).hexdigest(),
        "params": {
            "taxon_effect_method": "latent_space_centroid_v1",
            "effect_level": round(float(effect_level), 12),
            "effect_seed": int(effect_seed),
            "qiime_env": str(args.qiime_env),
            "sigma": float(sigma),
            "nni_prob": float(nni_prob),
            "random_seed": int(args.random_seed),
            "nni_support_threshold": (
                None
                if args.nni_support_threshold is None
                else float(args.nni_support_threshold)
            ),
        },
    }


def load_taxon_distance_cache(
    cache_file: Path,
    expected_metadata: Dict[str, Any],
):
    if not cache_file.exists():
        return None
    with open(cache_file, "rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        return None
    if payload.get("metadata") != expected_metadata:
        return None
    return payload.get("distance_matrix")


def save_taxon_distance_cache(
    cache_file: Path,
    metadata: Dict[str, Any],
    distance_matrix,
) -> None:
    with open(cache_file, "wb") as handle:
        pickle.dump(
            {
                "metadata": metadata,
                "distance_matrix": distance_matrix,
            },
            handle,
        )


def estimate_minimum_sample_size(
    *,
    target_power: float,
    target_omega2: float,
    alpha: float,
    min_n: int,
    max_n: int,
    curve_plot_dir: Path,
    evaluate_scenarios_for_n_fn: Callable[[int, str], pd.DataFrame],
) -> tuple[Optional[int], pd.DataFrame, pd.DataFrame]:
    coarse_step = 10
    n_results_by_n: Dict[int, Dict[str, Any]] = {}
    scenario_metrics_by_n: Dict[int, pd.DataFrame] = {}
    fit_results_by_n: Dict[int, Dict[str, object]] = {}
    scenario_metrics_frames: List[pd.DataFrame] = []
    minimum_n = None

    def evaluate_one_n(n_per_group: int, search_stage: str) -> bool:
        if n_per_group in n_results_by_n:
            return bool(n_results_by_n[n_per_group]["qualifies"])

        scenario_metrics_df = evaluate_scenarios_for_n_fn(n_per_group, search_stage).copy()
        scenario_metrics_df["search_stage"] = search_stage
        scenario_metrics_by_n[n_per_group] = scenario_metrics_df
        scenario_metrics_frames.append(scenario_metrics_df)

        fit_result = fit_sigmoid_curve(
            scenario_metrics_df=scenario_metrics_df,
            alpha=SIGMOID_DECISION_ANCHOR_POWER,
        )
        fit_results_by_n[n_per_group] = fit_result
        curve_eval = evaluate_sigmoid_curve(
            fit_result=fit_result,
            alpha=SIGMOID_DECISION_ANCHOR_POWER,
            target_omega2=target_omega2,
            target_power=target_power,
        )
        qualifies = bool(
            curve_eval["curve_reaches_target_power"]
            and not np.isnan(float(curve_eval["fitted_power_at_target_omega2"]))
            and float(curve_eval["fitted_power_at_target_omega2"]) >= target_power
        )
        n_results_by_n[n_per_group] = {
            "n_per_group": n_per_group,
            "search_stage": search_stage,
            "qualification_method": "anchored_sigmoid_curvefit_at_0.05",
            "target_omega2_bracketed": np.nan,
            "bracket_low_omega2": np.nan,
            "bracket_high_omega2": np.nan,
            "bracket_low_power": np.nan,
            "bracket_high_power": np.nan,
            "curve_fit_status": curve_eval["curve_fit_status"],
            "curve_reaches_target_power": curve_eval["curve_reaches_target_power"],
            "estimated_power_at_target_omega2": np.nan,
            "fitted_power_at_target_omega2": curve_eval["fitted_power_at_target_omega2"],
            "required_omega2_for_target_power": curve_eval["required_omega2_for_target_power"],
            "target_omega2": target_omega2,
            "target_power": target_power,
            "qualifies": qualifies,
        }
        return qualifies

    coarse_candidates = sorted(
        set(
            [min_n, max_n]
            + list(range(((min_n + coarse_step - 1) // coarse_step) * coarse_step, max_n + 1, coarse_step))
        )
    )

    previous_coarse_n: Optional[int] = None
    coarse_hit_n: Optional[int] = None
    for n_per_group in coarse_candidates:
        qualifies = evaluate_one_n(n_per_group, "coarse")
        if qualifies:
            coarse_hit_n = n_per_group
            break
        previous_coarse_n = n_per_group

    if coarse_hit_n is not None:
        fine_start = min_n if previous_coarse_n is None else previous_coarse_n + 1
        for n_per_group in range(fine_start, coarse_hit_n + 1):
            window_ns = list(range(n_per_group, n_per_group + MINIMUM_N_STABILITY_WINDOW))
            if window_ns[-1] > max_n:
                break
            qualifies = all(
                evaluate_one_n(
                    candidate_n,
                    "fine" if candidate_n <= coarse_hit_n else "fine_stability",
                )
                for candidate_n in window_ns
            )
            if qualifies:
                minimum_n = n_per_group
                break

    selected_plot_ns: List[int] = []
    if minimum_n is not None:
        for n_per_group in [minimum_n - 1, minimum_n, minimum_n + 1]:
            if min_n <= n_per_group <= max_n:
                if n_per_group not in n_results_by_n:
                    evaluate_one_n(n_per_group, "plot_context")
                selected_plot_ns.append(n_per_group)
    else:
        selected_plot_ns = sorted(n_results_by_n)[-3:]

    curve_plot_dir.mkdir(parents=True, exist_ok=True)
    for existing_plot in curve_plot_dir.glob("power_curve_n*.png"):
        try:
            existing_plot.unlink()
        except OSError:
            pass
    for n_per_group in sorted(set(selected_plot_ns)):
        save_sigmoid_curve_plot(
            scenario_metrics_df=scenario_metrics_by_n[n_per_group],
            fit_result=fit_results_by_n[n_per_group],
            alpha=SIGMOID_DECISION_ANCHOR_POWER,
            target_omega2=target_omega2,
            target_power=target_power,
            n_per_group=n_per_group,
            out_path=curve_plot_dir / f"power_curve_n{n_per_group}.png",
        )

    metrics_df = pd.concat(scenario_metrics_frames, ignore_index=True) if scenario_metrics_frames else pd.DataFrame()
    power_df = pd.DataFrame([n_results_by_n[n] for n in sorted(n_results_by_n)])
    return minimum_n, power_df, metrics_df


def prepare_qza_inputs(
    table_path: Path,
    tree_path: Path,
    taxonomy_path: Path,
    out_dir: Path,
) -> tuple[Path, Path, Path]:
    table_qza = table_path if table_path.suffix.lower() == ".qza" else convert_csv_to_feature_table_qza(
        table_path, out_dir / "table.qza"
    )
    tree_qza = tree_path if tree_path.suffix.lower() == ".qza" else convert_nwk_to_rooted_tree_qza(
        tree_path, out_dir / "rooted-tree.qza"
    )
    tax_qza = taxonomy_path if taxonomy_path.suffix.lower() == ".qza" else convert_csv_to_taxonomy_qza(
        taxonomy_path, out_dir / "taxonomy.qza"
    )
    return table_qza, tree_qza, tax_qza


def summarize_gene_distance_metrics(
    dm,
    group_map: pd.Series,
    boot_number: int,
    alpha: float,
    n_jobs: int,
    random_seed: int,
    subject_group_vector: List[int],
    omega2_floor: Optional[float],
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
    permutations: int = 999,
) -> Dict[str, float]:
    load_core_runtime()
    if dm is None:
        return {"power": np.nan, "mean_boot_omega2": np.nan, "true_omega2": np.nan, "failed_bootstraps": 0}

    dm_df = dm.to_data_frame()
    common_samples = dm_df.index.intersection(group_map.index)
    aligned_dm = dm_df.loc[common_samples, common_samples]
    aligned_group_map = group_map.loc[common_samples]
    if aligned_dm.empty or aligned_group_map.empty:
        return {"power": np.nan, "mean_boot_omega2": np.nan, "true_omega2": np.nan, "failed_bootstraps": 0}

    true_omega2 = compute_omega2(aligned_dm, group_map=aligned_group_map)
    boot_seeds = make_bootstrap_seeds(random_seed, boot_number)
    boot_results = Parallel(n_jobs=n_jobs)(
        delayed(run_gene_bootstrap_permanova)(
            int(seed),
            dm,
            aligned_group_map,
            subject_group_vector,
            failure_log_path=failure_log_path,
            failure_context=failure_context,
            permutations=int(permutations),
        )
        for seed in boot_seeds
    )
    p_vals, o2_vals, failed_flags = zip(*boot_results) if boot_results else ([], [], [])
    p_valid = [p for p in p_vals if not np.isnan(p)]
    power = (np.array(p_valid) <= alpha).mean() if p_valid else np.nan
    mean_boot_omega2 = np.nanmean(o2_vals) if len(o2_vals) > 0 else np.nan
    failed_bootstraps = int(np.sum(np.asarray(failed_flags, dtype=int))) if len(failed_flags) else 0

    if omega2_floor is not None:
        if not np.isnan(true_omega2):
            true_omega2 = max(true_omega2, omega2_floor)
        if not np.isnan(mean_boot_omega2):
            mean_boot_omega2 = max(mean_boot_omega2, omega2_floor)

    return {
        "power": power,
        "mean_boot_omega2": mean_boot_omega2,
        "true_omega2": true_omega2,
        "failed_bootstraps": failed_bootstraps,
    }


def summarize_protein_distance_metrics(
    dm: pd.DataFrame,
    group_map: pd.Series,
    boot_number: int,
    alpha: float,
    n_jobs: int,
    random_seed: int,
    subject_group_vector: List[int],
    permutations: int,
    rng_seeds: Optional[np.ndarray] = None,
    failure_log_path: Optional[Path] = None,
    failure_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    load_core_runtime()
    if dm is None or dm.empty:
        return {"power": np.nan, "mean_boot_omega2": np.nan, "true_omega2": np.nan, "failed_bootstraps": 0}

    if rng_seeds is None:
        rng_seeds = make_bootstrap_seeds(random_seed, boot_number)
    metrics = summarize_rwct_distance_metrics(
        phylofunc_dm_true=dm,
        group_map=group_map,
        boot_number=boot_number,
        subject_group_vector=subject_group_vector,
        alpha=alpha,
        n_jobs=n_jobs,
        rng_seeds=rng_seeds,
        permutations=permutations,
        failure_log_path=failure_log_path,
        failure_context=failure_context,
    )
    return {
        "power": metrics["power_phylofunc"],
        "mean_boot_omega2": metrics["mean_boot_omega2_phylofunc"],
        "true_omega2": metrics["true_omega2_phylofunc"],
        "failed_bootstraps": metrics["failed_bootstraps"],
    }


def ensure_input_files_exist(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files: " + ", ".join(missing))


def resolve_sample_size_bounds(
    observed_full_sample_n: int,
    min_n: int,
    max_n: Optional[int],
) -> tuple[int, int]:
    resolved_min_n = max(min_n, 2)
    resolved_max_n = max_n if max_n is not None else max(observed_full_sample_n * 3, observed_full_sample_n + 20)
    if resolved_min_n > resolved_max_n:
        raise ValueError(
            "No feasible sample size range remains after filtering: "
            f"min_n={resolved_min_n}, max_n={resolved_max_n}."
        )
    return resolved_min_n, resolved_max_n


def print_minimum_n_result(
    workflow_label: str,
    sigma: float,
    nni_prob: float,
    minimum_n: Optional[int],
    min_n: int,
    max_n: int,
) -> None:
    prefix = f"[{workflow_label} | sigma={sigma:.2f}, nni={nni_prob:.2f}]"
    if minimum_n is not None:
        print(f"{prefix} Minimum sample size per group: {minimum_n}")
        return
    print(
        f"{prefix} No minimum sample size was found within the searched range "
        f"[{min_n}, {max_n}]. Try widening the sample-size range, for example by "
        f"using a smaller --min-n or a larger --max-n."
    )


def run_one_tree_level_gene(
    args: argparse.Namespace,
    tree_dir: Path,
    work_dir: Path,
    filtered_table_qza: Path,
    filtered_group_map: pd.Series,
    current_tree_qza: Path,
    tax_qza: Path,
    sigma: float,
    nni_prob: float,
) -> Dict[str, Any]:
    failure_log_path = tree_dir / "permanova_failures.jsonl"
    if failure_log_path.exists():
        failure_log_path.unlink()
    increase_levels = np.linspace(0.0, args.increase_max, num=max(2, args.increase_num))
    decrease_levels = np.linspace(0.0, args.decrease_min, num=max(2, args.decrease_num))
    effect_levels = np.unique(np.concatenate([increase_levels, decrease_levels]))
    effect_levels.sort()

    cache_dir = tree_dir / "cache_taxon"
    cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_idx = int(np.argmin(np.abs(effect_levels)))
    baseline_level = float(effect_levels[baseline_idx])
    baseline_seed = args.random_seed + baseline_idx
    baseline_cache_file = cache_dir / f"dm_effect_{baseline_level:+.4f}.pkl"
    baseline_metadata = build_taxon_distance_cache_metadata(
        args=args,
        filtered_group_map=filtered_group_map,
        sigma=sigma,
        nni_prob=nni_prob,
        effect_level=baseline_level,
        effect_seed=baseline_seed,
    )
    baseline_dm = None
    if not args.force_recompute:
        baseline_dm = load_taxon_distance_cache(baseline_cache_file, baseline_metadata)
    if baseline_dm is None:
        baseline_table_path = export_group_pool_feature_table(
            filtered_table_qza,
            filtered_group_map,
            work_dir,
            "table_effect_+0.000.qza",
        )
        try:
            baseline_dm = compute_gemelli_rpca_distance(
                table_qza_path=baseline_table_path,
                tree_qza_path=current_tree_qza,
                taxonomy_qza_path=tax_qza,
                out_dir=work_dir,
                qiime_env_name=args.qiime_env,
                output_stem="effect_+0.000",
            )
        finally:
            if baseline_table_path.exists():
                try:
                    baseline_table_path.unlink()
                except OSError:
                    pass
        save_taxon_distance_cache(baseline_cache_file, baseline_metadata, baseline_dm)
    if baseline_dm is None:
        raise RuntimeError("Failed to compute baseline Gemelli distance matrix for taxon workflow.")

    for i, level in enumerate(effect_levels):
        cache_file = cache_dir / f"dm_effect_{level:+.4f}.pkl"
        current_seed = args.random_seed + i
        cache_metadata = build_taxon_distance_cache_metadata(
            args=args,
            filtered_group_map=filtered_group_map,
            sigma=sigma,
            nni_prob=nni_prob,
            effect_level=float(level),
            effect_seed=current_seed,
        )
        if not args.force_recompute and load_taxon_distance_cache(cache_file, cache_metadata) is not None:
            continue
        if np.isclose(level, 0.0):
            dm = baseline_dm
        else:
            dm = modulate_taxon_gemelli_geometry(baseline_dm, filtered_group_map, float(level))
        save_taxon_distance_cache(cache_file, cache_metadata, dm)

    full_subject_group_vector = filtered_group_map.value_counts().sort_index().tolist()
    scenarios: List[Dict[str, Any]] = []
    for i, level in enumerate(effect_levels):
        cache_file = cache_dir / f"dm_effect_{level:+.4f}.pkl"
        current_seed = args.random_seed + i
        cache_metadata = build_taxon_distance_cache_metadata(
            args=args,
            filtered_group_map=filtered_group_map,
            sigma=sigma,
            nni_prob=nni_prob,
            effect_level=float(level),
            effect_seed=current_seed,
        )
        dm = load_taxon_distance_cache(cache_file, cache_metadata)
        metrics = summarize_gene_distance_metrics(
            dm=dm,
            group_map=filtered_group_map,
            boot_number=args.boot_number,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            random_seed=args.random_seed + 10_000 + i,
            subject_group_vector=full_subject_group_vector,
            omega2_floor=args.omega2_floor,
            failure_log_path=failure_log_path,
            failure_context={
                "workflow": "taxon",
                "stage": "full_sample_scenario",
                "sigma": sigma,
                "nni_prob": nni_prob,
                "scenario_index": i,
                "effect_level": float(level),
            },
            permutations=int(getattr(args, "permutations", 999)),
        )
        scenarios.append(
            {
                "mode": "enhancement" if level >= 0 else "dilution",
                "effect_level": float(level),
                "num_swaps": np.nan,
                "dm": dm,
                "true_omega2": metrics["true_omega2"],
                "mean_boot_omega2": metrics["mean_boot_omega2"],
                "power_full_sample": metrics["power"],
                "failed_bootstraps": metrics["failed_bootstraps"],
            }
        )

    scenario_search_df = pd.DataFrame(
        [
            {
                "mode": s["mode"],
                "effect_level": s["effect_level"],
                "num_swaps": s["num_swaps"],
                "true_omega2": s["true_omega2"],
                "mean_boot_omega2": s["mean_boot_omega2"],
                "power_full_sample": s["power_full_sample"],
                "failed_bootstraps": s["failed_bootstraps"],
            }
            for s in scenarios
        ]
    )

    valid_scenarios = [
        s for s in scenarios
        if s["dm"] is not None and not np.isnan(s["true_omega2"])
    ]
    if not valid_scenarios:
        summary = {
            "mode": "taxon",
            "sigma": sigma,
            "nni_prob": nni_prob,
            "minimum_n_per_group": None,
            "status": "no_valid_scenarios",
        }
        (tree_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "summary": summary,
            "scenario_search": scenario_search_df,
            "power_by_sample_size": pd.DataFrame(),
            "scenario_metrics_by_sample_size": pd.DataFrame(),
            "output_dir": tree_dir,
        }

    baseline_scenario = min(valid_scenarios, key=lambda s: abs(s["effect_level"]))
    selected_scenario = min(
        valid_scenarios,
        key=lambda s: abs(s["true_omega2"] - args.target_omega2),
    )

    observed_full_sample_n = int(filtered_group_map.value_counts().min())
    min_n, max_n = resolve_sample_size_bounds(observed_full_sample_n, args.min_n, args.max_n)
    fitted_curve_dir = tree_dir / "fitted_curves"
    fitted_curve_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_fn(n_per_group: int, _search_stage: str) -> pd.DataFrame:
        subject_group_vector = [n_per_group] * len(sorted(filtered_group_map.dropna().unique()))
        print(f"  Evaluating cached scenarios at n_per_group={n_per_group}")

        def _evaluate(idx: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
            metrics = summarize_gene_distance_metrics(
                dm=scenario["dm"],
                group_map=filtered_group_map,
                boot_number=args.boot_number,
                alpha=args.alpha,
                n_jobs=1,
                random_seed=args.random_seed + (n_per_group * 10_000) + idx,
                subject_group_vector=subject_group_vector,
                omega2_floor=args.omega2_floor,
                failure_log_path=failure_log_path,
                failure_context={
                    "workflow": "taxon",
                    "stage": _search_stage,
                    "sigma": sigma,
                    "nni_prob": nni_prob,
                    "n_per_group": n_per_group,
                    "scenario_index": idx,
                    "effect_level": float(scenario["effect_level"]),
                },
                permutations=int(getattr(args, "permutations", 999)),
            )
            return {
                "n_per_group": n_per_group,
                "scenario_index": idx,
                "mode": scenario["mode"],
                "effect_level": scenario["effect_level"],
                "num_swaps": np.nan,
                "power": metrics["power"],
                "true_omega2": metrics["true_omega2"],
                "mean_boot_omega2": metrics["mean_boot_omega2"],
                "failed_bootstraps": metrics["failed_bootstraps"],
            }

        rows = Parallel(n_jobs=args.n_jobs)(
            delayed(_evaluate)(idx, scenario)
            for idx, scenario in enumerate(valid_scenarios)
            if scenario["dm"] is not None
        )
        return pd.DataFrame(rows)

    minimum_n, power_df, scenario_metrics_by_n_df = estimate_minimum_sample_size(
        target_power=args.target_power,
        target_omega2=args.target_omega2,
        alpha=args.alpha,
        min_n=min_n,
        max_n=max_n,
        curve_plot_dir=fitted_curve_dir,
        evaluate_scenarios_for_n_fn=evaluate_fn,
    )
    power_df.to_csv(tree_dir / "power_by_sample_size.csv", index=False)
    scenario_metrics_by_n_df.to_csv(tree_dir / "scenario_metrics_by_sample_size.csv", index=False)
    print_minimum_n_result("taxon", sigma, nni_prob, minimum_n, min_n, max_n)

    summary = {
        "mode": "taxon",
        "sigma": sigma,
        "nni_prob": nni_prob,
        "modulation_parameterization": "log_centroid_scale",
        "estimation_method": "anchored_sigmoid_curvefit_only",
        "sigmoid_anchor_power": SIGMOID_DECISION_ANCHOR_POWER,
        "coarse_step": 10,
        "minimum_n_stability_window": MINIMUM_N_STABILITY_WINDOW,
        "target_power": args.target_power,
        "target_omega2": args.target_omega2,
        "alpha": args.alpha,
        "boot_number": args.boot_number,
        "baseline_true_omega2": float(baseline_scenario["true_omega2"]),
        "baseline_mean_boot_omega2": float(baseline_scenario["mean_boot_omega2"]),
        "nearest_full_sample_mode": selected_scenario["mode"],
        "nearest_full_sample_effect_level": float(selected_scenario["effect_level"]),
        "nearest_full_sample_num_swaps": None,
        "nearest_full_sample_true_omega2": float(selected_scenario["true_omega2"]),
        "nearest_full_sample_mean_boot_omega2": float(selected_scenario["mean_boot_omega2"]),
        "nearest_full_sample_power": float(selected_scenario["power_full_sample"]),
        "failed_bootstraps_full_sample": int(scenario_search_df["failed_bootstraps"].sum()) if "failed_bootstraps" in scenario_search_df else 0,
        "failed_bootstraps_sample_size_search": int(scenario_metrics_by_n_df["failed_bootstraps"].sum()) if "failed_bootstraps" in scenario_metrics_by_n_df else 0,
        "permanova_failure_log": str(failure_log_path) if failure_log_path.exists() else None,
        "minimum_n_per_group": minimum_n,
        "observed_full_sample_n_per_group": observed_full_sample_n,
        "sweep_min_n": min_n,
        "sweep_max_n": max_n,
        "increase_levels": increase_levels.tolist(),
        "decrease_levels": decrease_levels.tolist(),
        "resolved_increase_max": float(args.increase_max),
        "resolved_decrease_min": float(args.decrease_min),
        "resolved_centroid_scale_min": math.exp(float(args.decrease_min)),
        "resolved_centroid_scale_max": math.exp(float(args.increase_max)),
        "omega2_floor": args.omega2_floor,
    }
    (tree_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "scenario_search": scenario_search_df,
        "power_by_sample_size": power_df,
        "scenario_metrics_by_sample_size": scenario_metrics_by_n_df,
        "output_dir": tree_dir,
    }


def run_gene_workflow(args: argparse.Namespace) -> Dict[str, Any]:
    args.increase_max, args.decrease_min = resolve_taxon_modulation_bounds(
        args.increase_max,
        args.decrease_min,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_input_files_exist([args.table, args.tree, args.taxonomy, args.group])

    tree_noise_levels = parse_tree_noise_arg(args.tree_noise)
    with tempfile.TemporaryDirectory(prefix="phylopower_taxon_") as temp_root_str:
        temp_root = Path(temp_root_str)
        table_qza, tree_qza, tax_qza = prepare_qza_inputs(args.table, args.tree, args.taxonomy, temp_root)
        group_df = pd.read_csv(args.group, encoding="utf-8-sig")
        group_map = pd.Series(group_df.group_name.values, index=group_df.sample_id)
        filtered_table_qza, filtered_group_map = prepare_group_filtered_table(
            base_table_qza_path=table_qza,
            group_map=group_map,
            out_dir=temp_root,
            output_name="group_filtered_table.qza",
        )
        prepared_tree_qza, base_tree_node = prepare_gene_tree_artifact(tree_qza, temp_root)

        cross_tree_rows: List[Dict[str, Any]] = []
        last_result: Optional[Dict[str, Any]] = None
        for tree_idx, (sigma, nni_prob) in enumerate(tree_noise_levels):
            tree_label = f"sigma{sigma:.2f}_nni{nni_prob:.2f}"
            tree_dir = args.out / tree_label
            tree_dir.mkdir(parents=True, exist_ok=True)
            tree_work_dir = temp_root / tree_label
            tree_work_dir.mkdir(parents=True, exist_ok=True)
            perturbed_tree_qza = None
            current_tree_qza = prepared_tree_qza
            perturbed_tree_qza = materialize_perturbed_tree(
                base_tree=base_tree_node,
                sigma=sigma,
                nni_prob=nni_prob,
                random_seed=args.random_seed,
                out_dir=tree_work_dir,
                output_stem=tree_label,
                output_format="qza",
                support_threshold=args.nni_support_threshold,
            )
            if perturbed_tree_qza is not None:
                current_tree_qza = perturbed_tree_qza
            try:
                result = run_one_tree_level_gene(
                    args=args,
                    tree_dir=tree_dir,
                    work_dir=tree_work_dir,
                    filtered_table_qza=filtered_table_qza,
                    filtered_group_map=filtered_group_map,
                    current_tree_qza=current_tree_qza,
                    tax_qza=tax_qza,
                    sigma=sigma,
                    nni_prob=nni_prob,
                )
            finally:
                if perturbed_tree_qza and perturbed_tree_qza.exists():
                    try:
                        perturbed_tree_qza.unlink()
                    except OSError:
                        pass
            summary = dict(result["summary"])
            cross_tree_rows.append({
                "sigma": sigma,
                "nni_prob": nni_prob,
                "minimum_n_per_group": summary.get("minimum_n_per_group"),
                "status": summary.get("status", "ok"),
            })
            last_result = result

        cross_tree_df = pd.DataFrame(cross_tree_rows)
        cross_tree_df.to_csv(args.out / "minimum_n_summary.csv", index=False)
        if last_result is None:
            raise RuntimeError("No tree-noise runs were executed.")
        return {
            **last_result,
            "minimum_n_summary": cross_tree_df,
            "output_dir": args.out,
        }


def run_one_tree_level_pro(
    args: argparse.Namespace,
    tree_dir: Path,
    current_tree_path: str,
    long_df: pd.DataFrame,
    group_map: pd.Series,
    sigma: float,
    nni_prob: float,
) -> Dict[str, Any]:
    failure_log_path = tree_dir / "permanova_failures.jsonl"
    if failure_log_path.exists():
        failure_log_path.unlink()
    if args.increase_num <= 0 or args.decrease_num <= 0:
        raise ValueError("Increase and decrease level counts must both be positive integers.")

    increase_levels = np.linspace(0.0, args.increase_max, num=args.increase_num)
    decrease_levels = np.linspace(0.0, args.decrease_min, num=args.decrease_num)
    cache_dir = tree_dir / "cache_taxonfunction"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / (
        f"rwct_ridge{args.ridge:g}_"
        f"inclevels{args.increase_num}_declevels{args.decrease_num}_"
        f"inc{args.increase_max:g}_dec{args.decrease_min:g}.pkl"
    )
    cache_metadata = build_taxonfunction_cache_metadata(args, current_tree_path)
    cached_scenarios = None if args.force_recompute else load_taxonfunction_scenario_cache(cache_file, cache_metadata)
    if cached_scenarios is not None:
        increase_scenarios, decrease_scenarios = cached_scenarios
    else:
        increase_scenarios, decrease_scenarios = precompute_rwct_distance_scenarios(
            df_scenario=long_df,
            tree_path_str=current_tree_path,
            group_map=group_map,
            enhancement_gammas=increase_levels,
            dilution_gammas=decrease_levels,
            ridge=args.ridge,
            normalize_weights=not args.no_normalize_weights,
        )
        save_taxonfunction_scenario_cache(cache_file, cache_metadata, increase_scenarios, decrease_scenarios)

    full_subject_group_vector = group_map.value_counts().sort_index().tolist()
    scenarios: List[Dict[str, Any]] = []
    increase_boot_seed_matrix = make_bootstrap_seed_matrix(
        args.random_seed, len(increase_scenarios), args.boot_number, 1
    )
    for i, scenario in enumerate(increase_scenarios):
        metrics = summarize_protein_distance_metrics(
            dm=scenario["dm"],
            group_map=group_map,
            boot_number=args.boot_number,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            random_seed=args.random_seed,
            subject_group_vector=full_subject_group_vector,
            permutations=args.permutations,
            rng_seeds=increase_boot_seed_matrix[i],
            failure_log_path=failure_log_path,
            failure_context={
                "workflow": "taxon-function",
                "stage": "full_sample_scenario",
                "sigma": sigma,
                "nni_prob": nni_prob,
                "mode": "enhancement",
                "scenario_index": i,
                "gamma": float(scenario["gamma"]),
            },
        )
        scenarios.append(
            {
                "mode": "enhancement",
                "mode_index": i,
                "gamma": float(scenario["gamma"]),
                "effect_level": float(scenario["gamma"]),
                "num_swaps": np.nan,
                "dm": scenario["dm"],
                "true_omega2": metrics["true_omega2"],
                "mean_boot_omega2": metrics["mean_boot_omega2"],
                "power_full_sample": metrics["power"],
                "failed_bootstraps": metrics["failed_bootstraps"],
            }
        )
    decrease_boot_seed_matrix = make_bootstrap_seed_matrix(
        args.random_seed, len(decrease_scenarios), args.boot_number, 2
    )
    for i, scenario in enumerate(decrease_scenarios):
        metrics = summarize_protein_distance_metrics(
            dm=scenario["dm"],
            group_map=group_map,
            boot_number=args.boot_number,
            alpha=args.alpha,
            n_jobs=args.n_jobs,
            random_seed=args.random_seed,
            subject_group_vector=full_subject_group_vector,
            permutations=args.permutations,
            rng_seeds=decrease_boot_seed_matrix[i],
            failure_log_path=failure_log_path,
            failure_context={
                "workflow": "taxon-function",
                "stage": "full_sample_scenario",
                "sigma": sigma,
                "nni_prob": nni_prob,
                "mode": "dilution",
                "scenario_index": i,
                "gamma": float(scenario["gamma"]),
            },
        )
        scenarios.append(
            {
                "mode": "dilution",
                "mode_index": i,
                "gamma": float(scenario["gamma"]),
                "effect_level": float(scenario["gamma"]),
                "num_swaps": np.nan,
                "dm": scenario["dm"],
                "true_omega2": metrics["true_omega2"],
                "mean_boot_omega2": metrics["mean_boot_omega2"],
                "power_full_sample": metrics["power"],
                "failed_bootstraps": metrics["failed_bootstraps"],
            }
        )

    scenario_search_df = pd.DataFrame(
        [
            {
                "mode": s["mode"],
                "gamma": s["gamma"],
                "effect_level": s["effect_level"],
                "num_swaps": s["num_swaps"],
                "true_omega2": s["true_omega2"],
                "mean_boot_omega2": s["mean_boot_omega2"],
                "power_full_sample": s["power_full_sample"],
                "failed_bootstraps": s["failed_bootstraps"],
            }
            for s in scenarios
        ]
    )

    valid_scenarios = [
        s for s in scenarios
        if s["dm"] is not None and not np.isnan(s["true_omega2"])
    ]
    if not valid_scenarios:
        summary = {
            "mode": "taxon-function",
            "sigma": sigma,
            "nni_prob": nni_prob,
            "minimum_n_per_group": None,
            "status": "no_valid_scenarios",
        }
        (tree_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "summary": summary,
            "scenario_search": scenario_search_df,
            "power_by_sample_size": pd.DataFrame(),
            "scenario_metrics_by_sample_size": pd.DataFrame(),
            "output_dir": tree_dir,
        }

    baseline_scenario = min(valid_scenarios, key=lambda s: abs(s["gamma"]))
    selected_scenario = min(
        valid_scenarios,
        key=lambda s: abs(s["true_omega2"] - args.target_omega2),
    )

    observed_full_sample_n = int(group_map.value_counts().min())
    min_n, max_n = resolve_sample_size_bounds(observed_full_sample_n, args.min_n, args.max_n)
    fitted_curve_dir = tree_dir / "fitted_curves"
    fitted_curve_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_fn(n_per_group: int, _search_stage: str) -> pd.DataFrame:
        subject_group_vector = [n_per_group] * len(sorted(group_map.dropna().unique()))
        print(f"  Evaluating cached scenarios at n_per_group={n_per_group}")
        increase_boot_seed_matrix = make_bootstrap_seed_matrix(
            args.random_seed, len(increase_scenarios), args.boot_number, 10, n_per_group
        )
        decrease_boot_seed_matrix = make_bootstrap_seed_matrix(
            args.random_seed, len(decrease_scenarios), args.boot_number, 20, n_per_group
        )

        def _evaluate(idx: int, scenario: Dict[str, Any]) -> Dict[str, Any]:
            if scenario["mode"] == "enhancement":
                scenario_rng_seeds = increase_boot_seed_matrix[scenario["mode_index"]]
            else:
                scenario_rng_seeds = decrease_boot_seed_matrix[scenario["mode_index"]]
            metrics = summarize_protein_distance_metrics(
                dm=scenario["dm"],
                group_map=group_map,
                boot_number=args.boot_number,
                alpha=args.alpha,
                n_jobs=1,
                random_seed=args.random_seed + (n_per_group * 10_000),
                subject_group_vector=subject_group_vector,
                permutations=args.permutations,
                rng_seeds=scenario_rng_seeds,
                failure_log_path=failure_log_path,
                failure_context={
                    "workflow": "taxon-function",
                    "stage": _search_stage,
                    "sigma": sigma,
                    "nni_prob": nni_prob,
                    "n_per_group": n_per_group,
                    "mode": scenario["mode"],
                    "scenario_index": idx,
                    "gamma": float(scenario["gamma"]),
                },
            )
            return {
                "n_per_group": n_per_group,
                "scenario_index": idx,
                "mode": scenario["mode"],
                "gamma": scenario["gamma"],
                "effect_level": scenario["effect_level"],
                "num_swaps": np.nan,
                "power": metrics["power"],
                "true_omega2": metrics["true_omega2"],
                "mean_boot_omega2": metrics["mean_boot_omega2"],
                "failed_bootstraps": metrics["failed_bootstraps"],
            }

        rows = Parallel(n_jobs=args.n_jobs)(
            delayed(_evaluate)(idx, scenario)
            for idx, scenario in enumerate(valid_scenarios)
            if scenario["dm"] is not None
        )
        return pd.DataFrame(rows)

    minimum_n, power_df, scenario_metrics_by_n_df = estimate_minimum_sample_size(
        target_power=args.target_power,
        target_omega2=args.target_omega2,
        alpha=args.alpha,
        min_n=min_n,
        max_n=max_n,
        curve_plot_dir=fitted_curve_dir,
        evaluate_scenarios_for_n_fn=evaluate_fn,
    )
    power_df.to_csv(tree_dir / "power_by_sample_size.csv", index=False)
    scenario_metrics_by_n_df.to_csv(tree_dir / "scenario_metrics_by_sample_size.csv", index=False)
    print_minimum_n_result("taxon-function", sigma, nni_prob, minimum_n, min_n, max_n)

    summary = {
        "mode": "taxon-function",
        "sigma": sigma,
        "nni_prob": nni_prob,
        "modulation_parameterization": "rwct_gamma",
        "estimation_method": "anchored_sigmoid_curvefit_only",
        "sigmoid_anchor_power": SIGMOID_DECISION_ANCHOR_POWER,
        "coarse_step": 10,
        "minimum_n_stability_window": MINIMUM_N_STABILITY_WINDOW,
        "target_power": args.target_power,
        "target_omega2": args.target_omega2,
        "alpha": args.alpha,
        "boot_number": args.boot_number,
        "baseline_true_omega2": float(baseline_scenario["true_omega2"]),
        "baseline_mean_boot_omega2": float(baseline_scenario["mean_boot_omega2"]),
        "nearest_full_sample_mode": selected_scenario["mode"],
        "nearest_full_sample_gamma": float(selected_scenario["gamma"]),
        "nearest_full_sample_effect_level": float(selected_scenario["effect_level"]),
        "nearest_full_sample_num_swaps": None,
        "nearest_full_sample_true_omega2": float(selected_scenario["true_omega2"]),
        "nearest_full_sample_mean_boot_omega2": float(selected_scenario["mean_boot_omega2"]),
        "nearest_full_sample_power": float(selected_scenario["power_full_sample"]),
        "failed_bootstraps_full_sample": int(scenario_search_df["failed_bootstraps"].sum()) if "failed_bootstraps" in scenario_search_df else 0,
        "failed_bootstraps_sample_size_search": int(scenario_metrics_by_n_df["failed_bootstraps"].sum()) if "failed_bootstraps" in scenario_metrics_by_n_df else 0,
        "permanova_failure_log": str(failure_log_path) if failure_log_path.exists() else None,
        "taxonfunction_cache_metadata": cache_metadata,
        "minimum_n_per_group": minimum_n,
        "observed_full_sample_n_per_group": observed_full_sample_n,
        "sweep_min_n": min_n,
        "sweep_max_n": max_n,
        "increase_levels": increase_levels.tolist(),
        "decrease_levels": decrease_levels.tolist(),
        "resolved_increase_max": float(args.increase_max),
        "resolved_decrease_min": float(args.decrease_min),
        "ridge": args.ridge,
        "normalize_weights": not args.no_normalize_weights,
        "permutations": args.permutations,
        "auto_gamma_cap": None if args.auto_gamma_cap is None else float(args.auto_gamma_cap),
        "auto_direction_max": None if args.auto_direction_max is None else float(args.auto_direction_max),
        "rwct_max_log_shift": float(args.rwct_max_log_shift),
    }
    (tree_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "summary": summary,
        "scenario_search": scenario_search_df,
        "power_by_sample_size": power_df,
        "scenario_metrics_by_sample_size": scenario_metrics_by_n_df,
        "output_dir": tree_dir,
    }


def run_protein_rwct_workflow(args: argparse.Namespace) -> Dict[str, Any]:
    args.out.mkdir(parents=True, exist_ok=True)
    ensure_input_files_exist([args.table, args.tree, args.group])
    if detect_input_data_type(args.table) == "taxon":
        raise ValueError(f"{args.table} looks like taxon data; taxon-function expects a Taxon-Function table.")

    tree_noise_levels = parse_tree_noise_arg(args.tree_noise)
    group_df = pd.read_csv(args.group, encoding="utf-8")
    group_map = pd.Series(group_df.group_name.values, index=group_df.sample_id, name="group")
    long_df, group_map = load_protein_long_table(args.table, group_map)
    if args.increase_max is None or args.decrease_min is None:
        args.increase_max, args.decrease_min, auto_info = resolve_rwct_modulation_bounds(
            df=long_df,
            group_map=group_map,
            ridge=args.ridge,
            normalize_weights=not args.no_normalize_weights,
            increase_max=args.increase_max,
            decrease_min=args.decrease_min,
            max_abs_log_shift=args.rwct_max_log_shift,
        )
        args.auto_gamma_cap = auto_info["auto_gamma_cap"]
        args.auto_direction_max = auto_info["auto_direction_max"]
    else:
        args.increase_max = float(args.increase_max)
        args.decrease_min = float(args.decrease_min)
        args.auto_gamma_cap = None
        args.auto_direction_max = None

    from skbio import TreeNode

    base_tree_node = TreeNode.read(str(args.tree), format="newick")

    cross_tree_rows: List[Dict[str, Any]] = []
    last_result: Optional[Dict[str, Any]] = None
    for tree_idx, (sigma, nni_prob) in enumerate(tree_noise_levels):
        tree_label = f"sigma{sigma:.2f}_nni{nni_prob:.2f}"
        tree_dir = args.out / tree_label
        tree_dir.mkdir(parents=True, exist_ok=True)
        current_tree_path = str(args.tree)
        perturbed_tree_path = materialize_perturbed_tree(
            base_tree=base_tree_node,
            sigma=sigma,
            nni_prob=nni_prob,
            random_seed=args.random_seed,
            out_dir=args.out,
            output_stem=tree_label,
            output_format="newick",
            support_threshold=args.nni_support_threshold,
        )
        if perturbed_tree_path is not None:
            current_tree_path = str(perturbed_tree_path)
        try:
            result = run_one_tree_level_pro(
                args=args,
                tree_dir=tree_dir,
                current_tree_path=current_tree_path,
                long_df=long_df,
                group_map=group_map,
                sigma=sigma,
                nni_prob=nni_prob,
            )
        finally:
            if perturbed_tree_path and perturbed_tree_path.exists():
                try:
                    perturbed_tree_path.unlink()
                except OSError:
                    pass
        summary = dict(result["summary"])
        cross_tree_rows.append({
            "sigma": sigma,
            "nni_prob": nni_prob,
            "minimum_n_per_group": summary.get("minimum_n_per_group"),
            "status": summary.get("status", "ok"),
        })
        last_result = result

    cross_tree_df = pd.DataFrame(cross_tree_rows)
    cross_tree_df.to_csv(args.out / "minimum_n_summary.csv", index=False)
    if last_result is None:
        raise RuntimeError("No tree-noise runs were executed.")
    return {
        **last_result,
        "minimum_n_summary": cross_tree_df,
        "output_dir": args.out,
    }


def compute_taxon(
    table: Union[str, Path],
    tree: Union[str, Path],
    taxonomy: Union[str, Path],
    group: Union[str, Path],
    target_power: float,
    target_omega2: float,
    out: Union[str, Path] = "./gene_min_sample_size_output",
    qiime_env: str = "qiime2-metagenome-2024.10",
    alpha: float = 0.05,
    boot_number: int = DEFAULT_BOOT_NUMBER,
    n_jobs: int = -1,
    random_seed: int = 42,
    min_n: int = 2,
    max_n: Optional[int] = None,
    tree_noise: Sequence[Tuple[float, float]] = ((0.0, 0.0),),
    increase_max: Optional[float] = DEFAULT_TAXON_INCREASE_MAX,
    increase_num: int = 30,
    decrease_min: Optional[float] = DEFAULT_TAXON_DECREASE_MIN,
    decrease_num: int = 75,
    omega2_floor: float = 0.0,
    nni_support_threshold: Optional[float] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Programmatic API for the specialized `taxon` workflow.

    This is the library equivalent of:

    `phylopower taxon ...`

    Returns the parsed summary plus the key result DataFrames that are also
    written to disk in ``out``.
    """
    _require_gene_runtime()
    load_core_runtime()
    args = argparse.Namespace(
        mode="taxon",
        workflow="taxon",
        table=Path(table),
        tree=Path(tree),
        taxonomy=Path(taxonomy),
        group=Path(group),
        target_power=target_power,
        target_omega2=target_omega2,
        out=Path(out),
        qiime_env=qiime_env,
        alpha=alpha,
        boot_number=boot_number,
        n_jobs=n_jobs,
        random_seed=random_seed,
        min_n=min_n,
        max_n=max_n,
        tree_noise=[f"{sigma},{nni}" for sigma, nni in tree_noise],
        increase_max=increase_max,
        increase_num=increase_num,
        decrease_min=decrease_min,
        decrease_num=decrease_num,
        omega2_floor=omega2_floor,
        nni_support_threshold=nni_support_threshold,
        force_recompute=force_recompute,
    )
    return run_gene_workflow(args)


def compute_taxon_function(
    table: Union[str, Path],
    tree: Union[str, Path],
    group: Union[str, Path],
    target_power: float,
    target_omega2: float,
    out: Union[str, Path] = "./pro_min_sample_size_output",
    alpha: float = 0.05,
    boot_number: int = DEFAULT_BOOT_NUMBER,
    n_jobs: int = -1,
    random_seed: int = 42,
    min_n: int = 2,
    max_n: Optional[int] = None,
    tree_noise: Sequence[Tuple[float, float]] = ((0.0, 0.0),),
    permutations: int = 999,
    ridge: float = 1.0,
    increase_max: Optional[float] = DEFAULT_RWCT_INCREASE_MAX,
    increase_num: int = DEFAULT_RWCT_LEVELS,
    decrease_min: Optional[float] = DEFAULT_RWCT_DECREASE_MIN,
    decrease_num: int = DEFAULT_RWCT_LEVELS,
    normalize_weights: bool = True,
    rwct_max_log_shift: float = RWCT_AUTO_MAX_LOG_SHIFT,
    nni_support_threshold: Optional[float] = None,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """Programmatic API for the specialized `taxon-function` workflow.

    This is the library equivalent of:

    `phylopower taxon-function ...`

    Returns the parsed summary plus the key result DataFrames that are also
    written to disk in ``out``.
    """
    _require_pro_rwct_runtime()
    load_core_runtime()
    args = argparse.Namespace(
        mode="taxon-function",
        workflow="taxon-function",
        table=Path(table),
        tree=Path(tree),
        group=Path(group),
        target_power=target_power,
        target_omega2=target_omega2,
        out=Path(out),
        alpha=alpha,
        boot_number=boot_number,
        n_jobs=n_jobs,
        random_seed=random_seed,
        min_n=min_n,
        max_n=max_n,
        tree_noise=[f"{sigma},{nni}" for sigma, nni in tree_noise],
        permutations=permutations,
        ridge=ridge,
        increase_max=increase_max,
        increase_num=increase_num,
        decrease_min=decrease_min,
        decrease_num=decrease_num,
        rwct_max_log_shift=rwct_max_log_shift,
        no_normalize_weights=not normalize_weights,
        nni_support_threshold=nni_support_threshold,
        force_recompute=force_recompute,
    )
    return run_protein_rwct_workflow(args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = create_argument_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    validate_mode_runtime(args)
    load_core_runtime()
    args.pipeline(args)


if __name__ == "__main__":
    main()
