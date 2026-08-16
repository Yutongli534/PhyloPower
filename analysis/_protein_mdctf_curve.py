#!/usr/bin/env python3
"""MDC-TF-lite protein synthetic pool sensitivity curve.

MDC-TF-lite keeps the Taxon-Function detection mask template-based, so each
synthetic sample inherits a real sample's edge count, taxon degree, function
degree, connectance, and TF pairing structure. Only the abundance layer is
modeled: pooled/own template residuals plus EB-shrunk taxon, function, and edge
group effects.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pcam_gen as P
from logistic_fit import fit_logistic, logistic_curve
from phylopower import core
from semisynthetic_power import summarize_distance_metrics_with_replacement


def _positive_floor(abund: np.ndarray, cols: np.ndarray) -> np.ndarray:
    floor = np.zeros(abund.shape[0], dtype=float)
    for i in range(abund.shape[0]):
        pos = abund[i, cols][abund[i, cols] > 0]
        floor[i] = float(pos.min() / 100.0) if len(pos) else 1e-9
    return floor


def _eb_weight(delta: np.ndarray, noise: np.ndarray) -> np.ndarray:
    signal = np.asarray(delta, dtype=float) ** 2
    noise = np.maximum(np.asarray(noise, dtype=float), 1e-8)
    return np.clip(signal / (signal + noise), 0.0, 1.0)


def _level_means(values: np.ndarray, level_ids: np.ndarray, n_levels: int) -> np.ndarray:
    out = np.zeros(n_levels, dtype=float)
    for k in range(n_levels):
        rows = level_ids == k
        if np.any(rows):
            out[k] = float(np.nanmean(values[rows]))
    return out


def _resolve_auto_edge_fraction(value: float | str) -> float:
    if str(value).strip().lower() == "auto":
        return 1.0
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("edge_fraction must be in [0, 1] or 'auto'.")
    return resolved


def _fit_mdctf(d: dict, edge_fraction: float | str = "auto") -> dict:
    groups = list(d["groups"])
    if len(groups) != 2:
        raise ValueError("MDC-TF-lite prototype currently expects exactly two groups.")

    A = d["abund"]
    L = d["L"]
    all_idx = np.concatenate([d["gs"][g] for g in groups])
    grand = L[:, all_idx].mean(axis=1)
    pooled_var = L[:, all_idx].var(axis=1, ddof=1) if len(all_idx) > 1 else np.zeros(A.shape[0])
    floor = _positive_floor(A, all_idx)

    taxon_ids = d["uid"].astype(int)
    n_taxa = int(taxon_ids.max()) + 1
    functions = d["meta"]["Function"].astype(str).to_numpy()
    _, function_ids = np.unique(functions, return_inverse=True)
    n_funcs = int(function_ids.max()) + 1

    group_mean = {g: L[:, d["gs"][g]].mean(axis=1) for g in groups}
    model = {
        "groups": groups,
        "taxon_ids": taxon_ids,
        "function_ids": function_ids,
        "grand": grand,
        "floor": floor,
        "all_idx": all_idx,
        "blocks": {},
    }

    for g in groups:
        idx = d["gs"][g]
        n_g = max(len(idx), 1)
        gmean = group_mean[g]
        delta = gmean - grand
        gvar = L[:, idx].var(axis=1, ddof=1) if len(idx) > 1 else np.zeros(A.shape[0])
        edge_w = _eb_weight(delta, gvar / n_g + pooled_var / max(len(all_idx), 1))

        tax_raw = _level_means(delta, taxon_ids, n_taxa)
        tax_noise = _level_means(gvar / n_g + pooled_var / max(len(all_idx), 1), taxon_ids, n_taxa)
        tax_w = _eb_weight(tax_raw, tax_noise)

        func_raw = _level_means(delta, function_ids, n_funcs)
        func_noise = _level_means(gvar / n_g + pooled_var / max(len(all_idx), 1), function_ids, n_funcs)
        func_w = _eb_weight(func_raw, func_noise)

        tax_effect = tax_w[taxon_ids] * tax_raw[taxon_ids]
        func_effect = func_w[function_ids] * func_raw[function_ids]
        structured = 0.5 * tax_effect + 0.5 * func_effect
        edge_resid = delta - structured
        effect = structured + _resolve_auto_edge_fraction(edge_fraction) * edge_w * edge_resid

        residuals = L[:, idx] - gmean[:, None]
        model["blocks"][g] = {
            "effect": effect,
            "mean": gmean,
            "residuals": residuals,
            "totals": A[:, idx].sum(axis=0),
        }
    model["pooled_residuals"] = np.column_stack([
        L[:, d["gs"][g]] - group_mean[g][:, None] for g in groups
    ])
    model["pooled_totals"] = A[:, all_idx].sum(axis=0)
    return model


def mdctf_pool(
    d: dict,
    M: int,
    seed: int,
    strength: float,
    *,
    edge_fraction: float | str = "auto",
    balanced_templates: bool = True,
    residual_mode: str = "random",
) -> tuple[pd.DataFrame, pd.Series]:
    model = _fit_mdctf(d, edge_fraction=edge_fraction)
    if residual_mode not in {"random", "template"}:
        raise ValueError("residual_mode must be 'random' or 'template'")
    rng = np.random.default_rng(int(seed))
    A = d["abund"]
    L = d["L"]
    E = A.shape[0]
    all_idx = model["all_idx"]
    ab = np.zeros((E, M * len(model["groups"])), dtype=float)
    labels: list[str] = []
    names: list[str] = []
    col = 0
    # `strength` has two roles. The template/support selection probability must
    # stay within [0, 1], but the abundance-level group effect can be scaled
    # above 1 for effect-enhancement figures and sensitivity sweeps. Keeping
    # those two knobs separate preserves taxon-function topology while allowing
    # a genuine stronger abundance contrast.
    effect_scale = float(strength)
    template_prob = float(np.clip(strength, 0.0, 1.0))
    template_orders: dict[str, np.ndarray] = {}
    pooled_repeats = int(np.ceil((M * len(model["groups"])) / max(len(all_idx), 1))) + 1
    template_orders["__pooled__"] = np.concatenate([
        rng.permutation(all_idx) for _ in range(pooled_repeats)
    ])
    for gg in model["groups"]:
        own = d["gs"][gg]
        reps = int(np.ceil(M / max(len(own), 1))) + 1
        template_orders[str(gg)] = np.concatenate([rng.permutation(own) for _ in range(reps)])
    own_pos = {str(g): {int(v): i for i, v in enumerate(d["gs"][g])} for g in model["groups"]}
    pooled_pos = {int(v): i for i, v in enumerate(all_idx)}
    pooled_cursor = 0

    for g in model["groups"]:
        own_idx = d["gs"][g]
        block = model["blocks"][g]
        own_cursor = 0
        for j in range(M):
            use_own = rng.random() < template_prob
            if use_own:
                if balanced_templates:
                    template_col = int(template_orders[str(g)][own_cursor])
                    own_cursor += 1
                else:
                    template_col = int(rng.choice(own_idx))
                resid_pool = block["residuals"]
                total_pool = block["totals"]
            else:
                if balanced_templates:
                    template_col = int(template_orders["__pooled__"][pooled_cursor])
                    pooled_cursor += 1
                else:
                    template_col = int(rng.choice(all_idx))
                resid_pool = model["pooled_residuals"]
                total_pool = model["pooled_totals"]

            mask = A[:, template_col] > 0
            if residual_mode == "template":
                if use_own:
                    rpos = own_pos[str(g)].get(template_col, 0)
                else:
                    rpos = pooled_pos.get(template_col, 0)
                resid = resid_pool[:, int(rpos)] if resid_pool.shape[1] else np.zeros(E)
            else:
                resid = resid_pool[:, int(rng.integers(resid_pool.shape[1]))] if resid_pool.shape[1] else np.zeros(E)
            latent = model["grand"] + effect_scale * block["effect"] + resid
            positive = np.maximum(np.expm1(latent), model["floor"])
            new = np.where(mask, positive, 0.0)

            if residual_mode == "template":
                total = float(A[:, template_col].sum())
            else:
                total = float(rng.choice(total_pool)) if len(total_pool) else float(new.sum())
            if new.sum() > 0 and total > 0:
                new *= total / new.sum()

            sample_id = f"md_{str(g)[:2]}_{j:04d}"
            ab[:, col] = new
            labels.append(str(g))
            names.append(sample_id)
            col += 1

    return pd.DataFrame(ab, columns=names), pd.Series(labels, index=names, name="group")


def _structure_rows(d: dict, table: pd.DataFrame, sgm: pd.Series, pilot_n: int, strength: float) -> pd.DataFrame:
    taxon_ids = d["uid"]
    funcs = d["meta"]["Function"].astype(str).to_numpy()
    rows = []
    for g in d["groups"]:
        real = d["abund"][:, d["gs"][g]] > 0
        syn = table.loc[:, sgm[sgm == g].index].to_numpy() > 0
        for kind, mat in [("real", real), ("synthetic", syn)]:
            edge_counts = mat.sum(axis=0)
            tax_deg = [len(np.unique(taxon_ids[mat[:, i]])) for i in range(mat.shape[1])]
            fun_deg = [len(np.unique(funcs[mat[:, i]])) for i in range(mat.shape[1])]
            rows.append({
                "pilot_n": pilot_n,
                "strength": strength,
                "group": str(g),
                "kind": kind,
                "edge_mean": float(np.mean(edge_counts)),
                "taxon_degree_mean": float(np.mean(tax_deg)),
                "function_degree_mean": float(np.mean(fun_deg)),
                "connectance": float(np.mean(edge_counts) / max(mat.shape[0], 1)),
            })
    return pd.DataFrame(rows)


def _run_point(d: dict, pilot_n: int, strength: float, M: int, seed: int, eval_n: int, boot: int, perms: int, edge_fraction: float) -> tuple[dict, pd.DataFrame]:
    tab, sgm = mdctf_pool(d, M, seed, strength, edge_fraction=edge_fraction)
    dm = P.recompute_distance(d, tab)
    omega = max(0.0, float(core.compute_omega2(dm, sgm)))
    metrics = summarize_distance_metrics_with_replacement(
        dm=dm,
        group_map=sgm,
        boot_number=boot,
        alpha=0.05,
        n_jobs=1,
        random_seed=seed + 31,
        n_per_group=eval_n,
        permutations=perms,
        omega2_floor=0.0,
    )
    row = {
        "method": "mdc_tf_lite",
        "pilot_n": int(pilot_n),
        "strength": float(strength),
        "eval_n": int(eval_n),
        "true_omega2": omega,
        "power": float(metrics["power"]),
        "edge_fraction": float(edge_fraction),
    }
    return row, _structure_rows(d, tab, sgm, pilot_n, strength)


def _fit_curve(sub: pd.DataFrame, x: np.ndarray) -> np.ndarray | None:
    dat = sub[["true_omega2", "power"]].dropna()
    if len(dat) < 4:
        return None
    fit = fit_logistic(dat, floor=0.05)
    params = fit.get("params")
    if not params:
        return None
    return logistic_curve(x, params["k"], params["x0"], floor=0.05)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pilots", default="7,10,19")
    p.add_argument("--strengths", default="0,0.15,0.3,0.45,0.6,0.75,0.9,1.0")
    p.add_argument("--eval-n", type=int, default=17)
    p.add_argument("--pool-M", type=int, default=300)
    p.add_argument("--boot", type=int, default=100)
    p.add_argument("--perms", type=int, default=99)
    p.add_argument("--edge-fraction", type=str, default="auto")
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--out", type=Path, default=Path("data/figdata/protein_mdctf_curve"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    args.edge_fraction = _resolve_auto_edge_fraction(args.edge_fraction)

    base = P.load_modality("protein")
    pilots = [int(x) for x in args.pilots.split(",") if x.strip()]
    strengths = [float(x) for x in args.strengths.split(",") if x.strip()]
    rows: list[dict] = []
    diag_frames: list[pd.DataFrame] = []

    for pn in pilots:
        pilot = P.pilot_view(base, pn, args.seed + pn * 1009)
        for si, strength in enumerate(strengths):
            seed = args.seed + pn * 10000 + si * 1291
            row, diag = _run_point(
                pilot, pn, strength, args.pool_M, seed, args.eval_n, args.boot, args.perms, args.edge_fraction
            )
            rows.append(row)
            diag_frames.append(diag)
            print(
                f"mdc_tf pilot={pn:2d} s={strength:.2f} "
                f"omega2={row['true_omega2']:.4f} power={row['power']:.3f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    diag = pd.concat(diag_frames, ignore_index=True) if diag_frames else pd.DataFrame()
    df.to_csv(args.out / "raw.csv", index=False)
    diag.to_csv(args.out / "structure_diag.csv", index=False)

    colors = {7: "#c2410c", 10: "#2563eb", 19: "#0f766e"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    xmax = max(0.08, float(df["true_omega2"].max()) * 1.08)
    x = np.linspace(0.0, xmax, 300)
    for pn in pilots:
        sub = df[df["pilot_n"] == pn].sort_values("true_omega2")
        c = colors.get(pn)
        axes[0].scatter(sub["true_omega2"], sub["power"], s=32, color=c, alpha=0.86, label=f"pilot {pn}")
        y = _fit_curve(sub, x)
        if y is not None:
            axes[0].plot(x, y, color=c, lw=2.0)
    axes[0].axhline(0.8, color="#666666", ls=":", lw=1.0)
    axes[0].axhline(0.05, color="#999999", ls="--", lw=0.8)
    axes[0].set_xlim(0.0, xmax)
    axes[0].set_ylim(-0.02, 1.03)
    axes[0].set_xlabel("true omega^2")
    axes[0].set_ylabel("power")
    axes[0].set_title("MDC-TF-lite curve")
    axes[0].grid(alpha=0.18)
    axes[0].legend(frameon=False, fontsize=8)

    if not diag.empty:
        b = diag[np.isclose(diag["strength"], 1.0)]
        ratios = []
        for (pn, g), ss in b.groupby(["pilot_n", "group"]):
            real = ss[ss["kind"] == "real"].iloc[0]
            syn = ss[ss["kind"] == "synthetic"].iloc[0]
            ratios.append({
                "pilot_n": int(pn),
                "edge": syn["edge_mean"] / max(real["edge_mean"], 1e-9),
                "taxon": syn["taxon_degree_mean"] / max(real["taxon_degree_mean"], 1e-9),
                "function": syn["function_degree_mean"] / max(real["function_degree_mean"], 1e-9),
                "connectance": syn["connectance"] / max(real["connectance"], 1e-9),
            })
        r = pd.DataFrame(ratios)
        metrics = ["edge", "taxon", "function", "connectance"]
        xs = np.arange(len(metrics))
        width = 0.22
        for i, pn in enumerate(pilots):
            vals = r[r["pilot_n"] == pn][metrics].mean(axis=0)
            axes[1].bar(xs + (i - 1) * width, vals, width=width, color=colors.get(pn), alpha=0.86, label=f"pilot {pn}")
        axes[1].axhline(1.0, color="#555555", ls=":", lw=1.0)
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels(metrics)
        axes[1].set_ylim(0.0, 1.35)
        axes[1].set_ylabel("synthetic / pilot")
        axes[1].set_title("baseline topology ratio")
        axes[1].grid(axis="y", alpha=0.18)

    fig.suptitle(f"Protein MDC-TF-lite @ eval_n={args.eval_n}, M={args.pool_M}, boot={args.boot}", fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out / "comparison.png", dpi=180)
    print(f"saved {args.out / 'comparison.png'}", flush=True)


if __name__ == "__main__":
    main()
