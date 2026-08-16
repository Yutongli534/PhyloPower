#!/usr/bin/env python3
"""Supplementary tree-error sample-size curves.

Run a Figure-5-style sample-size family under the original tree and one
representative tree-estimation-error setting. This reuses the ordination power
model used by the manuscript tree-error sensitivity figure, but fixes the
synthetic pool size so the eval-n sweep remains tractable.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "figures"))  # shared figstyle

from phylopower import core  # noqa: E402  (import first: installs the embedded-module finder)
import figstyle  # noqa: E402
import gene_power_workflow as gpw  # noqa: E402
import protein_power_workflow as ppw  # noqa: E402
from fig5 import (  # noqa: E402
    PROTEIN_GROUP,
    PROTEIN_TABLE,
    PROTEIN_TREE,
    _pilot_distance,
)
from semisynthetic_power import (  # noqa: E402
    ID_COLS,
    _build_ordination_model,
    _ordination_pool_dm,
    _ordination_pool_group_map,
    _ordination_scales_by_power_uniform,
    _read_group_map,
    _read_protein_long_table,
    _read_taxon_feature_table,
    _subsample_group_map,
    summarize_distance_metrics_with_replacement,
)


TREE_SETTINGS = [
    ("original", 0.0, 0.0),
    ("tree_error_0.50_0.50", 0.5, 0.5),
]

MODALITIES = {
    "gene": {
        "label": "Metagenomics (Gemelli)",
        "pilot_n": 10,
        "eval_ns": [4, 7, 10, 30, 50, 80],
        "tree": core.DATAGENE_DIR / "rooted-tree.nwk",
        "xmax": 0.60,
    },
    "protein": {
        "label": "Metaproteomics (PhyloFunc)",
        "pilot_n": 17,
        "eval_ns": [7, 10, 17, 30, 50, 80],
        "tree": PROTEIN_TREE,
        "xmax": 0.18,
    },
}


def make_args(tree: Path, sigma: float, nni: float, args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        alpha=0.05,
        target_power=0.80,
        pool_size_per_group=args.pool_m,
        boot_number=args.boot,
        permutations=args.permutations,
        n_jobs=1,
        omega2_floor=0.0,
        effect_grid="power-uniform",
        engine="ordination",
        ordination_enhance_max=3.0,
        omega_calibrated_max_scale=20.0,
        adaptive_reference_n=17.0,
        power_preview_boot_number=args.preview_boot,
        power_preview_permutations=args.preview_permutations,
        omega_grid_candidates=args.omega_grid_candidates,
        fit_power_min=0.15,
        increase_num=args.increase_num,
        decrease_num=args.decrease_num,
        center_mode="omega-calibrated",
        cov_estimator="ledoit-wolf",
        embed_dim=None,
        pool_cov=False,
        cov_eb_pool=False,
        pool_dist="student-t",
        pool_df="auto",
        protein_transform="vst",
        lowrank_rank=5,
        agg_clades=300,
        tree=tree,
        tree_jitter_sigma=float(sigma),
        tree_nni_prob=float(nni),
        tree_support_threshold=None,
        random_seed=args.seed,
        use_phylogeny=True,
        qiime_env=args.qiime_env,
        taxonomy=core.DATAGENE_DIR / "taxonomy.csv",
    )


def load_modality(modality: str, pilot_n: int, seed: int):
    if modality == "protein":
        group_map = _read_group_map(PROTEIN_GROUP)
        long_df, aligned_group_map = _read_protein_long_table(PROTEIN_TABLE, group_map)
        pilot_group = _subsample_group_map(aligned_group_map, pilot_n, seed)
        return long_df[ID_COLS + list(pilot_group.index)].copy(), pilot_group

    group_map = _read_group_map(core.DATAGENE_DIR / "group.csv")
    table, aligned_group_map = _read_taxon_feature_table(core.DATAGENE_DIR / "table.csv", group_map)
    pilot_group = _subsample_group_map(aligned_group_map, pilot_n, seed)
    return table[list(pilot_group.index)].copy(), pilot_group


def curve_fixed_pool(
    modality: str,
    model: dict,
    run_args: SimpleNamespace,
    eval_n: int,
    seed: int,
) -> pd.DataFrame:
    pool_group_map = _ordination_pool_group_map(model, run_args.pool_size_per_group)
    scales = _ordination_scales_by_power_uniform(
        run_args,
        eval_n,
        model,
        run_args.pool_size_per_group,
        pool_group_map,
        seed=seed,
        workflow="taxon" if modality == "gene" else "taxon-function",
    )
    rows = []
    for i, scale in enumerate(scales):
        dm = _ordination_pool_dm(
            model,
            float(scale),
            run_args.pool_size_per_group,
            seed + 9000 + i * 89,
        )
        metrics = summarize_distance_metrics_with_replacement(
            dm=dm,
            group_map=pool_group_map,
            boot_number=run_args.boot_number,
            alpha=run_args.alpha,
            n_jobs=1,
            random_seed=seed + i,
            n_per_group=int(eval_n),
            permutations=run_args.permutations,
            omega2_floor=run_args.omega2_floor,
        )
        rows.append(
            {
                "scale": float(scale),
                "true_omega2": float(metrics["true_omega2"]),
                "power": float(metrics["power"]),
                "mean_boot_omega2": float(metrics["mean_boot_omega2"]),
                "failed_bootstraps": int(metrics["failed_bootstraps"]),
                "mode": "dilution" if scale < 1.0 else "enhancement",
            }
        )
    return pd.DataFrame(rows)


def run_one(modality: str, setting: tuple[str, float, float], args: argparse.Namespace) -> pd.DataFrame:
    setting_label, sigma, nni = setting
    meta = MODALITIES[modality]
    pilot_seed = args.seed + int(meta["pilot_n"]) * 1009
    data, pilot_group = load_modality(modality, int(meta["pilot_n"]), pilot_seed)
    run_args = make_args(meta["tree"], sigma, nni, args)
    print(
        f"[tree-size] {modality} {setting_label}: pilot_n={meta['pilot_n']} "
        f"sigma={sigma:.2f} p_NNI={nni:.2f}",
        flush=True,
    )
    dm = _pilot_distance(modality, data, run_args, f"supp_{modality}_{setting_label}")
    common = dm.index.intersection(pilot_group.index)
    pilot_group = pilot_group.loc[common]
    dm = dm.loc[common, common]
    model = _build_ordination_model(
        dm,
        pilot_group,
        center_mode="omega-calibrated",
        cov_estimator="ledoit-wolf",
        pool_dist="student-t",
        pool_df="auto",
    )
    frames = []
    for eval_n in meta["eval_ns"]:
        df = curve_fixed_pool(
            modality,
            model,
            run_args,
            int(eval_n),
            seed=args.seed + int(eval_n) * 17 + int(1000 * sigma) + int(1000 * nni),
        )
        df.insert(0, "eval_n", int(eval_n))
        frames.append(df)
        n_transition = int(((df["power"] >= 0.15) & (df["power"] <= 0.95)).sum())
        print(
            f"[tree-size] {modality} {setting_label} eval_n={eval_n}: "
            f"{len(df)} points, transition={n_transition}",
            flush=True,
        )
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "setting", setting_label)
    out.insert(1, "sigma", float(sigma))
    out.insert(2, "p_nni", float(nni))
    out.insert(3, "modality", modality)
    out["pilot_n"] = int(meta["pilot_n"])
    out["pool_m_per_group"] = int(args.pool_m)
    out["boot"] = int(args.boot)
    out["permutations"] = int(args.permutations)
    return out


def plot_curves(df: pd.DataFrame, out_dir: Path) -> None:
    figstyle.apply_style()
    eval_order = {
        "gene": MODALITIES["gene"]["eval_ns"],
        "protein": MODALITIES["protein"]["eval_ns"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), squeeze=False)
    for row, modality in enumerate(["gene", "protein"]):
        meta = MODALITIES[modality]
        color_map, _, _ = figstyle.seq_colors(eval_order[modality])
        for col, (setting_label, sigma, nni) in enumerate(TREE_SETTINGS):
            ax = axes[row, col]
            sub_all = df[(df["modality"].eq(modality)) & (df["setting"].eq(setting_label))]
            for eval_n in eval_order[modality]:
                sub = sub_all[sub_all["eval_n"].eq(eval_n)].sort_values("true_omega2")
                if sub.empty:
                    continue
                ax.scatter(
                    sub["true_omega2"],
                    sub["power"],
                    s=16,
                    color=color_map[eval_n],
                    alpha=0.35,
                    linewidths=0,
                )
                sm = sub[["true_omega2", "power"]].dropna().sort_values("true_omega2")
                if len(sm) >= 2:
                    sm["power_mono"] = np.maximum.accumulate(sm["power"].to_numpy(float))
                    ax.plot(
                        sm["true_omega2"],
                        sm["power_mono"],
                        color=color_map[eval_n],
                        lw=2.1,
                        label=f"n={eval_n}",
                    )
            ax.axhline(0.8, color="#cc3333", ls="--", lw=1.1, alpha=0.7)
            ax.axhline(0.05, color="#666666", ls=":", lw=0.9, alpha=0.6)
            data_xmax = float(sub_all["true_omega2"].dropna().max()) if sub_all["true_omega2"].notna().any() else meta["xmax"]
            ax.set_xlim(0, max(meta["xmax"], data_xmax * 1.05))
            ax.set_ylim(-0.03, 1.03)
            ax.grid(True, color="#e5e7eb", lw=0.65, alpha=0.72)
            ax.set_xlabel(r"realized $\omega^2$")
            ax.set_ylabel("Power")
            ax.set_title(
                f"{meta['label']}\n"
                + (r"original tree" if setting_label == "original" else r"$\sigma=0.50$, $p_{\mathrm{NNI}}=0.50$"),
                fontweight="bold",
            )
            ax.legend(frameon=False, loc="lower right", fontsize=9)
    fig.tight_layout(w_pad=2.0, h_pad=1.4)
    fig.savefig(out_dir / "tree_error_sample_size_curves.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / "tree_error_sample_size_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("tree_error_sample_size_curves"))
    parser.add_argument("--qiime-env", default="qiime2-metagenome-2024.10")
    parser.add_argument("--pool-m", type=int, default=300)
    parser.add_argument("--boot", type=int, default=80)
    parser.add_argument("--permutations", type=int, default=99)
    parser.add_argument("--preview-boot", type=int, default=20)
    parser.add_argument("--preview-permutations", type=int, default=49)
    parser.add_argument("--omega-grid-candidates", type=int, default=30)
    parser.add_argument("--increase-num", type=int, default=8)
    parser.add_argument("--decrease-num", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260616)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    frames = []
    for modality in ["gene", "protein"]:
        for setting in TREE_SETTINGS:
            frames.append(run_one(modality, setting, args))
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(args.out / "tree_error_sample_size_curves.csv", index=False)
    plot_curves(df, args.out)
    print(f"[tree-size] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
