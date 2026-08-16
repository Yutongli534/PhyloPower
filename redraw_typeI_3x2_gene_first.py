#!/usr/bin/env python3
"""Redraw Type-I calibration as 3 rows x 2 columns.

Rows are analysis questions; columns are data types.
This lets the manuscript cite panels in order: (a,b), (c,d), (e,f),
with metagenomics before metaproteomics in every pair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from redraw_typeI_2x3_gene_first import (
    FIGDATA,
    GENE,
    PROTEIN,
    _load_cache,
    plot_alpha,
    plot_qq,
    plot_size,
)

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=FIGDATA / "fig1_null_pvalues.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "fig2_typeI_3x2_gene_first")
    ap.add_argument("--eval-ns", default="6,10,17,30,50")
    ap.add_argument("--qq-n", type=int, default=30)
    args = ap.parse_args()

    eval_ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    gene, protein = _load_cache(args.cache, eval_ns)
    args.out.mkdir(parents=True, exist_ok=True)

    # Wider, more manuscript-friendly 3 x 2 layout.
    # The earlier square canvas made the figure read too tall and narrow once
    # inserted into the manuscript. This keeps the non-jumping panel order while
    # giving each panel a wider aspect ratio.
    fig, axes = plt.subplots(3, 2, figsize=(16.6, 9.4), squeeze=False)
    plot_qq(axes[0, 0], gene, args.qq_n, "a", "Metagenomics", GENE)
    plot_qq(axes[0, 1], protein, args.qq_n, "b", "Metaproteomics", PROTEIN)
    plot_alpha(axes[1, 0], gene, "c", "Metagenomics", GENE, show_legend=True)
    plot_alpha(axes[1, 1], protein, "d", "Metaproteomics", PROTEIN, show_legend=True)
    summary = {
        "metagenomics": plot_size(axes[2, 0], gene, "e", "Metagenomics", GENE),
        "metaproteomics": plot_size(axes[2, 1], protein, "f", "Metaproteomics", PROTEIN),
    }

    fig.tight_layout(rect=(0.035, 0.0, 0.995, 1.0), w_pad=3.0, h_pad=0.05)
    fig.savefig(args.out / "fig2_typeI_3x2_gene_first_wider_flat.png", dpi=320, bbox_inches="tight")
    fig.savefig(args.out / "fig2_typeI_3x2_gene_first_wider_flat.pdf", bbox_inches="tight")
    plt.close(fig)
    (args.out / "fig2_typeI_3x2_gene_first_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(args.out / "fig2_typeI_3x2_gene_first_wider_flat.png")


if __name__ == "__main__":
    main()
