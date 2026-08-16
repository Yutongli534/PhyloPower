"""Shared publication figure style for all PhyloPower result figures (fig1, fig2, workflow plots).
Import and call apply_style() once at the top of any plotting script, and use the shared colors so
every figure in the paper has the same fonts, spines, palette, and reference-line conventions."""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- palette ---------------------------------------------------------------
REAL = "#22313f"        # real / observed data (dark slate)
SYN = "#e07b39"         # synthetic / generated (warm orange)
GROUP = ["#2f6db0", "#b03a3a"]   # two-group categorical (blue / muted red)
ACCENT = "#cc3333"      # the key reference line (e.g. target omega^2): dashed red
NEUTRAL = "#7f8c8d"     # secondary reference (e.g. target power, nominal alpha): gray dotted
BAND = "#e9ecef"        # shaded acceptance band (e.g. Bradley) fill
GRID = "#d9d9d9"
CMAP_SEQ = "viridis"    # sequential gradient (pilot size / eval_n)

# modality accents (colourblind-safe), if a figure needs to distinguish modalities
MODALITY = {"protein": "#1b9e77", "gene": "#7570b3"}


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#444444",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.6,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "axes.titlepad": 6,
    })


def bradley_band(ax, alpha: float = 0.05, label: str = "Bradley robust band") -> None:
    """Shade the Bradley (1978) robustness band [0.5a, 1.5a] and mark nominal alpha."""
    ax.axhspan(0.5 * alpha, 1.5 * alpha, color=BAND, zorder=0, label=label)
    ax.axhline(alpha, color=ACCENT, ls="--", lw=1.0, zorder=1)


def seq_colors(values):
    """Map a sorted list of values (pilot sizes / eval_n) to consistent sequential colors."""
    import numpy as np
    vals = sorted(values)
    norm = plt.Normalize(min(vals), max(vals))
    cmap = plt.get_cmap(CMAP_SEQ)
    return {v: cmap(norm(v)) for v in vals}, norm, cmap
