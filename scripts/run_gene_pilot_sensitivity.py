#!/usr/bin/env python3
"""Launch the PhyloPower gene pilot-sensitivity workflow under QIIME 2 2024.10.

The validated ``qiime2-metagenome-2024.10`` environment ships scikit-bio
0.6.0, whose ``skbio.stats.distance.permanova`` has no ``seed`` keyword and
permutes through the global ``numpy.random`` state, while
``phylopower.core.compute_permanova_p_value_with_status`` calls it with
``seed=``. This launcher wraps ``permanova`` so that a given seed reseeds the
global RNG for the duration of the call (the previous RNG state is saved and
restored), making bootstrap PERMANOVA p-values deterministic, then delegates
to ``run_paper_core_pilot_sensitivity.main`` unchanged.

Usage (must run with the QIIME 2 environment Python):

    /opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python \
        scripts/run_gene_pilot_sensitivity.py [arguments forwarded verbatim]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import skbio.stats.distance as _skbio_distance
from skbio.stats.distance import permanova as _permanova_unseeded


def _permanova_seeded(distance_matrix, grouping, column=None, permutations=999, seed=None):
    """scikit-bio 0.6.0-compatible permanova with a ``seed`` keyword."""
    if seed is None:
        return _permanova_unseeded(
            distance_matrix, grouping, column=column, permutations=permutations
        )
    state = np.random.get_state()
    try:
        np.random.seed(int(seed) % (2**32))
        return _permanova_unseeded(
            distance_matrix, grouping, column=column, permutations=permutations
        )
    finally:
        np.random.set_state(state)


_skbio_distance.permanova = _permanova_seeded

import run_paper_core_pilot_sensitivity as _runner

if __name__ == "__main__":
    _runner.main()
