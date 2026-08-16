"""PhyloPower's manuscript-aligned public API.

``paper_core`` is the authoritative, self-contained implementation.  Its
internal workflow modules are embedded in that file, so it has no runtime
dependency on a separate ``core.py`` or other project source files.
"""

from ._data import DATAGENE_DIR, DATAPRO_DIR, demo_path
from .paper_core import (
    compute_gene_min_sample_size,
    compute_protein_min_sample_size,
)

# Descriptive aliases retained for callers that used the original package
# terminology. Both aliases resolve to the authoritative paper_core
# implementations.
compute_taxon = compute_gene_min_sample_size
compute_taxon_function = compute_protein_min_sample_size

__all__ = [
    "compute_gene_min_sample_size",
    "compute_protein_min_sample_size",
    "compute_taxon",
    "compute_taxon_function",
    "demo_path",
    "DATAGENE_DIR",
    "DATAPRO_DIR",
]

__version__ = "0.1.0"
__author__ = "Phylopower authors"
__license__ = "MIT"
