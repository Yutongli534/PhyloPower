#!/usr/bin/env python3
"""Convert curatedMetagenomicData QinJ_2012 (T2D vs control) to PhyloPower gene inputs.

Thin wrapper around ``prepare_cmd_gene.py`` with the dataset fixed; extra CLI
arguments (e.g. ``--per-group``) are forwarded unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_cmd_gene import main

if __name__ == "__main__":
    main(["--dataset", "QinJ_2012", *sys.argv[1:]])
