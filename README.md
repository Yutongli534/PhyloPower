# PhyloPower

PhyloPower estimates the minimum balanced per-group sample size required for
community-level meta-omics studies at a target statistical power and realized
PERMANOVA effect size (ω²).

The manuscript-aligned implementation is `phylopower.paper_core`. It provides
two raw-pool workflows:

| Command | Synthetic generator | Distance |
|---|---|---|
| `gene` | PCAM | Gemelli phylogenetic RPCA |
| `protein` | MDC-TF-MC | PhyloFunc |

A step-by-step walkthrough of both workflows is provided in the tutorial
notebook [`tutorials/tutorial.ipynb`](tutorials/tutorial.ipynb).

Both the root `paper_core.py` and `phylopower/paper_core.py` are generated,
self-contained runners. They embed the lower-level core and all other
project-local runtime modules, so neither file imports an external
`phylopower/core.py`.

## Standalone use

The root [`paper_core.py`](paper_core.py) and package
[`phylopower/paper_core.py`](phylopower/paper_core.py) each contain all
project-local Python code required by both workflows. Either file can be
copied to an empty directory and executed without `core.py` or any other
PhyloPower source file at runtime.
`paper_core_standalone.py` is an identical, explicitly named copy.

It still requires the scientific Python packages and input data. Put
`datagene/` and/or `datapro/` next to the standalone file, set
`PHYLOPOWER_DATA_DIR`, or provide every input path explicitly.

For the protein workflow, install the standalone runtime dependencies with:

```bash
python -m pip install \
  "numpy>=1.22" "pandas>=1.5" "scipy>=1.9" "scikit-bio>=0.6.3" \
  "joblib>=1.2" "matplotlib>=3.7" "tqdm>=4.64" "psutil>=5.9" \
  phylofunc "biopython>=1.80"
```

Both pandas 2.x and 3.x are supported; the workflows are verified against
pandas 3.0 Copy-on-Write semantics (`DataFrame.to_numpy()` read-only views).

The gene workflow must be launched from the compatible QIIME 2 environment
that contains Gemelli and `biom-format`; selecting that environment only with
`--qiime-env` is not a substitute for activating it because PCAM calls the
Gemelli Python API in-process.

```bash
python paper_core.py --help

python paper_core.py gene \
  --table /data/table.csv \
  --tree /data/rooted-tree.nwk \
  --taxonomy /data/taxonomy.csv \
  --group /data/group.csv \
  --target-power 0.80 \
  --out gene_result

python paper_core.py protein \
  --table /data/protein_taxon_function.csv \
  --tree /data/rooted-tree.nwk \
  --group /data/group.csv \
  --target-power 0.80 \
  --out protein_result
```

If `--target-omega2` is omitted, the observed ω² of the resolved pilot is used
and recorded in `summary.json`.

The effect grid is widened automatically when the target ω² exceeds the
simulated range (`--pcam-scale-extend-max` for `gene`, `--mdctf-strength-max`
for `protein`). If the target still falls outside the realized ω² range, the
fitted power at the target would rest on curve extrapolation only: a warning
is printed, the extrapolated fit is not allowed to qualify, no minimum n is
reported, and `summary.json` records `"target_omega2_bracketed": false`
(together with `low_omega_support_warning`). Widen the grid with the flags
above and rerun in that case.

### Monte Carlo settings

The default (quick) settings are `--boot-number 200` bootstrap iterations per
effect level and `--permutations 199` permutations per PERMANOVA; scikit-bio
applies the `+1` correction internally, so p-value denominators are 200.
Publication-grade estimates should use substantially larger counts — see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). All stochastic steps, including
the PERMANOVA permutation streams and tree perturbation, are deterministically
derived from `--random-seed`, so a fixed seed reproduces a run bit-for-bit.

To inspect the exact embedded-module hashes:

```bash
python paper_core.py --standalone-info
```

Regenerate the standalone file from the readable source modules with:

```bash
python scripts/build_standalone.py
```

The maintainable build sources are `phylopower/_paper_core_source.py` and
`phylopower/_core_source.py`; both are tracked in the source tree under
`phylopower/`. They are used only to regenerate the self-contained runners;
the generated `phylopower/paper_core.py` does not read them at runtime.
Regeneration via `python scripts/build_standalone.py` is deterministic: for a
fixed source tree it rewrites the three runners byte-identically.

## Package installation

From this source directory:

```bash
python -m pip install .
phylopower --help
```

The installed `phylopower` command and `python -m phylopower` both invoke
`phylopower.paper_core`.

The protein workflow requires the dependencies declared in
`pyproject.toml`. The gene workflow additionally requires a QIIME 2
environment containing compatible versions of Gemelli, `biom-format`, and
scikit-bio. The manuscript development environment was named
`qiime2-metagenome-2024.10`. Activate that environment before installing and
running the gene workflow:

```bash
conda activate qiime2-metagenome-2024.10
python -m pip install .
phylopower gene --target-power 0.80 --out gene_result
```

## Python API

```python
from phylopower import (
    compute_gene_min_sample_size,
    compute_protein_min_sample_size,
    demo_path,
)

result = compute_protein_min_sample_size(
    table=demo_path("datapro", "protein_taxon_function_cleaned.csv"),
    tree=demo_path("datapro", "rooted-tree.nwk"),
    group=demo_path("datapro", "group.csv"),
    target_power=0.80,
    target_omega2=0.05,
    out="protein_result",
)
```

The compatibility aliases `compute_taxon` and `compute_taxon_function` point
to these manuscript-aligned functions; they do not call the older workflows
in `core.py`.

## Input files

### Gene workflow

- feature table: CSV with a `Taxon` column and one column per sample;
- rooted phylogenetic tree: Newick;
- taxonomy: CSV containing `Feature ID` and `Taxon`;
- group map: CSV containing `sample_id` and `group_name`.

### Protein workflow

- Taxon–Function table: CSV with `Taxon`, `Function`, and one column per
  sample;
- rooted phylogenetic tree: Newick;
- group map: CSV containing `sample_id` and `group_name`.

The current raw-pool implementation expects exactly two groups.

## Outputs

Each run writes:

- `summary.json`;
- `power_by_sample_size.csv`;
- `scenario_metrics_by_sample_size.csv`;
- `sample_size_decision.png`.

The Python API returns the same summary and two result tables in a dictionary.

## Reproducibility and tests

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). Quick checks:

```bash
python scripts/build_standalone.py
python -m pytest -q
```

The bundled datasets are convenient execution examples. Before a public
Zenodo release, the depositors should verify and document their original
accessions, citations, and redistribution terms.

## Citation and license

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). PhyloPower
is distributed under the MIT License; see [`LICENSE`](LICENSE).
