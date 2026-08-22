# Reproducibility

## Authoritative implementation

The scientific workflow used for the manuscript is
`phylopower/cli.py`. The installed CLI, module CLI, package-level API,
and generated standalone runner all resolve to this implementation.

`phylopower/cli.py` is a generated, self-contained module that embeds
the lower-level core and all other project-local runtime modules. It does not
require an external `phylopower/core.py`. The readable build sources
`phylopower/_cli_source.py` and `phylopower/_core_source.py` are
tracked in the source tree under `phylopower/` and are used only for
maintenance and deterministic regeneration (`python
scripts/build_standalone.py` rewrites the runners byte-identically for a
fixed source tree).

## Runtime environments

The protein workflow requires Python 3.9 or newer and the dependencies in
`pyproject.toml`, including NumPy, pandas (2.x and 3.x, the latter verified
against pandas 3.0 Copy-on-Write semantics), SciPy, scikit-bio, Biopython,
PhyloFunc, joblib, Matplotlib, and psutil.

The gene workflow additionally requires QIIME 2, Gemelli, and biom-format.
The development environment was `qiime2-metagenome-2024.10`. It must be
activated before launching the gene workflow because PCAM calls the Gemelli
Python API in-process; `--qiime-env` alone does not replace activation of a
compatible environment.

All stochastic public functions accept `random_seed`; the CLI default is
`20260614`. Every random stream — pool generation, bootstrap resampling, the
PERMANOVA permutations inside each bootstrap replicate, and tree
perturbation — is deterministically derived from this seed, so two runs with
the same seed and inputs produce bit-identical results (this is covered by
`tests/test_reproducibility.py`). Production analyses should record the full
command, input-file checksums, software environment, and output
`summary.json`.

## Standalone integrity

The root `phylopower_cli.py` and `phylopower/cli.py` embed the complete
project-local dependency closure. Run:

```bash
python phylopower_cli.py --standalone-info
```

to print SHA-256 hashes for every embedded source module. Rebuild it with:

```bash
python scripts/build_standalone.py
```

The generator is deterministic for a fixed source tree.

## Validation performed for this release

The standalone runner was copied outside the repository and tested with an
empty `PYTHONPATH`.

- Its CLI loaded when it was the only local Python file.
- The protein demo completed a reduced MDC-TF-MC/PhyloFunc smoke run.
- The gene demo completed a reduced PCAM/Gemelli smoke run in the QIIME 2
  environment.

The reduced validation parameters are intended only to test execution, not to
produce scientifically stable sample-size estimates. The CLI defaults
(`--boot-number 200`, `--permutations 199`; scikit-bio's internal `+1`
correction makes the p-value denominator 200) are quick settings.
Publication analyses should use substantially larger bootstrap and
permutation counts (for example `--boot-number 500 --permutations 999`).

## Monte Carlo uncertainty of power estimates

Every scenario power value is a binomial proportion over `--boot-number`
bootstrap replicates. `scenario_metrics_by_sample_size.csv` therefore reports,
alongside `power`, its Monte Carlo standard error (`power_mcse` =
sqrt(p(1-p)/B)) and a 95% Wilson interval (`power_wilson95_lower`,
`power_wilson95_upper`). At the default B=200 the worst-case standard error
(at power = 0.5) is about 0.035; at the conventional target power of 0.8 the
95% Wilson interval is roughly ±0.056 wide. If the interval width matters for
a design decision, increase `--boot-number`.

## Generator fidelity boundary (protein workflow)

The MDC-TF-MC generator is designed to preserve the taxon–function network
structure of the pilot data — edge counts, degree distributions, connectance,
and zero unsupported taxon–function edges — together with the
distance/effect structure that drives power (within/between group distances
and realized omega-squared). Feature-level marginal distributions (per-feature
mean and variance) are recovered less tightly than by the PCAM gene
generator; in independent checks the median real-split-calibrated discrepancy
for feature marginals was ~1.4 (pass threshold 1.0), while all network and
distance/effect fidelity metrics passed. Interpret the protein workflow's
power estimates as conditioned on the preserved distance structure rather
than on exact feature marginals.

## Bundled data

The release includes the input files used as convenient execution examples.
Before public deposition, verify the original dataset accessions, required
citations, and redistribution permissions, and add those details to the
Zenodo record.
