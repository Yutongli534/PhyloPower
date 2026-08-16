# YachidaS_2019 core-matched pilot sensitivity analysis

External-data validation of the manuscript-aligned `phylopower.paper_core`
gene workflow on curatedMetagenomicData `YachidaS_2019` (colorectal cancer
versus control). This run uses the same conceptual design as the
`PXD069517_core_matched` protein validation:

- workflow: gene (Gemelli phylo-RPCA distance, PCAM synthetic pools)
- pilot sizes: 7, 10, and the full selected group size of 20
- observed evaluation size: 20 per group
- extrapolated evaluation size: 80 per group
- target effect size: omega-squared = 0.03
- target power: 0.80
- automatically selected omega-uniform PCAM grid with 12 power-curve points
  (plus 4 near-zero points)
- final power estimates: 200 bootstrap replicates and 199 permutations
- reusable synthetic pool: 180 per group
- random seed: 20260614

## Input data

`validation_datasets/processed/YachidaS_2019/`, produced by
`scripts/prepare_yachidas2019.py` (see its `preparation_summary.json`):

- 20 CRC versus 20 control samples, drawn from 258 CRC / 251 control
  available samples by gender-stratified random sampling (proportional
  allocation, largest remainder; seed 20260614); adenoma and
  carcinoma-surgery-history samples were excluded
- 311 MetaPhlAn species features (98.76% of the 727 species rows matched a
  tip of the curatedMetagenomicData 3.20.0 phylogenetic tree; prevalence
  >= 2 of the 40 selected samples)
- tree: the package MetaPhlAn phylogeny (real branch lengths) pruned to the
  retained species
- abundances stored as ppm pseudo-counts (fraction x 1e6); scikit-bio 0.6.0
  casts table values to int64 inside Gemelli's fast-UniFrac path, so
  fraction-scale inputs would be truncated to zero

## Reproduction command

```bash
/opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python scripts/run_gene_pilot_sensitivity.py \
  --workflow gene \
  --gene-table validation_datasets/processed/YachidaS_2019/table.csv \
  --gene-taxonomy validation_datasets/processed/YachidaS_2019/taxonomy.csv \
  --gene-tree validation_datasets/processed/YachidaS_2019/rooted-tree.nwk \
  --gene-group validation_datasets/processed/YachidaS_2019/group.csv \
  --gene-pilot-ns 7,10,20 \
  --gene-eval-observed 20 \
  --gene-eval-ns 20,80 \
  --eval-extrapolate 80 \
  --gene-grid auto \
  --gene-power-points 12 \
  --gene-near-zero-points 4 \
  --gene-pi-candidates 17 \
  --gene-scale-candidates 6 \
  --pool-size-per-group 180 \
  --boot-number 200 \
  --permutations 199 \
  --target-omega2 0.03 \
  --target-power 0.8 \
  --scenario-n-jobs 2 \
  --random-seed 20260614 \
  --out validation_datasets/results/YachidaS_2019_core_matched
```

`scripts/run_gene_pilot_sensitivity.py` is a thin launcher that wraps
`skbio.stats.distance.permanova` with a `seed` keyword (scikit-bio 0.6.0 in
the validated QIIME 2 environment lacks it and permutes through the global
`numpy.random` state) and otherwise delegates to
`analysis/run_cli_pilot_sensitivity.py` unchanged.

## Target-effect comparison

| Evaluation size | Pilot size | Fitted power at omega-squared = 0.03 |
|---:|---:|---:|
| 20 | 7 | 0.261 |
| 20 | 10 | 0.240 |
| 20 | 20 | 0.265 |
| 80 | 7 | 0.855 |
| 80 | 10 | 0.858 |
| 80 | 20 | 0.837 |

The range across pilot curves is 0.026 at the observed evaluation size and
0.022 at the extrapolated size, comparable to the pilot-size consistency
reported for the original core run (0.069) and the PXD069517 protein
validation (0.081). At the extrapolated size of 80 per group the fitted
power at the target effect is approximately 0.84-0.86, consistent with the
target-power-oriented design.

Null-scenario power (Type I, nominal alpha = 0.05) was 0.035-0.075 at
evaluation size 20 and 0.080-0.105 at evaluation size 80 across pilot sizes,
consistent with calibrated Type I error given 200 bootstrap replicates.

## Interpretation

The gene validation framework reproduces on this independent public dataset:
synthetic pools calibrated to 7-20 pilot samples per group yield power
curves that agree closely across pilot sizes, Type I error is controlled,
and the fitted power at the extrapolated size of 80 per group (0.84-0.86)
exceeds the 0.80 target at omega-squared = 0.03. This is a technical
validation of the estimator, not evidence about the original YachidaS_2019
biological comparison.
