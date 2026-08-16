# QinJ_2012 core-matched pilot sensitivity analysis

External-data validation of the manuscript-aligned `phylopower.paper_core`
gene workflow on curatedMetagenomicData `QinJ_2012` (type 2 diabetes versus
control). This run uses the same conceptual design as the
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

`validation_datasets/processed/QinJ_2012/`, produced by
`scripts/prepare_qinj2012.py` (see its `preparation_summary.json`):

- 20 T2D versus 20 control samples, drawn from 170 T2D / 174 control
  available samples by gender-stratified random sampling (proportional
  allocation, largest remainder; seed 20260614)
- 285 MetaPhlAn species features (98.79% of the 659 species rows matched a
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
  --gene-table validation_datasets/processed/QinJ_2012/table.csv \
  --gene-taxonomy validation_datasets/processed/QinJ_2012/taxonomy.csv \
  --gene-tree validation_datasets/processed/QinJ_2012/rooted-tree.nwk \
  --gene-group validation_datasets/processed/QinJ_2012/group.csv \
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
  --out validation_datasets/results/QinJ_2012_core_matched
```

`scripts/run_gene_pilot_sensitivity.py` is a thin launcher that wraps
`skbio.stats.distance.permanova` with a `seed` keyword (scikit-bio 0.6.0 in
the validated QIIME 2 environment lacks it and permutes through the global
`numpy.random` state) and otherwise delegates to
`analysis/run_cli_pilot_sensitivity.py` unchanged.

## Target-effect comparison

| Evaluation size | Pilot size | Fitted power at omega-squared = 0.03 |
|---:|---:|---:|
| 20 | 7 | 0.263 |
| 20 | 10 | 0.247 |
| 20 | 20 | 0.272 |
| 80 | 7 | 0.846 |
| 80 | 10 | 0.848 |
| 80 | 20 | 0.848 |

The range across pilot curves is 0.025 at the observed evaluation size and
0.003 at the extrapolated size, comparable to the pilot-size consistency
reported for the original core run (0.069) and the PXD069517 protein
validation (0.081). At the extrapolated size of 80 per group the fitted
power at the target effect is approximately 0.85, consistent with the
target-power-oriented design.

Null-scenario power (Type I, nominal alpha = 0.05) was 0.035-0.060 at
evaluation size 20 and 0.050-0.095 at evaluation size 80 across pilot sizes,
consistent with calibrated Type I error given 200 bootstrap replicates.

## Interpretation

The gene validation framework reproduces on this independent public dataset:
synthetic pools calibrated to 7-20 pilot samples per group yield power
curves that agree closely across pilot sizes, Type I error is controlled,
and the fitted power at the extrapolated size of 80 per group (0.85) exceeds
the 0.80 target at omega-squared = 0.03. This is a technical validation of
the estimator, not evidence about the original QinJ_2012 biological
comparison.
