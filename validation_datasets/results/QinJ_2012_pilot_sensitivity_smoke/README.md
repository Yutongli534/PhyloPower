# QinJ_2012 gene-workflow smoke run

Quick pipeline check for the curatedMetagenomicData QinJ_2012 (type 2
diabetes versus control) PhyloPower gene validation. Uses deliberately small
settings; not for interpretation. The formal run lives in
`../QinJ_2012_core_matched/`.

## Design

- Dataset: curatedMetagenomicData `QinJ_2012` (ExperimentHub EH7235)
- Comparison: T2D versus control, 20 samples per group
  (gender-stratified random draw, seed 20260614; see
  `../../processed/QinJ_2012/preparation_summary.json`)
- Features: 285 MetaPhlAn species matched to the package phylogenetic tree
- Pilot size: 10 per group; evaluation size: 10 per group
- Target effect: omega-squared = 0.03; target power: 0.80
- Bootstrap replicates: 20; PERMANOVA permutations: 49
- Synthetic pool: 60 per group; reduced PCAM grid (5 pi x 3 scale candidates,
  4 power points, 2 near-zero points)
- Random seed: 20260614

## Reproduction command

```sh
/opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python scripts/run_gene_pilot_sensitivity.py \
  --workflow gene \
  --gene-table validation_datasets/processed/QinJ_2012/table.csv \
  --gene-taxonomy validation_datasets/processed/QinJ_2012/taxonomy.csv \
  --gene-tree validation_datasets/processed/QinJ_2012/rooted-tree.nwk \
  --gene-group validation_datasets/processed/QinJ_2012/group.csv \
  --gene-pilot-ns 10 \
  --gene-eval-observed 10 \
  --gene-eval-ns 10 \
  --gene-power-points 4 \
  --gene-near-zero-points 2 \
  --gene-pi-candidates 5 \
  --gene-scale-candidates 3 \
  --pool-size-per-group 60 \
  --boot-number 20 \
  --permutations 49 \
  --target-omega2 0.03 \
  --target-power 0.8 \
  --scenario-n-jobs 2 \
  --random-seed 20260614 \
  --out validation_datasets/results/QinJ_2012_pilot_sensitivity_smoke
```

`scripts/run_gene_pilot_sensitivity.py` is a thin launcher that wraps
`skbio.stats.distance.permanova` with a `seed` keyword (scikit-bio 0.6.0 in
the validated QIIME 2 environment lacks it and permutes through the global
`numpy.random` state) and otherwise delegates to
`run_paper_core_pilot_sensitivity.py` unchanged.

## Result

Pipeline completed end to end. Realized scenarios spanned omega-squared
0.000-0.344; the null scenario gave power 0.00 (nominal alpha 0.05), and
power rose monotonically with effect size (0.15 at omega-squared 0.034, 0.30
at 0.142, 1.00 at 0.344, 20 bootstrap replicates each), confirming Type I
control and a sane power gradient before the formal run.
