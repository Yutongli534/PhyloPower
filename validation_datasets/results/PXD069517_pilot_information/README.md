# PXD069517 pilot-information convergence analysis (metaproteomics)

This is the metaproteomic counterpart of the metagenomic pilot-information
supplement (`pilot_information_supplement/`, Supplementary Figure S1): an
external-data validation on PRIDE PXD069517 that isolates finite-pilot
information as a source of disagreement among extrapolated power curves.

The empirical cohort, effect grid, evaluation size, generator settings, and
common random numbers are held fixed; only the pilot subset and its size vary.
Curves are compared with the curve obtained from the full 14-per-group
cohort, so decreasing curve-to-reference error with pilot size is evidence
for finite-pilot information as the source of disagreement.

## Design

- Dataset: PRIDE PXD069517
- Comparison: celiac disease only (`CD_only`) versus celiac disease with
  poly-autoimmunity (`PolyAI_CD`), 14 samples per group
- Generator: PCAM (`pcam_gen.pcam_pool`, ndon=1) with the real PhyloFunc
  distance recomputed on every synthetic pool
- Pilot sizes: 5, 7, and 10 samples per group
- Repeated pilot draws: 20 per pilot size (sorted donor indices, so a
  full-size pilot cannot acquire artificial variability from donor order)
- Effect grid: donor mixing pi = 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 at scale = 1
  (pi = 0.5 is the true null point), then deviation amplification
  scale = 1.15, 1.3, 1.5, 1.7, 2.0 at pi = 1 — 11 points; the realized
  omega-squared of every generated pool is measured and used as the curve
  abscissa
- Evaluation size: 14 per group (the observed cohort size; power is computed
  by bootstrap with replacement from each pool)
- Reusable synthetic pool: 200 per group
- Power bootstrap replicates: 200
- PERMANOVA permutations: 199
- Random seed: 20260614
- Common random numbers across pilot sizes and repetitions: generation seeds
  depend only on the effect-grid index, so the comparison targets information
  loss from pilot subsampling rather than generator Monte Carlo noise
- Reference: the complete 14-per-group cohort analyzed with the same common
  random numbers (`raw_full_cohort_reference.csv`)

Because the unmodified PXD069517 distance matrix has omega-squared = -0.00693
(essentially no observed group separation), a small pilot's apparent group
difference is dominated by subsampling noise. This run is therefore a
technical validation of the extrapolation machinery — convergence of pilot
curves toward the full-cohort reference with increasing pilot size — not
evidence about biology.

## Reproducible command

```sh
python3 run_pilot_information_supplement_protein_pxd.py \
  --out validation_datasets/results/PXD069517_pilot_information
python3 export_pilot_information_two_panel_protein.py
```

All parameters default to the values listed above. The run caches each
repeated pilot draw (`raw_pilot_{5,7,10}.csv`), so an interrupted run resumes
where it stopped; use `--force` to restart from scratch. Plotting only:

```sh
python3 run_pilot_information_supplement_protein_pxd.py --plot-only \
  --out validation_datasets/results/PXD069517_pilot_information
```

## Outputs

- `pilot_information_raw.csv` — every (pilot draw x effect point) with the
  realized omega-squared and bootstrap power
- `pilot_information_curve_metrics.csv` — per-draw mean absolute
  curve-to-reference power difference and the omega-squared required for 80%
  power (input to the two-panel figure)
- `pilot_information_summary.json` — median metrics, Spearman/Friedman and
  paired Wilcoxon tests of convergence
- `pilot_information_supplement.{png,pdf}` — four-panel diagnostic
  (curves, between-pilot disagreement, curve error, threshold stability)
- `pilot_information_two_panel.{png,pdf}` — the two-panel export styled to
  match Supplementary Figure S1

## Result

Formal run (20 repeated pilot draws per size, 200 bootstrap replicates, 199
permutations): median curve-to-reference disagreement decreased from 0.1126
at pilot n=5 to 0.0534 at n=7 and 0.0192 at n=10. Across all repeated pilots,
pilot size was strongly negatively associated with curve disagreement
(Spearman rho=-0.936, P=5.90e-28). A repeated-measures Friedman test across
the three pilot sizes was also significant (chi-square=40.0, P=2.06e-9), as
were adjacent one-sided paired Wilcoxon comparisons (n=5 vs n=7 and n=7 vs
n=10: both P=9.54e-7). The realized omega-squared required for 80% power
approached the full-cohort reference (0.0408): median values were 0.0577,
0.0476, and 0.0419 for pilot n=5, 7, and 10, respectively.

## Important limitation

The PXD069517 tree is a rooted taxonomic hierarchy with unit branch lengths,
not a sequence-derived phylogeny (same limitation as the earlier PXD069517
validation runs).
