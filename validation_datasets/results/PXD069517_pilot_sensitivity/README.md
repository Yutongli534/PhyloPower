# PXD069517 pilot-size sensitivity analysis

This is an external-data technical validation of the manuscript-aligned
`phylopower.paper_core` protein workflow.

## Design

- Dataset: PRIDE PXD069517
- Comparison: celiac disease only (`CD_only`) versus celiac disease with
  poly-autoimmunity (`PolyAI_CD`)
- Available samples: 14 per group
- Pilot sizes: 5, 7, and 10 per group
- Evaluation size: 14 per group
- Target effect: omega-squared = 0.05
- Target power: 0.80
- Bootstrap replicates: 100
- PERMANOVA permutations: 99
- Reusable synthetic pool: 200 per group
- Random seed: 20260614

## Reproducible command

```sh
python3 run_paper_core_pilot_sensitivity.py \
  --workflow protein \
  --protein-table validation_datasets/processed/PXD069517/protein_taxon_function.csv \
  --protein-tree validation_datasets/processed/PXD069517/rooted-tree.nwk \
  --protein-group validation_datasets/processed/PXD069517/group.csv \
  --protein-pilot-ns 5,7,10 \
  --protein-eval-observed 14 \
  --protein-eval-ns 14 \
  --protein-strengths 0,0.25,0.5,0.75,1,1.5,2 \
  --pool-size-per-group 200 \
  --boot-number 100 \
  --permutations 99 \
  --target-omega2 0.05 \
  --target-power 0.8 \
  --scenario-n-jobs 1 \
  --random-seed 20260614 \
  --out validation_datasets/results/PXD069517_pilot_sensitivity
```

## Result

Fitted power at omega-squared = 0.05 was 0.820 for pilot n=5, 0.804 for
pilot n=7, and 0.785 for pilot n=10. The mean was 0.803, the sample-size
sensitivity CV was 2.22%, and the full range was 0.0357. Null-point power was
0.07, 0.04, and 0.04, close to the nominal alpha of 0.05. No bootstrap
replicates failed.

The curves support local pilot-size stability around the target effect for
this dataset. This is a technical external validation, not evidence of a
biological difference in the original cohort. The unmodified observed
distance matrix had omega-squared = -0.00693.

## Important limitation

The source workbook supplies a taxonomic lineage rather than a sequence-based
reference phylogeny. The validation tree is therefore a rooted taxonomic
hierarchy with unit branch lengths. A sequence-derived tree should be used for
the final biological analysis if matching sequences or reference genomes can
be obtained.

