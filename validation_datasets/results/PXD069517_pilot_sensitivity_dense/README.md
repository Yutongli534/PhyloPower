# PXD069517 dense pilot-size sensitivity analysis

This run keeps exactly one pilot draw for each pilot size. It increases the
effect-grid density and bootstrap precision to determine whether the curve
separation in the initial figure was caused by sparse effect points.

## Design

- Pilot sizes: 5, 7, and 10 per group
- Pilot draws: one draw per pilot size
- Evaluation size: 14 per group
- Formal effect points: 18 per pilot after target refinement
- Target effect: omega-squared = 0.05
- Target power: 0.80
- Bootstrap replicates: 300
- PERMANOVA permutations: 99
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
  --protein-strengths auto \
  --mdctf-strength-candidates 25 \
  --mdctf-strength-max 4 \
  --mdctf-power-points 15 \
  --mdctf-plateau-points 3 \
  --mdctf-preview-pool-size 200 \
  --mdctf-refine-target-points 3 \
  --power-preview-boot-number 30 \
  --power-preview-permutations 49 \
  --pool-size-per-group 200 \
  --boot-number 300 \
  --permutations 99 \
  --target-omega2 0.05 \
  --target-power 0.8 \
  --scenario-n-jobs 1 \
  --random-seed 20260614 \
  --out validation_datasets/results/PXD069517_pilot_sensitivity_dense
```

## Result

Fitted power at omega-squared = 0.05:

- pilot n=5: 0.742
- pilot n=7: 0.846
- pilot n=10: 0.896

The full target-power range was 0.154. All three pilots had directly simulated
points around omega-squared = 0.05, so none of these target estimates depends
on extrapolation. No bootstrap replicates failed.

Increasing the grid from 7 to 18 formal points per pilot removed the sparse-grid
and target-coverage problems but did not remove the curve separation. Under
the required single-draw design, the separation should therefore be reported
as observed pilot-size sensitivity rather than attributed to too few effect
points.

