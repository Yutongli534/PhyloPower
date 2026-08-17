# PXD069517 core-matched pilot sensitivity analysis

This run uses the same conceptual design as the original `core` sensitivity figure:

- workflow: protein
- pilot sizes: 7, 10, and the full available PXD group size of 14
- observed evaluation size: 14 per group
- extrapolated evaluation size: 80 per group
- target effect size: omega-squared = 0.03
- target power: 0.80
- automatically selected strength grid with 12 power-curve points
- final power estimates: 100 bootstrap replicates and 99 permutations

The full pilot size is 14 rather than the core figure's 17 because PXD069517
contains 14 samples in each analysis group.

## Reproduction command

```bash
python3 scripts/run_validation_pilot_sensitivity.py \
  --workflow protein \
  --protein-table validation_datasets/processed/PXD069517/protein_taxon_function.csv \
  --protein-tree validation_datasets/processed/PXD069517/rooted-tree.nwk \
  --protein-group validation_datasets/processed/PXD069517/group.csv \
  --protein-pilot-ns 7,10,14 \
  --protein-eval-observed 14 \
  --protein-eval-ns 14,80 \
  --eval-extrapolate 80 \
  --protein-strengths auto \
  --mdctf-strength-candidates 15 \
  --mdctf-strength-max 4 \
  --mdctf-power-points 12 \
  --mdctf-plateau-points 5 \
  --mdctf-preview-pool-size 180 \
  --mdctf-refine-target-points 0 \
  --power-preview-boot-number 20 \
  --power-preview-permutations 49 \
  --pool-size-per-group 180 \
  --boot-number 100 \
  --permutations 99 \
  --target-omega2 0.03 \
  --target-power 0.8 \
  --scenario-n-jobs 1 \
  --random-seed 20260614 \
  --out validation_datasets/results/PXD069517_core_matched
```

## Target-effect comparison

| Evaluation size | Pilot size | Fitted power at omega-squared = 0.03 |
|---:|---:|---:|
| 14 | 7 | 0.496 |
| 14 | 10 | 0.578 |
| 14 | 14 | 0.516 |
| 80 | 7 | 0.989 |
| 80 | 10 | 0.977 |
| 80 | 14 | 0.996 |

For comparison, the original core run reported target powers of
0.467, 0.518, and 0.536 at its observed evaluation size of 17. The range
across pilot curves is therefore 0.081 for PXD069517 and 0.069 for the
original core run, indicating comparable pilot-size consistency.

## Important limitation

The PXD069517 tree supplied to this run is a rooted taxonomic hierarchy with
unit branch lengths, whereas the original core analysis used its original
phylogenetic branch lengths. This run validates the executable workflow and
pilot-size sensitivity pattern, but should not be interpreted as a
branch-length-equivalent biological replication.
