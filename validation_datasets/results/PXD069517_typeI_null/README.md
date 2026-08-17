# PXD069517 Type-I calibration check (metaproteomics)

Null calibration of the PXD069517 metaproteomic workflow, mirroring the
main-text Figure 1 design: independent PCAM null pools
(`pcam_gen.pcam_null_pool`, donors drawn from the pooled cohort, no group
deviation) -> the real PhyloFunc distance recomputed on each pool ->
relabel-bootstrap PERMANOVA p-values (bootstrap with replacement from the
whole pool combined with random group labels, so a single pool's fixed
realized null effect cannot accumulate with sample size).

## Design

- Dataset: PRIDE PXD069517 (`CD_only` vs `PolyAI_CD`, 14 per group)
- Independent null pools: 10 (pool size 200 per group, ndon=1)
- Relabel replicates per pool: 100 per evaluation size
- Evaluation sizes: 14 and 80 per group
- PERMANOVA permutations: 199
- Nominal alpha: 0.05
- Random seed: 20260614

## Reproducible command

```sh
/opt/miniconda3/envs/qiime2-metagenome-2024.10/bin/python \
    figures/suppfig2_feasibility_spectrum.py --compute --workers 4
```

(The retired producer `analysis/run_pxd069517_typeI.py` was merged into
`figures/suppfig2_feasibility_spectrum.py` as its `--compute` mode and
archived in `_archive_scripts/`; run from the repository root.)

## Outputs

- `typeI_null_pvalues.csv` — every null p-value (pool seed x evaluation size
  x replicate)
- `typeI_summary.json` — rejection rate at alpha = 0.05 with its standard
  error and a KS uniformity test per evaluation size
- `typeI_null_qq.{png,pdf}` — observed versus expected null p-value QQ plot
  with a 95% beta band per evaluation size

## Result

Observed rejection rates at alpha = 0.05 were 0.041 (SE 0.006) at n=14 per
group and 0.052 (SE 0.007) at n=80 per group, both consistent with the
nominal level. The p-value distribution was uniform at n=80 (KS P=0.55); at
n=14 the KS test was significant (P=0.0056), reflecting the discreteness of
199-permutation PERMANOVA p-values on 28 samples rather than inflation — the
rejection rate at alpha is, if anything, slightly conservative.

## Important limitation

The PXD069517 tree is a rooted taxonomic hierarchy with unit branch lengths,
not a sequence-derived phylogeny.
