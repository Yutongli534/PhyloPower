# Pilot-information convergence supplement

## Purpose

This analysis isolates finite-pilot information as a source of disagreement
among extrapolated power curves. The empirical cohort, effect grid, evaluation
size, generator settings, and random numbers are held fixed; only the pilot
subset and its size vary.

## Proof-of-concept settings

- Modality: metagenomics with Gemelli distances
- Pilot sizes: 4, 7, and 10 samples per group
- Repeated pilot draws: 20 per pilot size
- Evaluation size: 80 samples per group
- Synthetic pool: 80 samples per group
- Power bootstrap replicates: 40
- PERMANOVA permutations: 99
- Reference: the complete empirical cohort analyzed with the same common random
  numbers
- Curve disagreement: mean absolute difference in power from the full-cohort
  reference across the displayed realized-effect range

Donor indices are sorted after pilot subsampling. This prevents a full-size
pilot from acquiring artificial variability merely because the same donors
appear in a different order. Common random numbers are used across pilot sizes
so the comparison targets information loss from pilot subsampling rather than
generator Monte Carlo noise.

## Main result

Median curve-to-reference disagreement decreased from 0.0323 at pilot n=4 to
0.0137 at n=7 and 0.0016 at n=10. Across all repeated pilots, pilot size was
strongly negatively associated with curve disagreement (Spearman rho=-0.878,
P=3.19e-20). A repeated-measures Friedman test across the three pilot sizes was
also significant (chi-square=33.6, P=5.06e-8), as were adjacent one-sided paired
Wilcoxon comparisons (n=4 vs n=7: P=1.05e-4; n=7 vs n=10: P=9.54e-7). The realized omega-squared
required for 80% power simultaneously approached the full-cohort reference
(0.01367): median values were 0.02110, 0.01677, and 0.01383 for pilot n=4, 7,
and 10, respectively.

## Draft caption

**Supplementary Figure X. Finite-pilot disagreement and convergence of
power-curve extrapolation.** Metagenomic pilots of 4, 7, or 10 samples per
group were repeatedly subsampled from a fixed empirical cohort and analyzed
with identical effect grids, evaluation size (n=80 per group), and common random
numbers. (a) Median fitted power curves and 10th-90th percentile envelopes over
20 repeated pilots; the black curve is the full-cohort reference. (b)
Individual curves illustrate greater between-pilot disagreement at the
smallest pilot size. (c) Mean absolute curve-to-reference power difference
decreased as pilot size increased. (d) The realized effect size required to
reach 80% power became less variable and converged toward the full-cohort
reference. These results indicate that disagreement among curves from small
pilots can arise from finite pilot information available for extrapolation and
should not, by itself, be interpreted as failure of the framework.

## Manuscript-quality rerun

Before submission, increase the number of repeated pilots, synthetic-pool size,
and power-bootstrap replicates, and repeat the analysis for the final
metaproteomic MDC-TF-MC pipeline. The present run establishes that the proposed
figure and diagnostic behave as intended, but it is a computational
proof-of-concept rather than the final high-precision supplement.
