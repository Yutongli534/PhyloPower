# Protein Taxon-Function Synthetic Pool Design

## Goal

Improve protein sensitivity-curve consistency by changing the synthetic sample pool, not by post-hoc curve fitting. The generator should:

- preserve sample-level Taxon-Function topology, including edge count, taxon degree, function degree, connectance, and degree heterogeneity;
- model both taxon and function dimensions, not independent Taxon-Function edges;
- keep the protein abundance/missingness character of LC-MS/metaproteomics data;
- avoid hard-coded constants and avoid using the full sensitivity curve as feedback;
- use only the pilot being evaluated, plus optional user-supplied external priors.

## Literature Anchors

Metaproteomics can be represented as a sample-specific proteomic content network: a bipartite graph linking taxa and expressed protein functions. Proteome-level functional redundancy work explicitly uses this network view and treats topology, presence/absence, protein abundance, and biomass as jointly informative.

Bipartite ecological-network literature warns that degree, connectance, network size, and sampling intensity can dominate apparent network indices. Therefore, synthetic pools should preserve these structural statistics or explicitly diagnose their drift.

Entropy/max-entropy bipartite null models and the Bipartite Configuration Model preserve expected node degrees. Degree-corrected stochastic block models extend block models by allowing heterogeneous node degrees within communities. These ideas map naturally to metaproteomics: taxa and functions have broad, uneven degree distributions, and a generator that ignores that heterogeneity can create unrealistic networks.

Proteomics missingness is often MNAR: low-abundance or absent ions are censored below detection, while MAR/MCAR acquisition effects can coexist. A protein generator should therefore generate the detection mask and positive abundance jointly, rather than treating zeros as ordinary Gaussian noise.

## Proposed Generator: TAF-DC Pool

TAF-DC = Taxon-Abundance-Function Degree-Corrected pool.

For each group and pilot:

1. Fit sample-level topology summaries
   - sample edge count distribution;
   - taxon degree distribution per sample;
   - function degree distribution per sample;
   - connectance;
   - nestedness or shared-neighbor motif summaries, when enough samples exist;
   - taxon and function marginal activity.

2. Estimate a degree-corrected bipartite detection model
   - probability of edge `(taxon t, function f)` in sample `i`:
     `logit p_itf = a_i + u_t + v_f + B_taxon_block,function_block + group_offset_tf`
   - `u_t` and `v_f` preserve taxon/function degree heterogeneity;
   - block terms preserve coarse taxon-function modularity;
   - group offsets are shrunk by pilot uncertainty and prevalence.

3. Generate a detection mask
   - draw target sample edge count from the pilot empirical distribution;
   - sample edges using the fitted degree-corrected probabilities;
   - optionally use sequential conditional Poisson sampling to match edge count exactly;
   - reject/repair only when diagnostics show impossible degree drift.

4. Generate positive abundance on present edges
   - decompose log-positive abundance:
     `log y_itf = sample_load_i + taxon_factor_it + function_factor_if + edge_residual_itf`;
   - sample/taxon/function factors are drawn from pilot residual pools with empirical-Bayes shrinkage;
   - edge-level group differences are shrunk by signal/noise and prevalence;
   - rescale total protein abundance using the pilot library/biomass distribution.

5. Effect modulation
   - do not use `pi` as the effect axis;
   - first calibrate topology/presence parameters to preserve within/between distance and degree diagnostics;
   - then modulate a single group-offset amplitude in the positive-abundance component;
   - choose final scale levels by a pilot-only preview of omega2/power transition, not by the full target curve.

## No-Peeking Rule

Allowed during generator fitting:

- the current pilot table and group labels;
- pilot internal diagnostics: degree drift, connectance drift, within/between distance KS, baseline omega2;
- a cheap preview over candidate generator parameters for the same pilot.

Not allowed:

- using full-data sensitivity curves;
- using held-out pilot sizes to tune constants;
- manually choosing scale ranges because they align published curves;
- selecting parameters based on the final cross-pilot overlap.

## Diagnostics To Report

Every synthetic pool should write:

- edge count KS and median ratio;
- taxon degree KS and median ratio;
- function degree KS and median ratio;
- connectance ratio;
- top taxon/function degree rank correlation;
- within-distance KS;
- between-distance KS;
- PERMDISP-like within-dispersion ratio;
- duplicate fraction expected from pool size and eval_n;
- effect grid coverage: number of transition points and saturated points.

## Implementation Plan

1. Add a new experimental generator mode, e.g. `protein_generator="taf-dc"`.
2. Implement topology fitting and diagnostics first; keep abundance generation identical to current PCAM.
3. Replace template-mask presence generation with degree-corrected conditional mask generation.
4. Add taxon/function/sample abundance factors.
5. Add empirical-Bayes shrinkage of group offsets.
6. Use local, preview-selected effect ranges, starting near the fitted baseline.
7. Compare against current PCAM with identical bootstrap/PERMANOVA settings.

The expected improvement is not that tiny pilots magically become certain, but that avoidable generator artifacts are removed: small pilots should no longer produce unrealistically narrow or overly separated Taxon-Function pools just because the template masks and abundance shifts were coupled too tightly.
