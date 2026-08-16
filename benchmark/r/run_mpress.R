#!/usr/bin/env Rscript
# MPrESS benchmark driver.
# For each (metric, effect tier, n per group): replicate MPrESS's estimation
# mechanism with a fixed replicate count -- sampling n samples per group
# without replacement from the pool (mpress:::.get_sampled_otus, the branch
# MPrESS uses whenever n <= available samples), distance computation through
# MPrESS's own phyloseq pipeline (mpress:::.get_distance_value; Bray-Curtis via
# vegan::vegdist on relative abundances, weighted UniFrac via
# phyloseq::distance(..., "wunifrac") with normalized=TRUE), and a PERMANOVA
# p-value per replicate. power = fraction of p < 0.05.
# Output: benchmark/results/power_mpress.csv

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
bench_dir <- dirname(dirname(normalizePath(file_arg)))
# patch vegan::adonis before loading mpress (see compat_adonis.R)
source(file.path(bench_dir, "r", "compat_adonis.R"))
suppressPackageStartupMessages({
  library(vegan)
  library(phyloseq)
  library(ape)
  library(mpress)
})
repo_dir <- dirname(bench_dir)
data_dir <- file.path(bench_dir, "data")
out_dir <- file.path(bench_dir, "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

N_GRID <- c(4, 6, 8, 10, 14, 20)
N_REP <- 200
ALPHA <- 0.05
SEED <- 20260614
N_CORES <- 4
METRICS <- c(braycurtis = "bray", wunifrac = "wunifrac")

# shared taxonomy table and tree (same features in every tier)
tax_raw <- read.csv(file.path(repo_dir, "phylopower", "datagene", "taxonomy.csv"),
                    stringsAsFactors = FALSE)
tax_split <- do.call(rbind, strsplit(tax_raw$Taxon, ";", fixed = TRUE))
colnames(tax_split) <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
rownames(tax_split) <- tax_raw$`Feature.ID`
TAX <- tax_table(tax_split)
TREE <- read.tree(file.path(repo_dir, "phylopower", "datagene", "rooted-tree.nwk"))
# midpoint-root for determinism (phyloseq::UniFrac randomly roots unrooted trees)
if (!is.rooted(TREE)) TREE <- phytools::midpoint.root(TREE)

build_phyloseq <- function(scale_tag) {
  pool <- read.csv(file.path(data_dir, sprintf("pool_scale%s.csv", scale_tag)),
                   row.names = 1, check.names = FALSE)
  grp <- read.csv(file.path(data_dir, sprintf("group_scale%s.csv", scale_tag)),
                  stringsAsFactors = FALSE)
  rownames(grp) <- grp$sample_id
  otu <- otu_table(as.matrix(pool), taxa_are_rows = TRUE)
  phyloseq(otu, sample_data(grp[, "group", drop = FALSE]), TAX, TREE)
}

lower_tri <- function(m) m[lower.tri(m)]
omega2_from_dm <- function(dm, groups) {
  n <- nrow(dm); k <- length(unique(groups))
  sst <- sum(lower_tri(dm)^2) / n
  ssw <- 0
  for (g in unique(groups)) {
    members <- rownames(dm)[groups == g]
    sub <- dm[members, members, drop = FALSE]
    if (length(members) > 1) ssw <- ssw + sum(lower_tri(sub)^2) / length(members)
  }
  ssa <- sst - ssw
  ms_w <- ssw / (n - k)
  (ssa - (k - 1) * ms_w) / (sst + ms_w)
}

tier_summary <- read.csv(file.path(data_dir, "tier_summary.csv"), stringsAsFactors = FALSE)
tier_summary$scale_tag <- vapply(tier_summary$between_scale, function(x)
  if (x == round(x)) sprintf("%.1f", x) else as.character(x), character(1))
tier_summary$scale_tag <- gsub("\\.", "p", tier_summary$scale_tag)

combos <- expand.grid(metric = names(METRICS), scale_tag = unique(tier_summary$scale_tag),
                      n = N_GRID, stringsAsFactors = FALSE)

run_one <- function(metric, scale_tag, n) {
  ps <- build_phyloseq(scale_tag)
  dist_metric <- METRICS[[metric]]
  # realized omega2 for this tier under MPrESS's own distance pipeline
  full_dist <- mpress:::.get_distance_value(ps, dist_metric)
  full_groups <- as(sample_data(ps), "data.frame")$group
  omega2 <- omega2_from_dm(as.matrix(full_dist), full_groups)

  set.seed(SEED + sum(utf8ToInt(paste(metric, scale_tag, n))))
  p_vals <- replicate(N_REP, {
    sub <- mpress:::.get_sampled_otus(ps, k = n, "group", c("Cd", "Ni"))
    d <- mpress:::.get_distance_value(sub, dist_metric)
    mpress:::.get_test_pvalue(sub, "group", d, "permanova")
  })
  data.frame(
    tool = "MPrESS",
    metric = metric,
    between_scale = tier_summary$between_scale[tier_summary$scale_tag == scale_tag &
                                                 tier_summary$metric == metric],
    omega2 = omega2,
    mean_boot_omega2 = NA_real_,
    n_per_group = n,
    power = mean(p_vals < ALPHA),
    n_sim = N_REP,
    alpha = ALPHA
  )
}

results <- parallel::mclapply(seq_len(nrow(combos)), function(i) {
  tryCatch(run_one(combos$metric[i], combos$scale_tag[i], combos$n[i]),
           error = function(e) data.frame(tool = "MPrESS", metric = combos$metric[i],
                                          between_scale = NA, omega2 = NA, mean_boot_omega2 = NA,
                                          n_per_group = combos$n[i], power = NA, n_sim = N_REP,
                                          alpha = ALPHA, error = conditionMessage(e)))
}, mc.cores = N_CORES)

out <- do.call(rbind, results)
out <- out[order(out$metric, out$between_scale, out$n_per_group), ]
write.csv(out, file.path(out_dir, "power_mpress.csv"), row.names = FALSE)
print(out)
