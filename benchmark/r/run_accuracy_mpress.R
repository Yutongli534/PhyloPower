#!/usr/bin/env Rscript
# Route A: MPrESS power ESTIMATES for the accuracy benchmark.
# Each "rep" is one standard MPrESS estimate for a cell: 200 of its own
# sampling-branch replicates (n per group drawn without replacement from the
# pool, distances recomputed through MPrESS's phyloseq pipeline, PERMANOVA via
# legacy adonis with 999 permutations), matching run_mpress.R.
# Usage: Rscript run_accuracy_mpress.R [pilot|extended]
# Output: benchmark/results/accuracy_estimates/mpress_<metric>_<tag>_n<n>.csv

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args_all, value = TRUE))
bench_dir <- dirname(dirname(normalizePath(file_arg)))
# patch vegan::adonis before loading mpress (see compat_adonis.R)
source(file.path(bench_dir, "r", "compat_adonis.R"))
suppressPackageStartupMessages({
  library(vegan)
  library(phyloseq)
  library(ape)
  library(mpress)
})

grid_name <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(grid_name)) grid_name <- "pilot"
GRIDS <- list(
  pilot = list(tiers = c(0, 1.0), ns = c(6, 10)),
  extended = list(tiers = c(0, 0.5, 1.0), ns = c(4, 6, 8, 10, 14))
)
grid <- GRIDS[[grid_name]]

repo_dir <- dirname(bench_dir)
data_dir <- file.path(bench_dir, "data")
out_dir <- file.path(bench_dir, "results", "accuracy_estimates")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

N_REPS <- 50
N_INTERNAL <- 200
ALPHA <- 0.05
SEED <- 20260614
N_CORES <- 4
METRICS <- c(braycurtis = "bray", wunifrac = "wunifrac")

tax_raw <- read.csv(file.path(repo_dir, "phylopower", "datagene", "taxonomy.csv"),
                    stringsAsFactors = FALSE)
tax_split <- do.call(rbind, strsplit(tax_raw$Taxon, ";", fixed = TRUE))
colnames(tax_split) <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
rownames(tax_split) <- tax_raw[["Feature.ID"]]
TAX <- tax_table(tax_split)
TREE <- read.tree(file.path(repo_dir, "phylopower", "datagene", "rooted-tree.nwk"))
if (!is.rooted(TREE)) TREE <- phytools::midpoint.root(TREE)

scale_tag <- function(x) gsub("\\.", "p", if (x == round(x)) sprintf("%.1f", x) else as.character(x))

build_phyloseq <- function(tag) {
  pool <- read.csv(file.path(data_dir, sprintf("pool_scale%s.csv", tag)),
                   row.names = 1, check.names = FALSE)
  grp <- read.csv(file.path(data_dir, sprintf("group_scale%s.csv", tag)), stringsAsFactors = FALSE)
  rownames(grp) <- grp$sample_id
  phyloseq(otu_table(as.matrix(pool), taxa_are_rows = TRUE),
           sample_data(grp[, "group", drop = FALSE]), TAX, TREE)
}

for (metric in names(METRICS)) {
  dist_metric <- METRICS[[metric]]
  for (scale in grid$tiers) {
    tag <- scale_tag(scale)
    ps <- build_phyloseq(tag)
    for (n in grid$ns) {
      rows <- parallel::mclapply(seq_len(N_REPS), function(rep) {
        set.seed(SEED + rep + sum(utf8ToInt(paste(metric, tag, n))))
        p_vals <- replicate(N_INTERNAL, {
          sub <- mpress:::.get_sampled_otus(ps, k = n, "group", c("Cd", "Ni"))
          d <- mpress:::.get_distance_value(sub, dist_metric)
          mpress:::.get_test_pvalue(sub, "group", d, "permanova")
        })
        data.frame(tool = "MPrESS", metric = metric, between_scale = scale,
                   n_per_group = n, rep = rep, power_est = mean(p_vals < ALPHA),
                   boot_number = N_INTERNAL, permutations = 999)
      }, mc.cores = N_CORES)
      out <- do.call(rbind, rows)
      write.csv(out, file.path(out_dir, sprintf("mpress_%s_scale%s_n%d.csv", metric, tag, n)),
                row.names = FALSE)
      cat(metric, "scale", scale, "n", n, "done\n", flush = TRUE)
    }
  }
}
cat("MPRESS ACCURACY DONE\n")
