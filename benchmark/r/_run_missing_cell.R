#!/usr/bin/env Rscript
# One-off: fill the single missing Route-A cell (MPrESS, wunifrac, scale 1.0, n=14).
# Logic mirrors run_accuracy_mpress.R exactly.

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args_all, value = TRUE))
bench_dir <- dirname(dirname(normalizePath(file_arg)))
source(file.path(bench_dir, "r", "compat_adonis.R"))
suppressPackageStartupMessages({
  library(vegan)
  library(phyloseq)
  library(ape)
  library(mpress)
})

repo_dir <- dirname(bench_dir)
data_dir <- file.path(bench_dir, "data")
out_dir <- file.path(bench_dir, "results", "accuracy_estimates")

N_REPS <- 50
N_INTERNAL <- 200
ALPHA <- 0.05
SEED <- 20260614
N_CORES <- 4

tax_raw <- read.csv(file.path(repo_dir, "phylopower", "datagene", "taxonomy.csv"),
                    stringsAsFactors = FALSE)
tax_split <- do.call(rbind, strsplit(tax_raw$Taxon, ";", fixed = TRUE))
colnames(tax_split) <- c("Kingdom", "Phylum", "Class", "Order", "Family", "Genus", "Species")
rownames(tax_split) <- tax_raw[["Feature.ID"]]
TAX <- tax_table(tax_split)
TREE <- read.tree(file.path(repo_dir, "phylopower", "datagene", "rooted-tree.nwk"))
if (!is.rooted(TREE)) TREE <- phytools::midpoint.root(TREE)

metric <- "wunifrac"
dist_metric <- "wunifrac"
scale <- 1.0
tag <- "1p0"
n <- 14

pool <- read.csv(file.path(data_dir, sprintf("pool_scale%s.csv", tag)),
                 row.names = 1, check.names = FALSE)
grp <- read.csv(file.path(data_dir, sprintf("group_scale%s.csv", tag)), stringsAsFactors = FALSE)
rownames(grp) <- grp$sample_id
ps <- phyloseq(otu_table(as.matrix(pool), taxa_are_rows = TRUE),
               sample_data(grp[, "group", drop = FALSE]), TAX, TREE)

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
cat("MISSING CELL DONE\n")
