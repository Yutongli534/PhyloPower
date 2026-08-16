#!/usr/bin/env Rscript
# Route A: micropower power ESTIMATES for the accuracy benchmark.
# Each "rep" is one standard micropower estimate for a cell:
# bootPower(boot_number = 200, subject_group_vector = c(n, n), alpha = 0.05),
# i.e. the package's documented with-replacement bootstrap of the distance
# matrix (PERMANOVA via legacy adonis, 1000 permutations, package default).
# Usage: Rscript run_accuracy_micropower.R [pilot|extended]
# Output: benchmark/results/accuracy_estimates/micropower_<metric>_<tag>_n<n>.csv

args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args_all, value = TRUE))
bench_dir <- dirname(dirname(normalizePath(file_arg)))
# patch vegan::adonis before loading micropower (see compat_adonis.R)
source(file.path(bench_dir, "r", "compat_adonis.R"))
suppressPackageStartupMessages({
  library(vegan)
  library(micropower)
})

grid_name <- commandArgs(trailingOnly = TRUE)[1]
if (is.na(grid_name)) grid_name <- "pilot"
GRIDS <- list(
  pilot = list(tiers = c(0, 1.0), ns = c(6, 10)),
  extended = list(tiers = c(0, 0.5, 1.0), ns = c(4, 6, 8, 10, 14))
)
grid <- GRIDS[[grid_name]]

data_dir <- file.path(bench_dir, "data")
out_dir <- file.path(bench_dir, "results", "accuracy_estimates")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

N_REPS <- 50
BOOT_NUMBER <- 200
ALPHA <- 0.05
SEED <- 20260614
N_CORES <- 4
METRICS <- c("braycurtis", "wunifrac")

scale_tag <- function(x) gsub("\\.", "p", if (x == round(x)) sprintf("%.1f", x) else as.character(x))

load_dm <- function(metric, tag) {
  dm <- as.matrix(read.csv(file.path(data_dir, sprintf("dm_%s_scale%s.csv", metric, tag)),
                           row.names = 1, check.names = FALSE))
  grp <- read.csv(file.path(data_dir, sprintf("group_scale%s.csv", tag)), stringsAsFactors = FALSE)
  g <- grp$group[match(rownames(dm), grp$sample_id)]
  idx <- ave(seq_along(g), g, FUN = seq_along)
  dimnames(dm) <- list(paste0(g, "s", idx), paste0(g, "s", idx))
  dm
}

for (metric in METRICS) {
  for (scale in grid$tiers) {
    tag <- scale_tag(scale)
    dm <- load_dm(metric, tag)
    for (n in grid$ns) {
      rows <- parallel::mclapply(seq_len(N_REPS), function(rep) {
        set.seed(SEED + rep + sum(utf8ToInt(paste(metric, tag, n))))
        bp <- bootPower(list(tier = dm), boot_number = BOOT_NUMBER,
                        subject_group_vector = c(n, n), alpha = ALPHA)
        data.frame(tool = "micropower", metric = metric, between_scale = scale,
                   n_per_group = n, rep = rep, power_est = unique(bp$power),
                   boot_number = BOOT_NUMBER, permutations = 1000)
      }, mc.cores = N_CORES)
      out <- do.call(rbind, rows)
      write.csv(out, file.path(out_dir, sprintf("micropower_%s_scale%s_n%d.csv", metric, tag, n)),
                row.names = FALSE)
      cat(metric, "scale", scale, "n", n, "done\n", flush = TRUE)
    }
  }
}
cat("MICROPOWER ACCURACY DONE\n")
