#!/usr/bin/env Rscript
# micropower benchmark driver.
# For each (metric, effect tier, n per group): micropower::bootPower()
# bootstraps the exported pool distance matrix (with replacement) and estimates
# PERMANOVA power at alpha = 0.05. boot_number = 200 replicates per combo.
# Output: benchmark/results/power_micropower.csv

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
bench_dir <- dirname(dirname(normalizePath(file_arg)))
# patch vegan::adonis before loading micropower (see compat_adonis.R)
source(file.path(bench_dir, "r", "compat_adonis.R"))
suppressPackageStartupMessages({
  library(vegan)
  library(micropower)
})
data_dir <- file.path(bench_dir, "data")
out_dir <- file.path(bench_dir, "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

N_GRID <- c(4, 6, 8, 10, 14, 20)
BOOT_NUMBER <- 200
ALPHA <- 0.05
SEED <- 20260614
N_CORES <- 4

tier_summary <- read.csv(file.path(data_dir, "tier_summary.csv"), stringsAsFactors = FALSE)
tier_summary$scale_tag <- vapply(tier_summary$between_scale, function(x)
  if (x == round(x)) sprintf("%.1f", x) else as.character(x), character(1))
tier_summary$scale_tag <- gsub("\\.", "p", tier_summary$scale_tag)

load_dm <- function(metric, scale_tag) {
  dm <- as.matrix(read.csv(file.path(data_dir, sprintf("dm_%s_scale%s.csv", metric, scale_tag)),
                           row.names = 1, check.names = FALSE))
  grp <- read.csv(file.path(data_dir, sprintf("group_scale%s.csv", scale_tag)),
                  stringsAsFactors = FALSE)
  g <- grp$group[match(rownames(dm), grp$sample_id)]
  idx <- ave(seq_along(g), g, FUN = seq_along)
  # micropower expects dimnames formatted "<group>s<i>" (groupNames strips "s.*")
  dimnames(dm) <- list(paste0(g, "s", idx), paste0(g, "s", idx))
  dm
}

combos <- expand.grid(metric = c("braycurtis", "wunifrac"),
                      scale_tag = unique(tier_summary$scale_tag),
                      n = N_GRID, stringsAsFactors = FALSE)

run_one <- function(metric, scale_tag, n) {
  dm <- load_dm(metric, scale_tag)
  set.seed(SEED + sum(utf8ToInt(paste(metric, scale_tag, n))))
  bp <- bootPower(list(tier = dm), boot_number = BOOT_NUMBER,
                  subject_group_vector = c(n, n), alpha = ALPHA)
  data.frame(
    tool = "micropower",
    metric = metric,
    between_scale = tier_summary$between_scale[tier_summary$scale_tag == scale_tag &
                                                 tier_summary$metric == metric],
    omega2 = unique(bp$simulated_omega2),
    mean_boot_omega2 = mean(bp$observed_omega2),
    n_per_group = n,
    power = unique(bp$power),
    n_sim = BOOT_NUMBER,
    alpha = ALPHA
  )
}

results <- parallel::mclapply(seq_len(nrow(combos)), function(i) {
  tryCatch(run_one(combos$metric[i], combos$scale_tag[i], combos$n[i]),
           error = function(e) data.frame(tool = "micropower", metric = combos$metric[i],
                                          between_scale = NA, omega2 = NA, mean_boot_omega2 = NA,
                                          n_per_group = combos$n[i], power = NA, n_sim = BOOT_NUMBER,
                                          alpha = ALPHA, error = conditionMessage(e)))
}, mc.cores = N_CORES)

out <- do.call(rbind, results)
out <- out[order(out$metric, out$between_scale, out$n_per_group), ]
write.csv(out, file.path(out_dir, "power_micropower.csv"), row.names = FALSE)
print(out)
