#!/usr/bin/env Rscript
# Cross-language validation of the omega^2 definition.
# Recomputes omega^2 from the exported distance matrices with a from-scratch
# Anderson SS decomposition and compares against PhyloPower's
# phylopower.core.compute_omega2 values stored in tier_summary.csv.
# If micropower is installed, micropower::calcOmega2 is checked too.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
bench_dir <- dirname(dirname(normalizePath(file_arg)))
data_dir <- file.path(bench_dir, "data")
out_dir <- file.path(bench_dir, "results")
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

lower_tri <- function(m) m[lower.tri(m)]

omega2_from_dm <- function(dm, groups) {
  # dm: square numeric matrix; groups: factor aligned to dimnames(dm)
  n <- nrow(dm)
  k <- length(unique(groups))
  sst <- sum(lower_tri(dm)^2) / n
  ssw <- 0
  for (g in unique(groups)) {
    members <- rownames(dm)[groups == g]
    sub <- dm[members, members, drop = FALSE]
    if (length(members) > 1) ssw <- ssw + sum(lower_tri(sub)^2) / length(members)
  }
  ssa <- sst - ssw
  df_b <- k - 1
  df_w <- n - k
  ms_w <- ssw / df_w
  (ssa - df_b * ms_w) / (sst + ms_w)
}

py <- read.csv(file.path(data_dir, "tier_summary.csv"))
has_micropower <- requireNamespace("micropower", quietly = TRUE)

py$scale_tag <- vapply(py$between_scale, function(x)
  if (x == round(x)) sprintf("%.1f", x) else as.character(x), character(1))
py$scale_tag <- gsub("\\.", "p", py$scale_tag)

rows <- list()
for (i in seq_len(nrow(py))) {
  scale_tag <- py$scale_tag[i]
  metric <- py$metric[i]
  dm <- as.matrix(read.csv(file.path(data_dir, sprintf("dm_%s_scale%s.csv", metric, scale_tag)),
                           row.names = 1, check.names = FALSE))
  grp <- read.csv(file.path(data_dir, sprintf("group_scale%s.csv", scale_tag)),
                  stringsAsFactors = FALSE)
  groups <- grp$group[match(rownames(dm), grp$sample_id)]
  stopifnot(!any(is.na(groups)))

  r_omega2 <- omega2_from_dm(dm, groups)
  row <- data.frame(
    between_scale = py$between_scale[i],
    metric = metric,
    omega2_phylopower = py$omega2_full_pool[i],
    omega2_r_manual = r_omega2,
    abs_diff = abs(r_omega2 - py$omega2_full_pool[i])
  )
  if (has_micropower) {
    dm_renamed <- dm
    dimnames(dm_renamed) <- lapply(dimnames(dm_renamed), function(ids)
      paste0(gsub("^syn_", "", ids), collapse = NULL))
    # micropower expects names like g1s1: rebuild as <group>s<index>
    idx <- ave(seq_along(groups), groups, FUN = seq_along)
    new_names <- paste0(groups, "s", idx)
    dimnames(dm_renamed) <- list(new_names, new_names)
    row$omega2_micropower <- micropower::calcOmega2(dm_renamed)
  }
  rows[[i]] <- row
}
res <- do.call(rbind, rows)
write.csv(res, file.path(out_dir, "omega2_validation.csv"), row.names = FALSE)
cat("max |R - Python| :", max(res$abs_diff), "\n")
if (has_micropower) {
  cat("max |micropower - Python| :", max(abs(res$omega2_micropower - res$omega2_phylopower)), "\n")
}
print(res)
