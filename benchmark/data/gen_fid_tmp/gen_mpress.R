
suppressMessages({ library(HMP); library(vegan); library(dirmult) })
args <- commandArgs(trailingOnly = TRUE)
pilot <- round(as.matrix(read.csv(args[1], row.names = 1, check.names = FALSE)))
grp <- read.csv(args[2], stringsAsFactors = FALSE)
out_file <- args[3]
pool_size <- as.integer(args[4])
set.seed(as.integer(args[5]))
groups <- unique(grp$group)
gen_all <- list()
for (g in groups) {
  g_samples <- grp$sample_id[grp$group == g]
  g_table <- pilot[, g_samples, drop = FALSE]
  g_table <- g_table[rowSums(g_table) > 0, , drop = FALSE]
  # HMP::dirmult expects SAMPLES x FEATURES
  g_table_t <- t(g_table)
  fit <- tryCatch(HMP::dirmult(g_table_t), error = function(e) NULL)
  K <- nrow(g_table)
  n_reads <- max(1000, round(mean(colSums(g_table))))
  if (is.null(fit) || inherits(fit, "try-error")) {
    prob <- colSums(g_table_t) / sum(g_table_t)
    sim_data <- t(rmultinom(pool_size, n_reads, prob))
  } else {
    # HMP fit provides pi and theta like dirmult
    sim <- tryCatch(dirmult::simPop(J = pool_size, K = K, n = n_reads,
                       pi = fit$pi, theta = fit$theta),
                    error = function(e) NULL)
    if (is.null(sim)) {
      prob <- colSums(g_table_t) / sum(g_table_t)
      sim_data <- t(rmultinom(pool_size, n_reads, prob))
    } else {
      sim_data <- sim$data
    }
  }
  sim_data <- t(sim_data)
  rownames(sim_data) <- colnames(g_table_t)
  colnames(sim_data) <- paste0(g, "_h", 1:pool_size)
  gen_all[[g]] <- sim_data
}
pool_cols <- pool_size * length(groups)
all_taxa <- rownames(pilot)
gen_table <- matrix(0, nrow = length(all_taxa), ncol = pool_cols)
rownames(gen_table) <- all_taxa
colnames(gen_table) <- character(pool_cols)
ci <- 1
for (g in groups) {
  common <- intersect(all_taxa, rownames(gen_all[[g]]))
  gen_table[common, ci:(ci + pool_size - 1)] <- gen_all[[g]][common, ]
  colnames(gen_table)[ci:(ci + pool_size - 1)] <- colnames(gen_all[[g]])
  ci <- ci + pool_size
}
gen_labels <- rep(groups, each = pool_size)
jac <- as.matrix(vegdist(t(gen_table), method = "jaccard", binary = TRUE))
n_tot <- ncol(gen_table)
pairs <- which(upper.tri(jac), arr.ind = TRUE)
same_group <- gen_labels[pairs[, 1]] == gen_labels[pairs[, 2]]
w <- jac[upper.tri(jac)][same_group]
b <- jac[upper.tri(jac)][!same_group]
write.csv(data.frame(group = c(rep("within", length(w)), rep("between", length(b))),
                       distance = c(w, b)),
            out_file, row.names = FALSE)
cat(sprintf("  MPrESS OK: within=%d between=%d\n", length(w), length(b)))
