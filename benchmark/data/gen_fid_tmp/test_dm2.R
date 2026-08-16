suppressMessages({ library(dirmult); library(vegan) })
pilot <- round(as.matrix(read.csv("pilot_1346697099_table.csv", row.names = 1, check.names = FALSE)))
grp <- read.csv("pilot_1346697099_group.csv", stringsAsFactors = FALSE)
g <- "control"
g_samples <- grp$sample_id[grp$group == g]
g_table <- pilot[, g_samples, drop = FALSE]
g_table <- g_table[rowSums(g_table) > 0, , drop = FALSE]
g_table_t <- t(g_table)  # SAMPLES x FEATURES for dirmult
cat("g_table_t dim (samples x features):", dim(g_table_t), "\n")
fit <- dirmult(g_table_t)
cat("fit$pi length:", length(fit$pi), "\n")
K <- ncol(g_table_t)
n_reads <- max(1000, round(mean(rowSums(g_table_t))))
cat("K:", K, "n_reads:", n_reads, "\n")
sim <- simPop(J = 10, K = K, n = n_reads, pi = fit$pi, theta = fit$theta)
cat("sim$data dim:", dim(sim$data), "\n")
# sim$data should be SAMPLES x FEATURES (J x K)
