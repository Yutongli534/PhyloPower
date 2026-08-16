suppressMessages({ library(dirmult); library(vegan) })
pilot <- round(as.matrix(read.csv("pilot_1346697099_table.csv", row.names = 1, check.names = FALSE)))
grp <- read.csv("pilot_1346697099_group.csv", stringsAsFactors = FALSE)
g <- "control"
g_samples <- grp$sample_id[grp$group == g]
g_table <- pilot[, g_samples, drop = FALSE]
g_table <- g_table[rowSums(g_table) > 0, , drop = FALSE]
cat("g_table dim:", dim(g_table), "\n")
fit <- dirmult(g_table)
cat("fit$pi length:", length(fit$pi), "\n")
cat("K = nrow(g_table):", nrow(g_table), "\n")
sim <- simPop(J = 10, K = nrow(g_table), n = 1000, pi = fit$pi, theta = fit$theta)
cat("sim$data dim:", dim(sim$data), "\n")
