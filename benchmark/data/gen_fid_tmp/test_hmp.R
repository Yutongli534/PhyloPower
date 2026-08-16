suppressMessages({ library(HMP); library(vegan) })
pilot <- round(as.matrix(read.csv("pilot_1346697099_table.csv", row.names = 1, check.names = FALSE)))
grp <- read.csv("pilot_1346697099_group.csv", stringsAsFactors = FALSE)
g <- "control"
g_samples <- grp$sample_id[grp$group == g]
g_table <- pilot[, g_samples, drop = FALSE]
g_table <- g_table[rowSums(g_table) > 0, , drop = FALSE]
g_table_t <- t(g_table)
cat("g_table_t dim:", dim(g_table_t), "\n")
fit <- dirmult(g_table_t)
cat("fit$pi length:", length(fit$pi), "\n")
K <- ncol(g_table_t)
n_r <- max(1000, round(mean(rowSums(g_table_t))))
sim_data <- Dirichlet.multinomial(fit, 10, n_r)
cat("Dirichlet.multinomial output dim:", dim(sim_data), "\n")
