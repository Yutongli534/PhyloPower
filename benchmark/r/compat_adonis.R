# Compatibility shim: vegan >= 2.7 made adonis() defunct, but micropower 0.4
# and mpress 1.0.0 both call the legacy adonis() interface (returning $aov.tab
# with R2 / Pr(>F)), with formulas referencing caller-frame variables
# (e.g. `as.dist(dm) ~ colnames(dm)` or `dist.val ~ group`). adonis2() is
# numerically identical for a single-factor PERMANOVA design, so we evaluate
# both sides of the incoming formula in its own environment and delegate.
adonis2_shim <- function(formula, data = NULL, permutations = 999, ...) {
  fenv <- environment(formula)
  if (is.null(fenv)) fenv <- parent.frame()
  y <- eval(formula[[2]], envir = fenv, enclos = parent.frame())
  group <- if (!is.null(data)) {
    eval(formula[[3]], envir = as.data.frame(data), enclos = fenv)
  } else {
    eval(formula[[3]], envir = fenv, enclos = parent.frame())
  }
  res <- vegan::adonis2(y ~ .benchmark_group,
                        data = data.frame(.benchmark_group = group),
                        permutations = permutations, ...)
  out <- list(aov.tab = as.data.frame(res))
  class(out) <- "adonis"
  out
}
# NOTE: must run BEFORE library(micropower)/library(mpress) so their import
# environments pick up the shimmed binding instead of the defunct original.
loadNamespace("vegan")
assignInNamespace("adonis", adonis2_shim, ns = "vegan")
