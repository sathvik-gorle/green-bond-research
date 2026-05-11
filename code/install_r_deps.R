#!/usr/bin/env Rscript
# Install CRAN packages needed for knitr/paper build and volatility models.

pkgs <- c(
  "rugarch", "rmgarch", "ggplot2", "yaml",
  "knitr", "kableExtra", "dplyr", "tidyr",
  "scales", "gridExtra", "rmarkdown"
)

miss <- pkgs[!pkgs %in% rownames(installed.packages())]
if (length(miss)) {
  install.packages(miss, repos = "https://cloud.r-project.org")
} else {
  message("All listed R packages already installed.")
}
