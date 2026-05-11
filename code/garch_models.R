#!/usr/bin/env Rscript
# =============================================================================
# C3: GARCH models — EGARCH-X on difference series + DCC-GARCH
#
# Primary target: r_diff_BGRN_LQD (green minus corporate return differential)
# Also fit on r_BGRN and r_LQD separately for comparison
# DCC-GARCH: green vs corporate vs carbon vs clean energy
#
# Exports: parameter tables, conditional volatility series, plots
# =============================================================================

suppressPackageStartupMessages({
  library(rugarch)
  library(rmgarch)
  library(ggplot2)
  library(yaml)
})

cat("========================================\n")
cat("  C3  GARCH VOLATILITY MODELS\n")
cat("========================================\n\n")

# --- Config -------------------------------------------------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  normalizePath(getwd())
}
CODE_DIR <- normalizePath(get_script_dir())
REPO_ROOT <- normalizePath(file.path(CODE_DIR, ".."))
cfg  <- yaml.load_file(file.path(CODE_DIR, "configs", "config.yaml"))

features_path <- file.path(REPO_ROOT, cfg$paths$processed, "features.csv")
out_tables    <- file.path(REPO_ROOT, cfg$paths$tables)
out_figures   <- file.path(REPO_ROOT, cfg$paths$figures)
dir.create(out_tables, recursive = TRUE, showWarnings = FALSE)
dir.create(out_figures, recursive = TRUE, showWarnings = FALSE)

# Canonical names for vxreg* rows in exported CSV (matches default exog order above)
VXREG_MAP_UNHEDGED <- c(
  vxreg1 = "VIX_z",
  vxreg2 = "delta_IG_OAS",
  vxreg3 = "r_KRBN",
  vxreg4 = "delta_10y",
  vxreg5 = "CPU_z"
)
VXREG_MAP_HEDGED <- c(
  vxreg1 = "VIX_z",
  vxreg2 = "delta_IG_OAS",
  vxreg3 = "r_KRBN",
  vxreg4 = "CPU_z"
)

# Friendly names for EGARCH-X vxreg coefficients (order follows ext_mat columns)
decorate_egarch_vxregs <- function(params_df, ext_col_names) {
  df <- as.data.frame(params_df)
  if (!("Parameter" %in% names(df))) {
    df$Parameter <- rownames(df)
  }
  reg <- df$Parameter
  out <- as.character(reg)
  for (i in seq_along(out)) {
    if (grepl("^vxreg[0-9]+$", out[i])) {
      j <- suppressWarnings(as.integer(sub("^vxreg", "", out[i])))
      if (!is.na(j) && j >= 1L && j <= length(ext_col_names)) {
        out[i] <- ext_col_names[j]
      }
    }
  }
  df$Variable <- out
  df
}

# --- Load data ----------------------------------------------------------------
cat("Loading features from:", features_path, "\n")
df <- read.csv(features_path)
if ("Date" %in% names(df)) {
  df$Date <- as.Date(df$Date)
} else if ("X" %in% names(df)) {
  df$Date <- as.Date(df$X)
  df$X <- NULL
}
cat(sprintf("  Loaded %d obs x %d cols\n\n", nrow(df), ncol(df)))

# =============================================================================
# A.  ARCH EFFECT TESTS
# =============================================================================

cat("--- A. ARCH effect tests (Engle LM) ---\n")

test_assets <- c("r_BGRN", "r_LQD", "r_KRBN", "r_ICLN", "r_SPY",
                 "r_diff_BGRN_LQD", "r_diff_GRNB_LQD")
test_assets <- test_assets[test_assets %in% names(df)]

arch_tests <- data.frame()
for (a in test_assets) {
  r <- df[[a]]
  r <- r[!is.na(r)]
  if (length(r) < 30) next

  resid2 <- (r - mean(r))^2
  n <- length(resid2)
  lags <- 5
  y <- resid2[(lags + 1):n]
  X <- do.call(cbind, lapply(1:lags, function(l) resid2[(lags + 1 - l):(n - l)]))
  X <- cbind(1, X)
  fit <- lm(y ~ X - 1)
  R2 <- summary(fit)$r.squared
  LM_stat <- n * R2
  LM_p <- 1 - pchisq(LM_stat, df = lags)

  arch_tests <- rbind(arch_tests, data.frame(
    Asset = a, LM_stat = round(LM_stat, 2), p_value = round(LM_p, 4),
    ARCH_present = ifelse(LM_p < 0.05, "YES", "NO")
  ))
  cat(sprintf("  %-25s  LM=%.2f  p=%.4f  %s\n", a, LM_stat, LM_p,
              ifelse(LM_p < 0.05, "ARCH present", "No ARCH")))
}

write.csv(arch_tests, file.path(out_tables, "arch_tests.csv"), row.names = FALSE)

# =============================================================================
# B.  UNIVARIATE GARCH: GARCH(1,1), EGARCH(1,1), GJR for key series
# =============================================================================

cat("\n--- B. Fitting univariate GARCH models ---\n")

model_specs <- list(
  sGARCH   = list(model = "sGARCH",   garchOrder = c(1, 1)),
  eGARCH   = list(model = "eGARCH",   garchOrder = c(1, 1)),
  gjrGARCH = list(model = "gjrGARCH", garchOrder = c(1, 1))
)

garch_dists <- cfg$garch$distributions  # ["norm", "std", "sstd"]

target_series <- c("r_BGRN", "r_LQD", "r_diff_BGRN_LQD")
target_series <- target_series[target_series %in% names(df)]

best_models  <- list()
all_ic       <- data.frame()

for (a in target_series) {
  cat(sprintf("\n  Asset: %s\n", a))
  r <- df[[a]]
  r <- r[!is.na(r)]

  best_aic <- Inf
  best_fit <- NULL
  best_name <- ""

  for (spec_name in names(model_specs)) {
    for (dist in garch_dists) {
      spec <- ugarchspec(
        variance.model     = model_specs[[spec_name]],
        mean.model         = list(armaOrder = c(0, 0), include.mean = TRUE),
        distribution.model = dist
      )
      fit <- tryCatch(ugarchfit(spec, r, solver = "hybrid"), error = function(e) NULL)
      if (!is.null(fit)) {
        aic_val <- infocriteria(fit)[1]
        bic_val <- infocriteria(fit)[2]
        all_ic <- rbind(all_ic, data.frame(
          Asset = a, Model = spec_name, Dist = dist,
          AIC = round(aic_val, 5), BIC = round(bic_val, 5)
        ))
        if (aic_val < best_aic) {
          best_aic <- aic_val
          best_fit <- fit
          best_name <- paste0(spec_name, "_", dist)
        }
      }
    }
  }

  best_models[[a]] <- best_fit
  cat(sprintf("    Best: %s (AIC=%.5f)\n", best_name, best_aic))
}

write.csv(all_ic, file.path(out_tables, "garch_model_comparison.csv"), row.names = FALSE)

# =============================================================================
# C.  EGARCH-X: Exogenous regressors in variance equation
# =============================================================================

cat("\n--- C. EGARCH-X (exogenous in variance) ---\n")

# Target: r_diff_BGRN_LQD (the green-minus-corporate difference)
# Exogenous: VIX_z, delta_IG_OAS, r_KRBN, delta_10y, CPU_z
egarchx_cols <- c("r_diff_BGRN_LQD", "VIX_z", "delta_IG_OAS", "r_KRBN", "delta_10y", "CPU_z")
egarchx_cols <- egarchx_cols[egarchx_cols %in% names(df)]

if ("r_diff_BGRN_LQD" %in% egarchx_cols && length(egarchx_cols) >= 3) {

  egarchx_df <- df[, egarchx_cols, drop = FALSE]
  egarchx_df <- egarchx_df[complete.cases(egarchx_df), ]

  y_diff   <- egarchx_df$r_diff_BGRN_LQD
  ext_cols <- setdiff(egarchx_cols, "r_diff_BGRN_LQD")
  ext_mat  <- as.matrix(egarchx_df[, ext_cols])

  cat(sprintf("  EGARCH-X sample: %d obs, exogenous: %s\n",
              nrow(egarchx_df), paste(ext_cols, collapse = ", ")))

  # EGARCH-X(1,1)-t
  egarchx_spec <- ugarchspec(
    variance.model = list(model = "eGARCH", garchOrder = c(1, 1),
                           external.regressors = ext_mat),
    mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"
  )

  egarchx_fit <- tryCatch(
    ugarchfit(egarchx_spec, y_diff, solver = "hybrid"),
    error = function(e) { cat(sprintf("  EGARCH-X fit error: %s\n", e$message)); NULL }
  )

  if (!is.null(egarchx_fit)) {
    cat("  EGARCH-X(1,1)-t fitted successfully\n")
    cat("  Parameter estimates:\n")
    print(egarchx_fit@fit$matcoef)

    # Extract conditional volatility
    cond_vol_diff <- as.numeric(sigma(egarchx_fit)) * sqrt(252)

    # Also fit base EGARCH without exogenous for comparison
    base_spec <- ugarchspec(
      variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
      mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
      distribution.model = "std"
    )
    base_fit <- tryCatch(ugarchfit(base_spec, y_diff, solver = "hybrid"), error = function(e) NULL)

    if (!is.null(base_fit)) {
      cat(sprintf("  Base EGARCH AIC=%.5f  vs  EGARCH-X AIC=%.5f\n",
                  infocriteria(base_fit)[1], infocriteria(egarchx_fit)[1]))
    }

    # Save EGARCH-X parameter table
    params <- as.data.frame(egarchx_fit@fit$matcoef)
    params$Parameter <- rownames(params)
    params <- decorate_egarch_vxregs(params, ext_cols)
    params$Variable <- ifelse(
      params$Parameter %in% names(VXREG_MAP_UNHEDGED),
      VXREG_MAP_UNHEDGED[params$Parameter],
      params$Variable
    )
    write.csv(params, file.path(out_tables, "egarchx_parameters.csv"), row.names = FALSE)
  }
} else {
  cat("  Skipping EGARCH-X: required columns not available\n")
  egarchx_fit <- NULL
}

# =============================================================================
# C2.  EGARCH-X on rate-hedged differential (Upgrade 1)
# =============================================================================

cat("\n--- C2. EGARCH-X on r_diff_hedged ---\n")

egarchx_hedged_cols <- c("r_diff_hedged", "VIX_z", "delta_IG_OAS", "r_KRBN", "CPU_z")
egarchx_hedged_cols <- egarchx_hedged_cols[egarchx_hedged_cols %in% names(df)]

egarchx_hedged_fit <- NULL
if ("r_diff_hedged" %in% egarchx_hedged_cols && length(egarchx_hedged_cols) >= 3) {

  eh_df <- df[, egarchx_hedged_cols, drop = FALSE]
  eh_df <- eh_df[complete.cases(eh_df), ]

  y_hedged   <- eh_df$r_diff_hedged
  ext_h_cols <- setdiff(egarchx_hedged_cols, "r_diff_hedged")
  ext_h_mat  <- as.matrix(eh_df[, ext_h_cols])

  cat(sprintf("  EGARCH-X hedged sample: %d obs, exogenous: %s\n",
              nrow(eh_df), paste(ext_h_cols, collapse = ", ")))

  eh_spec <- ugarchspec(
    variance.model = list(model = "eGARCH", garchOrder = c(1, 1),
                           external.regressors = ext_h_mat),
    mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"
  )

  egarchx_hedged_fit <- tryCatch(
    ugarchfit(eh_spec, y_hedged, solver = "hybrid"),
    error = function(e) { cat(sprintf("  EGARCH-X hedged fit error: %s\n", e$message)); NULL }
  )

  if (!is.null(egarchx_hedged_fit)) {
    cat("  EGARCH-X(1,1)-t on hedged diff fitted successfully\n")
    cat("  Parameter estimates:\n")
    print(egarchx_hedged_fit@fit$matcoef)

    # Save parameter table
    params_h <- as.data.frame(egarchx_hedged_fit@fit$matcoef)
    params_h$Parameter <- rownames(params_h)
    params_h <- decorate_egarch_vxregs(params_h, ext_h_cols)
    params_h$Variable <- ifelse(
      params_h$Parameter %in% names(VXREG_MAP_HEDGED),
      VXREG_MAP_HEDGED[params_h$Parameter],
      params_h$Variable
    )
    write.csv(params_h, file.path(out_tables, "egarchx_hedged_parameters.csv"),
              row.names = FALSE)

    # Compare AIC
    if (!is.null(egarchx_fit)) {
      cat(sprintf("  Unhedged EGARCH-X AIC=%.5f  vs  Hedged EGARCH-X AIC=%.5f\n",
                  infocriteria(egarchx_fit)[1], infocriteria(egarchx_hedged_fit)[1]))
    }
  }
} else {
  cat("  r_diff_hedged not available. Skipping hedged EGARCH-X.\n")
}

# =============================================================================
# D.  DCC-GARCH: Dynamic conditional correlations
# =============================================================================

cat("\n--- D. DCC-GARCH ---\n")

dcc_cols <- c("r_BGRN", "r_LQD", "r_KRBN", "r_ICLN")
dcc_cols <- dcc_cols[dcc_cols %in% names(df)]

if (length(dcc_cols) >= 3) {

  dcc_df <- df[, dcc_cols, drop = FALSE]
  dcc_df <- dcc_df[complete.cases(dcc_df), ]
  dcc_mat <- as.matrix(dcc_df)

  cat(sprintf("  DCC sample: %d obs, assets: %s\n", nrow(dcc_df), paste(dcc_cols, collapse = ", ")))

  # Univariate GARCH(1,1)-t spec for each margin
  uspec <- ugarchspec(
    variance.model     = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model         = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "std"
  )
  multi_spec <- multispec(replicate(length(dcc_cols), uspec))
  dcc_spec   <- dccspec(uspec = multi_spec, dccOrder = c(1, 1), distribution = "mvt")

  dcc_fit <- tryCatch(
    dccfit(dcc_spec, dcc_mat),
    error = function(e) { cat(sprintf("  DCC fit error: %s\n", e$message)); NULL }
  )

  if (!is.null(dcc_fit)) {
    cat("  DCC-GARCH fitted successfully\n")

    # Extract time-varying correlations
    Rt <- rcor(dcc_fit)
    T_len <- dim(Rt)[3]
    dates_dcc <- df$Date[match(rownames(dcc_df), rownames(df))]
    if (length(dates_dcc) < T_len) {
      dates_dcc <- tail(df$Date[!is.na(df[[dcc_cols[1]]])], T_len)
    }

    dcc_corr <- data.frame(Date = dates_dcc[1:T_len])
    for (i in 2:length(dcc_cols)) {
      pair_name <- paste0("Green_", gsub("r_", "", dcc_cols[i]))
      dcc_corr[[pair_name]] <- Rt[1, i, ]
    }

    # Plot DCC correlations
    dcc_long <- reshape(dcc_corr, direction = "long",
                         varying = names(dcc_corr)[-1],
                         v.names = "Correlation",
                         timevar = "Pair",
                         times = names(dcc_corr)[-1])

    p_dcc <- ggplot(dcc_long, aes(x = Date, y = Correlation, color = Pair)) +
      geom_line(linewidth = 0.5) +
      geom_hline(yintercept = 0, linetype = "dashed", alpha = 0.5) +
      labs(title = "DCC-GARCH: Dynamic Correlations with Green Bonds",
           x = "", y = "Conditional Correlation") +
      theme_minimal() +
      theme(legend.position = "bottom")
    ggsave(file.path(out_figures, "dcc_correlations.pdf"), p_dcc, width = 10, height = 5)
    cat("  Saved dcc_correlations.pdf\n")
  }
} else {
  cat("  Not enough DCC columns. Skipping.\n")
  dcc_fit <- NULL
}

# =============================================================================
# E.  Conditional volatility plot
# =============================================================================

cat("\n--- E. Conditional volatility plots ---\n")

# Collect conditional volatilities from best univariate models
vol_df <- data.frame(Date = df$Date)
for (a in names(best_models)) {
  if (!is.null(best_models[[a]])) {
    r <- df[[a]]
    valid_idx <- which(!is.na(r))
    vol_series <- as.numeric(sigma(best_models[[a]])) * sqrt(252)
    vol_col <- rep(NA, nrow(df))
    vol_col[valid_idx] <- vol_series
    vol_df[[a]] <- vol_col
  }
}

# Add EGARCH-X conditional vol for diff series
if (!is.null(egarchx_fit)) {
  valid_idx <- which(complete.cases(df[, egarchx_cols]))
  vol_egx <- rep(NA, nrow(df))
  vol_egx[valid_idx] <- as.numeric(sigma(egarchx_fit)) * sqrt(252)
  vol_df[["EGARCHX_diff"]] <- vol_egx
}

vol_df <- vol_df[complete.cases(vol_df[, -1, drop = FALSE]), ]

if (nrow(vol_df) > 0) {
  vol_long <- reshape(vol_df, direction = "long",
                       varying = names(vol_df)[-1],
                       v.names = "AnnVol",
                       timevar = "Asset",
                       times = names(vol_df)[-1])

  p_vol <- ggplot(vol_long, aes(x = Date, y = AnnVol * 100, color = Asset)) +
    geom_line(linewidth = 0.4, alpha = 0.8) +
    labs(title = "Annualized Conditional Volatility",
         x = "", y = "Volatility (%)") +
    theme_minimal() +
    theme(legend.position = "bottom", legend.text = element_text(size = 8))
  ggsave(file.path(out_figures, "conditional_volatility.pdf"), p_vol, width = 10, height = 5)
  cat("  Saved conditional_volatility.pdf\n")
}

# =============================================================================
# F.  Save all results
# =============================================================================

cat("\n--- F. Saving GARCH results ---\n")

save(best_models, all_ic, arch_tests,
     egarchx_fit, egarchx_hedged_fit, dcc_fit,
     file = file.path(REPO_ROOT, cfg$paths$processed, "garch_results.RData"))
cat("  Saved garch_results.RData\n")

cat("\ngarch_models.R complete.\n")
