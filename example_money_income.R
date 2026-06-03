# =============================================================================
# example_money_income.R
# Time-Varying Granger Causality — Money–Income Relationship
# =============================================================================
#
# This script replicates the money–income causality analysis following the
# approach of Shi, Hurn & Phillips (2020), using the tvgc() function adapted
# from the Stata code of Otero & Smith (2021).
#
# Research question:
#   Does money (M1) Granger-cause income (GDP) in the United States?
#   Does the causal relationship change over time?
#
# Data:
#   money-income-data.rmd — quarterly U.S. time series including:
#     li  : log real GDP (income)
#     lm1 : log M1 money stock
#     lp  : log price level (GDP deflator)
#     r   : short-term interest rate
#
# Reference:
#   Shi, S., Hurn, S. & Phillips, P. C. B. (2020). Causal change detection
#   in possibly integrated systems: revisiting the money–income relationship.
#   Journal of Financial Econometrics, 18(1), 158–180.
#   https://doi.org/10.1093/jjfinec/nbz004
#
# =============================================================================


# -----------------------------------------------------------------------------
# 0. Dependencies
# -----------------------------------------------------------------------------

library(dplyr)    # data preparation
# ggplot2 is loaded internally by tvgc_plot() — no need to attach it here

# Load tvgc() and tvgc_plot() from the same directory
source("tvgc.R")


# -----------------------------------------------------------------------------
# 1. Load and prepare data
# -----------------------------------------------------------------------------

data <- readRDS("../Data/money-income-data.rds")

# Select only the four variables needed for this analysis.
# Column order matters:
#   - Column 1 (li)  = dependent variable (Y). The test asks whether
#                      the remaining variables Granger-cause li.
#   - Columns 2–4    = RHS variables (candidate causes): lm1, lp, r.
#
# Remove the date column BEFORE calling as.matrix(). Keeping a date column
# causes as.matrix() to coerce all values to character, which will produce
# an error inside tvgc(). Dates are passed separately to tvgc_plot() below.

final_data <- data |>
  select(li, lm1, lp, r) |>
  as.matrix()

# Quick check: dimensions and column names
dim(final_data)           # T x 4
colnames(final_data)      # "li" "lm1" "lp" "r"


# -----------------------------------------------------------------------------
# 2. Run the TVGC test
# -----------------------------------------------------------------------------
#
# Key parameter choices:
#
#   p = 2          VAR lag order. The test restricts only these p lags under H0.
#
#   d = 1          Toda–Yamamoto lag augmentation order. Set d equal to the
#                  maximum suspected integration order of the system (here I(1)).
#                  This ensures valid Wald inference without prior unit root
#                  testing or cointegration restrictions.
#
#   trend = TRUE   A linear trend is included in the VAR. Appropriate when
#                  the series exhibit deterministic trends (common in levels).
#
#   window = 72    Minimum sub-sample window width (in observations = quarters).
#                  72 quarters = 18 years. Following Shi et al. (2020), the
#                  window should be large enough to estimate the VAR reliably
#                  but small enough to detect localised causality episodes.
#
#   sizecontrol = 60
#                  Bootstrap size-control parameter. The bootstrap sub-sample
#                  has width sizecontrol * 6 = 360 observations. Larger values
#                  improve size control at the cost of computation time.
#
#   boot = 199     Number of bootstrap replications for critical values.
#                  Use boot >= 499 for publication-quality results.
#
#   robust = TRUE  Use heteroskedasticity-consistent (HC/sandwich) standard
#                  errors for the Wald statistic. Recommended for financial
#                  and macroeconomic time series with heteroskedastic errors.

tvgc_1 <- tvgc(
  data        = final_data,
  p           = 2,
  d           = 1,
  trend       = TRUE,
  window      = 72,
  sizecontrol = 60,
  boot        = 499,
  seed        = 123,
  robust      = TRUE
)


# -----------------------------------------------------------------------------
# 3. Inspect results
# -----------------------------------------------------------------------------
#
# tvgc() returns a list. The main output is tvgc_1$stats: a matrix with one
# row per RHS variable and three columns, one per window scheme:
#
#   Max_Wald_FE  — maximum Wald statistic, Forward Expanding Window
#   Max_Wald_RO  — maximum Wald statistic, Rolling Window
#   Max_Wald_RE  — maximum Wald statistic, Recursive Evolving Window
#
# Note: li (column 1, the dependent variable) does NOT appear in the results.
# Only the RHS variables are listed — the ones being tested as causes of li.

rownames(tvgc_1$stats)    # "lm1" "lp" "r"
tvgc_1$stats              # test statistics for each variable and scheme

# Bootstrap critical values at 90%, 95%, and 99%:
tvgc_1$cv95               # compare against tvgc_1$stats to assess significance

# The raw Wald statistic time series (length Nt = T - window + 1) are stored
# in tvgc_1$mats, accessible by variable name and scheme:
#   tvgc_1$mats$lm1$FE   — forward expanding series for lm1
#   tvgc_1$mats$lm1$RO   — rolling window series for lm1
#   tvgc_1$mats$lm1$RE   — recursive evolving series for lm1


# -----------------------------------------------------------------------------
# 4. Plot results
# -----------------------------------------------------------------------------
#
# tvgc_plot() requires the FULL date vector (length T), not a trimmed version.
# The function shifts the index internally so that each Wald statistic is
# aligned with the END DATE of its window.
#
# Do not subset or trim dates_vec before passing it — that would shift the
# x-axis and misalign the shaded rejection periods.

date <- as.Date(data$date)    # length T — full sample
length(date) == nrow(final_data)   # must be TRUE

# --- Plot selected variables and schemes ------------------------------------
#
# vars    : subset of RHS variables to plot. NULL = all.
# All three window schemes (FE, RW, RE) are always plotted automatically.
# Use `vars` to restrict which RHS variables are shown.
#
# Each variable x scheme produces one named plot in the returned list:
#   plots$lm1_FE   — Forward Expanding Window
#   plots$lm1_RW   — Rolling Window
#   plots$lm1_RE   — Recursive Evolving Window
#
# The plot shows:
#   - Blue area    : Wald statistic over time
#   - Orange line  : bootstrap critical value at the chosen level (pct)
#   - Grey shading : periods where Wald > critical value (rejection of H0,
#                    i.e. evidence of Granger causality at that point in time)

# --- Plot selected variables (all three schemes each) -----------------------

plots <- tvgc_plot(
  res   = tvgc_1,
  dates = date,
  pct   = 95,                  # 95% bootstrap critical value
  vars  = c("lm1", "lp")      # M1 and price level — 6 plots total
)

# Access individual plots
plots$lm1_FE    # Does M1 Granger-cause income? — Forward Expanding
plots$lm1_RW    # Does M1 Granger-cause income? — Rolling Window
plots$lm1_RE    # Does M1 Granger-cause income? — Recursive Evolving
plots$lp_FE
plots$lp_RW
plots$lp_RE

# --- Plot all variables (3 variables x 3 schemes = 9 plots) ----------------

plots_all <- tvgc_plot(tvgc_1, dates = date, pct = 95)

plots_all$lm1_FE
plots_all$lm1_RW
plots_all$lm1_RE
plots_all$lp_FE
plots_all$lp_RW
plots_all$lp_RE
plots_all$r_FE
plots_all$r_RW
plots_all$r_RE

# --- Save plots to disk -----------------------------------------------------

ggplot2::ggsave("lm1_FE.pdf", plots$lm1_FE, width = 8, height = 5)
ggplot2::ggsave("lm1_RW.pdf", plots$lm1_RW, width = 8, height = 5)
ggplot2::ggsave("lm1_RE.pdf", plots$lm1_RE, width = 8, height = 5)


# -----------------------------------------------------------------------------
# 6. Testing causality in both directions
# -----------------------------------------------------------------------------
#
# The specification above tests: do lm1, lp, r cause li?
# To test the reverse direction (does li cause lm1?), swap the column order
# so that lm1 becomes the dependent variable (column 1) and li moves to a
# RHS column.

final_data_rev <- data |>
  select(lm1, li, lp, r) |>   # lm1 is now the dependent variable
  as.matrix()

tvgc_2 <- tvgc(
  data        = final_data_rev,
  p           = 2,
  d           = 1,
  trend       = TRUE,
  window      = 72,
  sizecontrol = 60,
  boot        = 199,
  robust      = TRUE
)

# Does li Granger-cause lm1?
tvgc_2$stats
plots_rev <- tvgc_plot(tvgc_2, dates = date, pct = 95,
                       vars = "li")   # produces li_FE, li_RW, li_RE
