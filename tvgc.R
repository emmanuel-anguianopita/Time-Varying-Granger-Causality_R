# =============================================================================
# tvgc.R — Time-Varying Granger Causality Test
# =============================================================================
#
# Implementation of the time-varying Granger causality (TVGC) test proposed by:
#
#   Otero, J. & Smith, J. (2021). "Testing for Granger non-causality in
#   heterogeneous panels." Journal of Applied Econometrics, 36(7), 858–876.
#   https://doi.org/10.1002/jae.2831
#
# The original test is based on the framework of:
#
#   Toda, H. Y. & Yamamoto, T. (1995). "Statistical inference in vector
#   autoregressions with possibly integrated processes." Journal of
#   Econometrics, 66(1–2), 225–250.
#   https://doi.org/10.1016/0304-4076(94)01616-8
#
#   Dolado, J. J. & Lütkepohl, H. (1996). "Making Wald tests work for
#   cointegrated VAR systems." Econometric Reviews, 15(4), 369–386.
#   https://doi.org/10.1080/07474939608800362
#
# The three window schemes (forward-expanding, rolling, recursive-expanding)
# follow the exposition in:
#
#   Rossi, B. (2005). "Optimal tests for nested model selection with
#   underlying parameter instability." Econometric Theory, 21(5), 962–990.
#
# Bootstrap critical values use the size-control approach described in the
# Otero & Smith (2021) supplementary Stata code, adapted here to R.
#
# -----------------------------------------------------------------------------
# Adaptation notes
# -----------------------------------------------------------------------------
#
# This R implementation adapts the original Otero & Smith (2021) Stata code
# with the following changes:
#
#   1. Three window schemes are computed simultaneously over a single Nt x Nt
#      Wald-statistic matrix per RHS variable, rather than three separate loops.
#      - Row 1        → Forward-Expanding (FE): fixed start t=1, growing end.
#      - Diagonal     → Rolling (RO): fixed-width window of size `window`.
#      - Col max(1:c) → Recursive-Expanding (RE): max over all valid starts
#                       up to column c, consistent with Rossi (2005).
#
#   2. Lag-augmentation follows Toda & Yamamoto (1995) / Dolado & Lütkepohl
#      (1996): the VAR is estimated with p + d lags but only the first p are
#      restricted under H0, where d = integration order of the system.
#
#   3. Bootstrap residuals are resampled jointly across equations (row-wise)
#      to preserve contemporaneous cross-equation correlation, which the
#      original Stata code does not enforce explicitly.
#
#   4. Parallelism is implemented via parallel::mclapply over the outer
#      dimension of each stage (rows in Stage 1, variables in Stage 2).
#      Windows falls back to sequential automatically.
#
#   5. Heteroskedasticity-consistent (HC) standard errors are optionally
#      available via the `robust` argument (sandwich-type variance estimator).
#
# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
#
#   Required : none beyond base R
#   Suggested: ggplot2 (for tvgc_plot), parallel (for cores > 1)
#
# -----------------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------------
#
#   # Basic usage (sequential):
#   res <- tvgc(data_mat, p = 2, d = 1, boot = 499, seed = 42)
#   tvgc_plot(res, dates = dates_vec, pct = 95)
#
#   # Parallel (Mac / Linux):
#   res <- tvgc(data_mat, p = 2, d = 1, boot = 499, seed = 42,
#               cores = parallel::detectCores() - 1L)
#
#   # data_mat must be a numeric matrix or data.frame with NO date column.
#   # Column 1 = dependent variable (Y); columns 2:K = RHS variables (X).
#   # Pass the date vector separately to tvgc_plot() via the `dates` argument.
#
# -----------------------------------------------------------------------------
# Author
# -----------------------------------------------------------------------------
#
#   Adapted by J. E. Anguiano Pita
#   SECIHTI – Universidad de Guadalajara (CUCEA / DEEC)
#   Based on Otero & Smith (2021) and their accompanying Stata implementation.
#
# =============================================================================


# =============================================================================
#' Time-Varying Granger Causality Test
#'
#' @description
#' Tests whether each RHS variable Granger-causes the first (dependent) column
#' of \code{data}, using three window schemes simultaneously:
#'
#' \describe{
#'   \item{Forward-Expanding (FE)}{Start fixed at \eqn{t = 1}, end grows from
#'     \eqn{t + w - 1} to \eqn{T}. Captures structural change occurring after
#'     the beginning of the sample.}
#'   \item{Rolling (RO)}{Fixed-width window of size \code{window} slides from
#'     \eqn{t = 1} to \eqn{t = T - w + 1}. Captures localised instability.}
#'   \item{Recursive-Expanding (RE)}{For each end-point \eqn{tt}, the maximum
#'     Wald statistic over all valid start points \eqn{t \le tt} is recorded.
#'     Captures any onset of Granger causality regardless of starting date.}
#' }
#'
#' The test statistic in each window is a Wald statistic for the joint
#' exclusion of the first \code{p} lags of each RHS variable from the equation
#' for column 1. Lag-augmentation by \code{d} extra lags (Toda & Yamamoto 1995;
#' Dolado & Lütkepohl 1996) ensures validity under unknown integration order.
#'
#' Bootstrap critical values are computed under H0 using residual resampling
#' on a sub-sample of size \code{sizecontrol * 6}, following the size-control
#' approach in the Otero & Smith (2021) supplementary Stata code. Residuals
#' are resampled jointly across equations to preserve contemporaneous
#' cross-equation correlation.
#'
#' @param data
#'   A numeric matrix or data.frame. The \strong{first column} is the dependent
#'   variable (Y); columns 2 through K are the RHS variables whose
#'   Granger-causal role is tested. \strong{Do not include a date column}:
#'   dates are irrelevant for lag computation (which is position-based) and
#'   will cause type errors. Pass dates separately to \code{tvgc_plot()}.
#' @param p
#'   Lag order of the VAR (integer, default \code{2}). Only the first \code{p}
#'   lags are restricted under H0; the extra \code{d} augmentation lags are
#'   always unrestricted.
#' @param d
#'   Maximum integration order of the system (integer, default \code{1}).
#'   Sets the number of extra lags added under the Toda–Yamamoto procedure.
#'   Use \code{d = 0} for a stationary system (no augmentation).
#' @param window
#'   Initial (minimum) window width in observations (integer). Defaults to
#'   \code{floor(0.2 * nrow(data))}. Must satisfy
#'   \code{p + d + 1 + window + sizecontrol - 1 <= nrow(data)}.
#' @param boot
#'   Number of bootstrap replications for critical values (integer,
#'   default \code{199}). At least \code{20} replications are enforced.
#'   Use \code{boot >= 499} for publication.
#' @param seed
#'   Integer random seed passed to \code{set.seed()} before the bootstrap
#'   (optional). Set for reproducibility.
#' @param sizecontrol
#'   Size-control parameter (integer, default \code{12}). Controls the
#'   sub-sample length used in the bootstrap: bootstrap window width is
#'   \code{sizecontrol * 6}; the sub-sample covers
#'   \code{sizecontrol + sizecontrol * 6 - 1} observations. Larger values
#'   increase bootstrap accuracy at the cost of computation time.
#' @param trend
#'   Logical. If \code{TRUE}, a linear time trend is included in the VAR
#'   (default \code{FALSE}).
#' @param robust
#'   Logical. If \code{TRUE}, heteroskedasticity-consistent (HC / sandwich)
#'   standard errors are used for the Wald statistic (default \code{FALSE}).
#' @param cores
#'   Number of parallel workers (integer, default \code{1} = sequential).
#'   Uses \code{parallel::mclapply} (fork-based). On Windows this argument is
#'   ignored and the function runs sequentially with a message. On Mac/Linux,
#'   set to \code{parallel::detectCores() - 1L} for maximum parallelism.
#'   Progress bars are suppressed in parallel mode.
#'
#' @return
#' Invisibly, a named list with the following elements:
#' \describe{
#'   \item{\code{stats}}{(K-1) x 3 numeric matrix of test statistics.
#'     Rows = RHS variables; columns = \code{Max_Wald_FE}, \code{Max_Wald_RO},
#'     \code{Max_Wald_RE}.}
#'   \item{\code{cv90}, \code{cv95}, \code{cv99}}{(K-1) x 3 matrices of
#'     bootstrap critical values at the 90th, 95th, and 99th percentiles,
#'     with the same layout as \code{stats}.}
#'   \item{\code{mats}}{Named list of (K-1) elements, one per RHS variable
#'     (e.g. \code{res$mats$MXN}). Each element is itself a named list with
#'     three numeric vectors of length \code{Nt = nrow(data) - window + 1}:
#'     \code{FE} (forward-expanding), \code{RO} (rolling), and
#'     \code{RE} (recursive-expanding). Each value is the Wald statistic
#'     at the corresponding end-of-window position.}
#'   \item{\code{p}, \code{d}, \code{window}, \code{boot},
#'     \code{sizecontrol}}{Parameters used in the run.}
#' }
#' The \code{stats} matrix also carries an attribute \code{"depname"} with the
#' name of the dependent variable (column 1 of \code{data}), used by
#' \code{tvgc_plot()}.
#'
#' @references
#' Otero, J. & Smith, J. (2021). Testing for Granger non-causality in
#' heterogeneous panels. \emph{Journal of Applied Econometrics}, 36(7),
#' 858–876. \doi{10.1002/jae.2831}
#'
#' Toda, H. Y. & Yamamoto, T. (1995). Statistical inference in vector
#' autoregressions with possibly integrated processes.
#' \emph{Journal of Econometrics}, 66(1–2), 225–250.
#' \doi{10.1016/0304-4076(94)01616-8}
#'
#' Dolado, J. J. & Lütkepohl, H. (1996). Making Wald tests work for
#' cointegrated VAR systems. \emph{Econometric Reviews}, 15(4), 369–386.
#' \doi{10.1080/07474939608800362}
#'
#' @examples
#' \dontrun{
#' # Simulate a bivariate I(1) system
#' set.seed(1)
#' n  <- 300
#' e  <- matrix(rnorm(n * 2), n, 2)
#' y  <- apply(e, 2, cumsum)
#' colnames(y) <- c("Y", "X")
#'
#' # Run TVGC (H0: X does not Granger-cause Y)
#' res <- tvgc(y, p = 2, d = 1, boot = 199, seed = 42)
#'
#' # Plot results
#' tvgc_plot(res, pct = 95)
#'
#' # With dates on x-axis
#' dates <- seq(as.Date("2000-01-01"), by = "month", length.out = n)
#' tvgc_plot(res, dates = dates, pct = 95)
#'
#' # Parallel (Mac / Linux)
#' res <- tvgc(y, p = 2, d = 1, boot = 499, seed = 42,
#'             cores = parallel::detectCores() - 1L)
#' }
#'
#' @seealso \code{\link{tvgc_plot}}
#' @export
# =============================================================================
tvgc <- function(data, p = 2, d = 1, window = NULL, boot = 199, seed = NULL,
                 sizecontrol = 12, trend = FALSE, robust = FALSE, cores = 1L) {
  
  data    <- as.matrix(data)
  storage.mode(data) <- "double"   # ensure numeric; prevents crossprod() type errors
  Tobs    <- nrow(data)
  K       <- ncol(data)
  nms     <- if (!is.null(colnames(data))) colnames(data) else paste0("V", seq_len(K))
  
  wwid      <- if (is.null(window) || window <= 0) floor(0.2 * Tobs) else as.integer(window)
  bootrepl  <- max(20L, as.integer(boot))
  sc        <- as.integer(sizecontrol)
  lag_start <- p + d + 1L   # first row with a complete lag block
  
  if (lag_start + wwid + sc - 1L > Tobs)
    stop("initial window + sizecontrol exceeds available observations")
  if (!is.null(seed)) set.seed(seed)
  
  # ---------- parallel setup ------------------------------------------------
  # parallel::mclapply uses process forking and is unavailable on Windows.
  on_windows <- .Platform$OS.type == "windows"
  n_cores    <- if (on_windows) 1L else max(1L, as.integer(cores))
  .lapply    <- if (n_cores > 1L) {
    function(X, FUN) parallel::mclapply(X, FUN, mc.cores = n_cores,
                                        mc.set.seed = TRUE)
  } else {
    lapply
  }
  if (on_windows && cores > 1L)
    message("Parallel execution is not supported on Windows; running sequentially.")
  
  # ---------- helpers -------------------------------------------------------
  
  # .lags: build a lagged copy of `mat` for each lag order in `lv`.
  # Returns a matrix of dimensions nrow(mat) x (ncol(mat) * length(lv));
  # pre-sample rows are filled with NA.
  .lags <- function(mat, lv) {
    n  <- nrow(mat); nc <- ncol(mat)
    out <- matrix(NA_real_, n, nc * length(lv))
    for (li in seq_along(lv)) {
      l <- lv[li]
      if (l < n)
        out[(l + 1L):n, ((li - 1L) * nc + 1L):(li * nc)] <-
          mat[1L:(n - l), , drop = FALSE]
    }
    out
  }
  
  # .buildX: assemble the OLS design matrix for a window indexed by `idx`.
  # Column layout: [Y_lag1 ... Y_lagp | Z_lag(p+1) ... Z_lag(p+d) | (trend,) 1]
  # All blocks are coerced to double to prevent type-mismatch in crossprod().
  .buildX <- function(xl, zl, idx) {
    n   <- length(idx)
    tau <- if (trend) {
      matrix(as.double(c(idx, rep(1L, n))), n, 2L)   # [t, 1]
    } else {
      matrix(1.0, n, 1L)                              # [1]
    }
    xl_sub <- matrix(as.double(xl[idx, , drop = FALSE]), n, ncol(xl))
    if (!is.null(zl)) {
      zl_sub <- matrix(as.double(zl[idx, , drop = FALSE]), n, ncol(zl))
      cbind(xl_sub, zl_sub, tau)
    } else {
      cbind(xl_sub, tau)
    }
  }
  
  # .wstat: compute the Wald statistic for H0: lags of each variable j
  # (j = 2..K) do not Granger-cause variable 1, within window (Y, Xf).
  #
  # Under the design layout in .buildX, variable j occupies columns
  # j, j+K, j+2K, ... across the p lag blocks (1-based indexing).
  # The restriction matrix R (p x ncol(Xf)) selects exactly those columns.
  #
  # Returns a length-K vector; W[1] is unused, W[j] is the statistic for
  # H0: variable j does not Granger-cause variable 1.
  .wstat <- function(Y, Xf, use_robust = FALSE) {
    ok  <- complete.cases(Y, Xf)
    Y   <- Y[ok, , drop = FALSE]
    Xf  <- Xf[ok, , drop = FALSE]
    n   <- sum(ok)
    nb  <- ncol(Xf)
    if (n <= nb) return(rep(NA_real_, K))
    
    y    <- Y[, 1L]
    XpXi <- tryCatch(solve(crossprod(Xf)), error = function(e) NULL)
    if (is.null(XpXi)) return(rep(NA_real_, K))
    
    b  <- drop(XpXi %*% crossprod(Xf, y))
    e2 <- drop(y - Xf %*% b)^2
    
    # Variance estimator: homoskedastic OLS or HC sandwich
    om <- if (!use_robust) {
      (sum(e2) / n) * XpXi
    } else {
      XpXi %*% (t(Xf) %*% (Xf * e2)) %*% XpXi
    }
    om <- (om + t(om)) / 2   # symmetrise to guard against floating-point drift
    
    W <- rep(NA_real_, K)
    for (j in 2L:K) {
      lag_cols_j <- j + (seq_len(p) - 1L) * K   # columns: j, j+K, j+2K, ...
      R  <- matrix(0, p, nb)
      for (k in seq_len(p)) R[k, lag_cols_j[k]] <- 1L
      Rb <- R %*% b
      Ri <- tryCatch(solve(R %*% om %*% t(R)), error = function(e) NULL)
      if (!is.null(Ri)) W[j] <- as.numeric(t(Rb) %*% Ri %*% Rb)
    }
    W
  }
  
  # ---------- pre-compute lag matrices --------------------------------------
  
  Xl <- .lags(data, seq_len(p))                              # lags 1..p
  Zl <- if (d > 0L) .lags(data, (p + 1L):(p + d)) else NULL # lags (p+1)..(p+d)
  
  # ---------- Stage 1: Wald statistics over all windows --------------------
  #
  # mats[[i]] is an Nt x Nt matrix where entry [t, col] holds the Wald
  # statistic for variable (i+1) in the window t:(col + wwid - 1).
  # The three window schemes are extracted from mats after the loop:
  #   FE: row 1 (start fixed at t=1)
  #   RO: diagonal (rolling window of fixed width wwid)
  #   RE: column-wise max over rows 1..col (recursive-expanding)
  #
  # Parallelism: each row t is independent, so we parallelize over t.
  # Progress bar runs only in sequential mode (forked workers cannot share
  # the parent's stdout cleanly).
  
  Nt   <- Tobs - wwid + 1L
  mats <- lapply(seq_len(K - 1L), function(i) matrix(NA_real_, Nt, Nt))
  names(mats) <- nms[-1L]   # name each element after its RHS variable
  
  cat("\nStage 1/2 — Computing Wald statistics\n")
  if (n_cores == 1L)
    pb_test <- utils::txtProgressBar(min = 0, max = Nt, style = 3, width = 50)
  
  row_results <- .lapply(seq_len(Nt), function(t) {
    row_vals <- vector("list", K - 1L)
    for (i in seq_len(K - 1L)) row_vals[[i]] <- rep(NA_real_, Nt)
    for (tt in (t + wwid - 1L):Tobs) {
      idx <- t:tt
      idx <- idx[idx >= lag_start]
      if (length(idx) < p + 2L) next
      W   <- .wstat(data[idx, , drop = FALSE], .buildX(Xl, Zl, idx),
                    use_robust = robust)
      col <- tt - wwid + 1L
      for (i in 2L:K) row_vals[[i - 1L]][col] <- W[i]
    }
    if (n_cores == 1L) utils::setTxtProgressBar(pb_test, t)
    row_vals
  })
  
  if (n_cores == 1L) close(pb_test)
  
  for (t in seq_len(Nt))
    for (i in seq_len(K - 1L))
      mats[[i]][t, ] <- row_results[[t]][[i]]
  
  # ---------- Reshape mats: raw Nt x Nt matrices -> named FE / RO / RE lists -
  #
  # Each element of mats is now a list with three named numeric vectors of
  # length Nt, one per window scheme:
  #   FE — forward-expanding  : row 1 of the raw matrix
  #   RO — rolling            : diagonal of the raw matrix
  #   RE — recursive-expanding: column-wise max over rows 1..col
  #
  # Access example: res$mats$MXN$FE, res$mats$MXN$RO, res$mats$MXN$RE
  
  mats <- lapply(mats, function(m) {
    fe <- m[1L, ]
    ro <- diag(m)
    re <- sapply(seq_len(Nt), function(col) {
      vals <- m[seq_len(col), col]
      if (all(is.na(vals))) NA_real_ else max(vals, na.rm = TRUE)
    })
    list(FE = fe, RO = ro, RE = re)
  })
  
  # ---------- Summary statistics --------------------------------------------
  #
  # Max_Wald_FE/RO/RE: maximum of each scheme's series across all windows.
  
  gcres <- matrix(
    NA_real_, K - 1L, 3L,
    dimnames = list(nms[-1L], c("Max_Wald_FE", "Max_Wald_RO", "Max_Wald_RE"))
  )
  for (i in seq_len(K - 1L)) {
    gcres[i, 1L] <- max(mats[[i]]$FE, na.rm = TRUE)
    gcres[i, 2L] <- max(mats[[i]]$RO, na.rm = TRUE)
    gcres[i, 3L] <- max(mats[[i]]$RE, na.rm = TRUE)
  }
  
  # ---------- Stage 2: Bootstrap critical values ----------------------------
  #
  # Following the size-control approach in Otero & Smith (2021):
  #   - Bootstrap window width : wwid_b = sizecontrol * 6
  #   - Sub-sample length      : full_b = sizecontrol + wwid_b - 1
  #   - Residuals fitted from  : the full-sample restricted VAR under H0
  #
  # Residual rows are resampled jointly across equations (single index vector)
  # to preserve contemporaneous cross-equation correlation — an extension
  # beyond the original Stata code.
  #
  # Parallelism: each variable j is independent, so we parallelize over j.
  
  wwid_b <- sc * 6L
  full_b <- sc + wwid_b - 1L   # last valid end-point in bootstrap sub-sample
  n_use  <- full_b + p + d      # rows needed (includes pre-sample for lags)
  
  idx_fit <- lag_start:Tobs
  tau_fit <- if (trend) cbind(idx_fit, 1L) else matrix(1L, length(idx_fit), 1L)
  Xf_fit  <- if (!is.null(Zl)) {
    matrix(as.double(cbind(Xl[idx_fit, ], Zl[idx_fit, ], tau_fit)),
           nrow = length(idx_fit))
  } else {
    matrix(as.double(cbind(Xl[idx_fit, ], tau_fit)),
           nrow = length(idx_fit))
  }
  ok_fit <- complete.cases(Xf_fit, data[idx_fit, ])
  Xf_ok  <- Xf_fit[ok_fit, ]
  Y_ok   <- matrix(as.double(data[idx_fit, , drop = FALSE][ok_fit, ]),
                   nrow = sum(ok_fit), ncol = K)
  n_ok   <- sum(ok_fit)
  
  if (n_ok < n_use)
    warning(sprintf(
      paste("Bootstrap requires %d observations but only %d complete cases",
            "are available. Bootstrap critical values may be unreliable.",
            "Consider reducing sizecontrol or increasing sample size."),
      n_use, n_ok), call. = FALSE)
  n_use <- min(n_use, n_ok)
  
  bsmat <- matrix(NA_real_, bootrepl, 3L * (K - 1L))
  
  cat("\nStage 2/2 — Bootstrap critical values\n")
  
  boot_results <- .lapply(2L:K, function(j) {
    
    lag_cols_j <- j + (seq_len(p) - 1L) * K
    
    # Fit restricted VAR under H0 per equation (equation 1 drops lag_cols_j)
    xb_r <- e_r <- matrix(NA_real_, n_ok, K)
    for (eq in seq_len(K)) {
      Xeq        <- if (eq == 1L) Xf_ok[, -lag_cols_j, drop = FALSE] else Xf_ok
      b_eq       <- drop(solve(crossprod(Xeq), crossprod(Xeq, Y_ok[, eq])))
      xb_r[, eq] <- drop(Xeq %*% b_eq)
      e_r[, eq]  <- Y_ok[, eq] - xb_r[, eq]
    }
    xb_sub <- xb_r[seq_len(n_use), , drop = FALSE]
    
    cat(sprintf("  H0: %s does not Granger-cause %s  [%d replications]\n",
                nms[j], nms[1L], bootrepl))
    if (n_cores == 1L)
      pb_boot <- utils::txtProgressBar(min = 0, max = bootrepl,
                                       style = 3, width = 50)
    
    res_j <- matrix(NA_real_, bootrepl, 3L)
    
    for (b in seq_len(bootrepl)) {
      # Joint row resample preserves contemporaneous cross-equation correlation
      ridx  <- sample.int(n_ok, n_use, replace = TRUE)
      bdata <- xb_sub + e_r[ridx, , drop = FALSE]
      
      Xl_b <- .lags(bdata, seq_len(p))
      Zl_b <- if (d > 0L) .lags(bdata, (p + 1L):(p + d)) else NULL
      
      mat_b <- matrix(NA_real_, sc, sc)
      
      for (t_b in seq_len(sc)) {
        tt_b_start <- t_b + wwid_b - 1L
        if (tt_b_start > full_b) next
        for (tt_b in tt_b_start:full_b) {
          idx_b <- t_b:tt_b
          idx_b <- idx_b[idx_b >= lag_start]
          if (length(idx_b) < p + 2L) next
          
          Yb   <- bdata[idx_b, , drop = FALSE]
          Xfb  <- .buildX(Xl_b, Zl_b, idx_b)
          ok_b <- complete.cases(Yb, Xfb)
          nb   <- sum(ok_b)
          if (nb <= ncol(Xfb)) next
          
          Yb2  <- Yb[ok_b, , drop = FALSE]
          Xfb2 <- Xfb[ok_b, , drop = FALSE]
          yb   <- Yb2[, 1L]
          
          XpXi_b <- tryCatch(solve(crossprod(Xfb2)), error = function(e) NULL)
          if (is.null(XpXi_b)) next
          
          bb  <- drop(XpXi_b %*% crossprod(Xfb2, yb))
          e2b <- drop(yb - Xfb2 %*% bb)^2
          omb <- if (!robust) {
            (sum(e2b) / nb) * XpXi_b
          } else {
            XpXi_b %*% (t(Xfb2) %*% (Xfb2 * e2b)) %*% XpXi_b
          }
          omb <- (omb + t(omb)) / 2
          
          lag_cols_jb <- j + (seq_len(p) - 1L) * K
          R   <- matrix(0, p, length(bb))
          for (k in seq_len(p)) R[k, lag_cols_jb[k]] <- 1L
          Rb  <- R %*% bb
          Ri  <- tryCatch(solve(R %*% omb %*% t(R)), error = function(e) NULL)
          
          col_b <- tt_b - wwid_b + 1L
          if (!is.null(Ri) && col_b >= 1L && col_b <= sc)
            mat_b[t_b, col_b] <- as.numeric(t(Rb) %*% Ri %*% Rb)
        }
      }
      
      res_j[b, 1L] <- max(mat_b[1L, ], na.rm = TRUE)   # FE
      res_j[b, 2L] <- max(diag(mat_b),  na.rm = TRUE)   # RO
      re_b <- sapply(seq_len(sc), function(col) {        # RE
        vals <- mat_b[seq_len(col), col]
        if (all(is.na(vals))) NA_real_ else max(vals, na.rm = TRUE)
      })
      res_j[b, 3L] <- max(re_b, na.rm = TRUE)
      
      if (n_cores == 1L) utils::setTxtProgressBar(pb_boot, b)
    }
    
    if (n_cores == 1L) close(pb_boot)
    res_j
  })
  
  for (j in 2L:K) {
    fc <- (j - 2L) * 3L + 1L   # column offset in bsmat for variable j
    bsmat[, fc:(fc + 2L)] <- boot_results[[j - 1L]]
  }
  
  bsmat[!is.finite(bsmat)] <- NA_real_
  
  # ---------- Critical values -----------------------------------------------
  
  .cv <- function(q) {
    m <- matrix(NA_real_, K - 1L, 3L,
                dimnames = list(nms[-1L],
                                c("Max_Wald_FE", "Max_Wald_RO", "Max_Wald_RE")))
    for (j in 2L:K) {
      fc       <- (j - 2L) * 3L + 1L
      cols     <- fc:(fc + 2L)
      m[j - 1L, ] <- apply(bsmat[, cols, drop = FALSE], 2L,
                           quantile, probs = q, na.rm = TRUE)
    }
    m
  }
  
  cv90 <- .cv(0.90); cv95 <- .cv(0.95); cv99 <- .cv(0.99)
  
  # ---------- Print output --------------------------------------------------
  
  la <- if (d > 0L) "LA-" else ""
  tr <- if (trend) " with trend" else ""
  cat(sprintf("\nTime-varying %sVAR Granger causality test%s\n", la, tr))
  cat(sprintf("H0: %s is NOT Granger-caused\n\n", nms[1L]))
  cat("Test statistics:\n"); print(round(gcres, 4L))
  for (pct in c("90", "95", "99")) {
    cv_obj <- get(paste0("cv", pct))
    cat(sprintf("\n%sth percentile critical values [%d bootstrap replications]:\n",
                pct, bootrepl))
    print(round(cv_obj, 4L))
  }
  
  attr(gcres, "depname") <- nms[1L]
  
  invisible(list(
    stats       = gcres,
    cv90        = cv90,  cv95 = cv95,  cv99 = cv99,
    mats        = mats,
    p           = p,     d    = d,     window = wwid,
    boot        = bootrepl,            sizecontrol = sc
  ))
}


# =============================================================================
#' Plot Time-Varying Granger Causality Results
#'
#' @description
#' Produces one faceted ggplot per RHS variable, showing the time path of the
#' Wald statistic for each of the three window schemes (Forward-Expanding,
#' Rolling, Recursive-Expanding) alongside the bootstrap critical value
#' threshold. Periods where the statistic exceeds the dashed threshold indicate
#' rejection of the null hypothesis of Granger non-causality.
#'
#' @param res
#'   Object returned by \code{\link{tvgc}}.
#' @param dates
#'   Optional vector of dates or labels of length equal to \code{nrow(data)}
#'   (the original sample size passed to \code{tvgc()}). Used to label the
#'   x-axis. If \code{NULL} (default), integer observation indices are used.
#'   \strong{Note}: dates must not be included in the \code{data} argument
#'   of \code{tvgc()}; pass them here instead.
#' @param pct
#'   Which bootstrap critical value to overlay: \code{90}, \code{95}
#'   (default), or \code{99}.
#'
#' @return
#' Invisibly, a named list of \code{ggplot} objects, one per RHS variable.
#' Each plot is also printed to the active graphics device.
#'
#' @references
#' Otero, J. & Smith, J. (2021). Testing for Granger non-causality in
#' heterogeneous panels. \emph{Journal of Applied Econometrics}, 36(7),
#' 858–876. \doi{10.1002/jae.2831}
#'
#' @examples
#' \dontrun{
#' res   <- tvgc(data_mat, p = 2, d = 1, boot = 199, seed = 42)
#' dates <- seq(as.Date("2000-01-03"), by = "day", length.out = nrow(data_mat))
#' plots <- tvgc_plot(res, dates = dates, pct = 95)
#' # Save a specific panel:
#' ggplot2::ggsave("tvgc_X.pdf", plots[["X"]], width = 8, height = 6)
#' }
#'
#' @seealso \code{\link{tvgc}}
#' @export
# =============================================================================
tvgc_plot <- function(res, dates = NULL, pct = 95) {
  if (!requireNamespace("ggplot2", quietly = TRUE))
    stop("Please install ggplot2: install.packages('ggplot2')")
  
  cv_key <- paste0("cv", pct)
  if (!cv_key %in% names(res))
    stop("pct must be 90, 95, or 99")
  
  cv_mat <- res[[cv_key]]
  mats   <- res$mats
  wwid   <- res$window
  xnames <- rownames(res$stats)
  K1     <- length(xnames)
  
  types   <- c("Forward expanding", "Rolling", "Recursive expanding")
  col_map <- c("Forward expanding"   = "#2166ac",
               "Rolling"             = "#d6604d",
               "Recursive expanding" = "#4dac26")
  
  plots <- vector("list", K1)
  
  for (i in seq_len(K1)) {
    fe <- mats[[i]]$FE
    ro <- mats[[i]]$RO
    re <- mats[[i]]$RE
    Nt <- length(fe)
    
    # x-axis: end-of-window observation index
    x_idx <- seq_len(Nt) + wwid - 1L
    if (!is.null(dates)) {
      if (length(dates) < max(x_idx))
        stop("'dates' is shorter than the series length")
      x_vals <- dates[x_idx]
    } else {
      x_vals <- x_idx
    }
    
    df <- data.frame(
      x    = rep(x_vals, 3L),
      stat = c(fe, ro, re),
      type = factor(rep(types, each = Nt), levels = types),
      cv   = rep(c(cv_mat[i, 1L], cv_mat[i, 2L], cv_mat[i, 3L]), each = Nt)
    )
    df <- df[is.finite(df$stat), ]
    
    p <- ggplot2::ggplot(df, ggplot2::aes(x = x, y = stat, colour = type)) +
      ggplot2::geom_line(linewidth = 0.7) +
      ggplot2::geom_line(ggplot2::aes(y = cv, colour = type),
                         linetype = "dashed", linewidth = 0.5) +
      ggplot2::scale_colour_manual(values = col_map, name = NULL) +
      ggplot2::facet_wrap(~type, ncol = 1L, scales = "free_y") +
      ggplot2::labs(
        title    = sprintf("TVGC: %s Granger-causes %s?",
                           xnames[i], attr(res$stats, "depname")),
        subtitle = sprintf("Dashed line = %dth pct bootstrap critical value", pct),
        x        = NULL,
        y        = "Wald statistic"
      ) +
      ggplot2::theme_bw(base_size = 11) +
      ggplot2::theme(
        legend.position  = "none",
        strip.background = ggplot2::element_rect(fill = "grey92"),
        panel.grid.minor = ggplot2::element_blank()
      )
    
    print(p)
    plots[[i]] <- p
    names(plots)[i] <- xnames[i]
  }
  
  invisible(plots)
}
