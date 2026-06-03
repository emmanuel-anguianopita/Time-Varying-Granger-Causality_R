# tvgc — Time-Varying Granger Causality Test in R
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20530954-blue)](https://doi.org/10.5281/zenodo.20530954)

R implementation of the **time-varying Granger causality (TVGC) test** proposed by Shi, Hurn & Phillips (2020), based on the lag-augmented VAR framework of Toda & Yamamoto (1995) and Dolado & Lütkepohl (1996).

---

## Background

Standard Granger causality tests assume a fixed causal structure throughout the sample. This is often implausible in macroeconomic and financial data subject to structural breaks, policy changes, or crisis episodes.

**Shi, Hurn & Phillips (2020)** introduced the three window-based algorithms implemented here — forward expanding (FE), rolling window (RW), and recursive evolving (RE) — as data-driven methods for detecting change points in causal relationships within a lag-augmented VAR framework. The paper derives the limit distributions for the subsample Wald statistics and develops bootstrap methods to control family-wise size across the recursive testing algorithms. Simulation evidence in that paper suggests that the recursive evolving algorithm provides the most reliable results, followed by the rolling window method.

**Otero & Smith (2021)** extend this framework to heterogeneous panel settings and provide a companion Stata implementation. This R code adapts that Stata implementation to a single time-series setting, with several extensions (see [Adaptation notes](#adaptation-notes)).

Both approaches build on the lag-augmented VAR of Toda & Yamamoto (1995) and Dolado & Lütkepohl (1996), which ensures valid inference under unknown integration order without prior unit root testing or detrending.

---

## References

> Shi, S., Hurn, S. & Phillips, P. C. B. (2020). Causal change detection in possibly integrated systems: revisiting the money–income relationship. *Journal of Financial Econometrics*, 18(1), 158–180. https://doi.org/10.1093/jjfinec/nbz004

> Otero, J. & Smith, J. (2021). Testing for Granger non-causality in heterogeneous panels. *Journal of Applied Econometrics*, 36(7), 858–876. https://doi.org/10.1002/jae.2831

> Toda, H. Y. & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. *Journal of Econometrics*, 66(1–2), 225–250. https://doi.org/10.1016/0304-4076(94)01616-8

> Dolado, J. J. & Lütkepohl, H. (1996). Making Wald tests work for cointegrated VAR systems. *Econometric Reviews*, 15(4), 369–386. https://doi.org/10.1080/07474939608800362

> Rossi, B. (2005). Optimal tests for nested model selection with underlying parameter instability. *Econometric Theory*, 21(5), 962–990.

---

## Window schemes

Following Shi, Hurn & Phillips (2020), the test computes a Wald statistic for each sub-sample window and reports the maximum across all windows of each type:

| Scheme | Abbreviation | Description |
|---|---|---|
| Forward Expanding Window | `FE` | Start fixed at $t = 1$, end grows from $t + w - 1$ to $T$ |
| Rolling Window | `RW` | Fixed-width window of size `window` slides across the sample |
| Recursive Evolving Window | `RE` | For each end-point $tt$, maximum Wald over all starts $t \leq tt$ |

Rejection of H₀ in any window indicates an **episode** of Granger causality. The time path of the statistic (see `tvgc_plot()`) reveals *when* causality appears or disappears.

---

## Installation

No package installation required. Source the file directly:

```r
source("tvgc.R")
```

**Dependencies:**

- Base R only (no external packages required for `tvgc()`)
- `ggplot2` for `tvgc_plot()`
- `parallel` for multi-core execution (included in base R)

---

## Input data

`tvgc()` expects a **numeric matrix** as its first argument. Three rules apply before calling the function:

1. **No date column.** If your data.frame has a date or index column, remove it first. Dates cause `as.matrix()` to coerce the entire object to `character`, which breaks all internal matrix operations. Pass the date vector separately to `tvgc_plot()` via the `dates` argument.

2. **Coerce to `as.matrix()`.** The function calls `as.matrix()` and `storage.mode(data) <- "double"` internally, but it is good practice to pass a clean numeric matrix explicitly to avoid unexpected type coercions.

3. **Column 1 is the dependent variable (Y).** The function tests whether each of columns 2 through K Granger-causes column 1. Column 1 itself never appears as a candidate cause and will not show up in the results or plots.

```r
# Correct — li is the dependent variable; lm1, lp, r are tested as causes
dates_vec  <- df$date
data_mat   <- data |> select(li, lm1, lp, r) |> as.matrix()
res        <- tvgc(data_mat, ...)
tvgc_plot(res, dates = dates_vec)

# Wrong — will error
res <- tvgc(df)                                 # date column included
```

### What the results contain

After running `tvgc()`, `rownames(res$stats)` returns only the **RHS variables** (columns 2:K), never the dependent variable:

```r
data_mat <- data |> select(li, lm1, lp, r) |> as.matrix()
res      <- tvgc(data_mat, ...)

rownames(res$stats)
#> [1] "lm1" "lp" "r"
#
# Each row answers: does this variable Granger-cause li?
# li itself is not listed because it is the dependent variable, not a cause.
```

To test causality in **both directions** (e.g. does `li` cause `lm1` AND does `lm1` cause `li`), run the function twice with the variable order swapped:

```r
# Direction 1: do lm1, lp, r cause li?
data_1 <- data |> select(li, lm1, lp, r) |> as.matrix()
res_1  <- tvgc(data_1, p = 2, d = 1, boot = 499, seed = 42)

# Direction 2: do li, lp, r cause lm1?
data_2 <- data |> select(lm1, li, lp, r) |> as.matrix()
res_2  <- tvgc(data_2, p = 2, d = 1, boot = 499, seed = 42)
```

---

## Usage

```r
# 1. Prepare data — numeric matrix only, NO date column
#    Column 1 = dependent variable (Y)
#    Columns 2:K = RHS variables to test

dates_vec <- df$date                          # save dates separately
data_mat  <- as.matrix(df[, c("Y", "X1", "X2")])

# 2. Run the test
res <- tvgc(
  data        = data_mat,
  p           = 2,      # VAR lag order
  d           = 1,      # integration order (Toda-Yamamoto augmentation)
  boot        = 499,    # bootstrap replications (>= 499 for publication)
  seed        = 42,     # reproducibility
  sizecontrol = 12,     # bootstrap size-control parameter
  trend       = FALSE,  # include linear trend?
  robust      = FALSE,  # HC standard errors?
  cores       = 1L      # parallel workers (Mac/Linux only)
)

# 3. Plot — all variables, all window schemes
plots <- tvgc_plot(res, dates = dates_vec, pct = 95)
```

### Parallel execution (Mac / Linux)

```r
res <- tvgc(data_mat, p = 2, d = 1, boot = 499, seed = 42,
            cores = parallel::detectCores() - 1L)
```

> **Windows**: `parallel::mclapply` is fork-based and not available on Windows. The function falls back to sequential execution automatically and prints a message.

---

## Plotting with `tvgc_plot()`

`tvgc_plot()` visualises the time path of the Wald statistic for each RHS variable and window scheme. Each plot shows:

- A **filled area** (blue) for the Wald statistic over time.
- An **orange line** for the bootstrap critical value at the chosen significance level.
- **Grey shaded bands** marking periods where the Wald statistic exceeds the critical value — i.e. episodes of rejection of H₀ (evidence of Granger causality).

Each combination of variable × window scheme is saved as a **separate named object** in the returned list.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `res` | — | Object returned by `tvgc()`. |
| `dates` | `NULL` | `Date` vector of length equal to `nrow(data)`. **Do not include in `tvgc()`** — pass here for x-axis labels only. If `NULL`, integer indices are used. |
| `pct` | `95` | Significance level for the bootstrap critical value line: `90`, `95`, or `99`. |
| `vars` | `NULL` | RHS variables to plot. Character vector of names (e.g. `"lm1"`) or integer index. `NULL` plots all. All three window schemes (FE, RW, RE) are always plotted for each selected variable. |

### Naming convention

Returned plots are named `"<variable>_<scheme>"`:

```r
plots <- tvgc_plot(res, dates = dates_vec, pct = 95)

plots$lm1_FE   # Forward Expanding Window — does lm1 Granger-cause Y?
plots$lm1_RW   # Rolling Window
plots$lm1_RE   # Recursive Evolving Window
plots$lp_FE
plots$r_RE
# ...
```

### Selecting variables

All three window schemes (FE, RW, RE) are always plotted. Use `vars` to restrict which RHS variables are shown:

```r
# One variable — produces three plots: lm1_FE, lm1_RW, lm1_RE
tvgc_plot(res, dates = dates_vec, pct = 95, vars = "lm1")

# Two variables — produces six plots
tvgc_plot(res, dates = dates_vec, pct = 95, vars = c("lm1", "lp"))

# All variables (default) — produces 3 × K plots
tvgc_plot(res, dates = dates_vec, pct = 95)

# Save a specific plot
ggplot2::ggsave("lm1_RE.pdf", plots$lm1_RE, width = 8, height = 5)
```

### Note on `dates`

`dates` must be a vector of length `T` (the full sample), **not** `Nt = T - window + 1`. The function shifts the index internally to align each Wald statistic with the **end date** of its window. Passing a pre-trimmed vector will produce a misaligned x-axis.

```r
# Correct
dates_vec <- as.Date(df$date)          # length T
tvgc_plot(res, dates = dates_vec)

# Wrong — x-axis will be misaligned
tvgc_plot(res, dates = dates_vec[-(1:wwid)])   # do not trim
```

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `data` | — | Numeric matrix (`as.matrix()`). **No date column** — remove it before calling and pass it to `tvgc_plot()` instead. Column 1 = dependent variable; columns 2:K = RHS variables. |
| `p` | `2` | VAR lag order. Only first `p` lags are restricted under H₀. |
| `d` | `1` | Integration order for Toda–Yamamoto augmentation. Use `d = 0` for stationary data. |
| `window` | `floor(0.2 * T)` | Minimum window width in observations. |
| `boot` | `199` | Bootstrap replications. Use ≥ 499 for publication. |
| `seed` | `123` | Random seed for reproducibility. Matches the `seed(123)` used in the Otero & Smith (2021) Stata implementation. Pass `NULL` to disable. |
| `sizecontrol` | `12` | Bootstrap size-control parameter (bootstrap window = `sizecontrol * 6`). |
| `trend` | `FALSE` | Include a linear trend in the VAR. |
| `robust` | `FALSE` | Use HC (sandwich) standard errors. |
| `cores` | `1L` | Parallel workers. Ignored on Windows. |

---

## Output

`tvgc()` returns an invisible list:

| Element | Description |
|---|---|
| `stats` | (K−1) × 3 matrix of maximum Wald statistics, columns: `Max_Wald_FE`, `Max_Wald_RO`, `Max_Wald_RE`. Rows = RHS variables. |
| `cv90`, `cv95`, `cv99` | (K−1) × 3 matrices of bootstrap critical values, same layout as `stats`. |
| `mats` | Named list of (K−1) elements (one per RHS variable). Each element is a list with three numeric vectors of length `Nt = T - window + 1`: `$FE`, `$RO`, `$RE`. These are the raw Wald series used by `tvgc_plot()`. |
| `p`, `d`, `window`, `boot`, `sizecontrol` | Parameters used in the run. |

---

## Reproducibility and the original paper

Bootstrap critical values depend on the random number generator used. This has two implications:

**Within this implementation:** the default `seed = 123` — matching the `seed(123)` used in the Otero & Smith (2021) Stata code — guarantees that two users running `tvgc()` with the same data and parameters will obtain **identical results**. To change the seed, pass any integer (e.g. `seed = 42`). To disable the seed and get a different draw each run, pass `seed = NULL`.

**Versus the original Shi, Hurn & Phillips (2020) paper:** the paper was implemented in **Matlab** and does **not report a random seed**. Because Matlab and R use different random number generators, bootstrap critical values from this function will differ numerically from those in the paper even with identical data and settings. This is expected and does not indicate an error.

What *should* match are the **Wald statistics**, which are deterministic (no randomness involved). If your `res$stats` values are close to the maxima visible in Figure 2 of the paper, the implementation is correct.

To verify that your conclusions are robust to the choice of seed, run:

```r
seeds <- c(123, 42, 456, 789, 2024)

cv_check <- sapply(seeds, function(s) {
  res <- tvgc(final_data, p = 2, d = 1, trend = TRUE,
              window = 72, sizecontrol = 60,
              boot = 499, seed = s,
              robust = TRUE, hc_type = "HC0")
  res$cv95["lm1", ]
})

round(cv_check, 3)   # critical values should be stable across seeds
```

If the Wald statistic consistently exceeds the critical value across all seeds, the conclusion is robust regardless of which seed the original authors used.

---

## Adaptation notes

This R implementation extends the original Otero & Smith (2021) Stata code — itself an adaptation of the Shi, Hurn & Phillips (2020) framework — in the following ways:

1. **Single-pass matrix**: All three window schemes are extracted from a single Nt × Nt Wald-statistic matrix per variable, rather than three separate loops.

2. **Corrected RE statistic**: The recursive-expanding statistic takes the column-wise maximum over rows 1 through `col` only, consistent with Rossi (2005). The naive full-column maximum mixes start points and produces an upward-biased statistic.

3. **Joint residual resampling**: Bootstrap residuals are resampled row-wise across all equations simultaneously, preserving contemporaneous cross-equation correlation. The original Stata code does not enforce this.

4. **Parallelism**: `parallel::mclapply` over the outer loop of each stage (rows in Stage 1, variables in Stage 2). Automatic fallback to sequential on Windows.

5. **HC standard errors**: Optional heteroskedasticity-consistent Wald statistic via the `robust` argument.

6. **Type safety**: All design matrices are coerced to `double` to prevent `crossprod()` errors when input data contains integer or mixed-type columns.

---

## Example output

```
Stage 1/2 — Computing Wald statistics
  |==================================================| 100%

Stage 2/2 — Bootstrap critical values
  H0: X does not Granger-cause Y  [499 replications]
  |==================================================| 100%

Time-varying LA-VAR Granger causality test
H0: Y is NOT Granger-caused

Test statistics:
    Max_Wald_FE Max_Wald_RO Max_Wald_RE
lm1    8.680759    26.26639    26.26639
lp    12.288680    14.87708    26.66078
r      7.936891    48.09570    51.62437

95th percentile critical values [499 bootstrap replications]:
    Max_Wald_FE Max_Wald_RO Max_Wald_RE
lm1   10.977718   12.300516   12.602930
lp     6.609465    7.478081    8.928484
r     12.578064   11.051764   12.750582

---
```

## Author

Adapted by Javier Emmanuel Anguiano Pita  
SECIHTI – Universidad de Guadalajara <br>
Based on Otero & Smith (2021) and their accompanying Stata implementation.

