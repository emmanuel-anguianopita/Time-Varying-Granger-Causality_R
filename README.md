# tvgc — Time-Varying Granger Causality Test in R

R implementation of the **time-varying Granger causality (TVGC) test** proposed by Otero & Smith (2021), based on the lag-augmented VAR framework of Toda & Yamamoto (1995) and Dolado & Lütkepohl (1996).

## Background

Standard Granger causality tests assume a fixed causal structure throughout the sample. This is often implausible in macroeconomic and financial data subject to structural breaks, policy changes, or crisis episodes.

**Shi, Hurn & Phillips (2020)** introduced the three window-based algorithms implemented here — forward recursive (FE), rolling window (RO), and recursive evolving (RE) — as data-driven methods for detecting change points in causal relationships within a lag-augmented VAR framework. The paper derives the limit distributions for the subsample Wald statistics and develops bootstrap methods to control family-wise size across the recursive testing algorithms. Simulation evidence in that paper suggests that the recursive evolving algorithm provides the most reliable results, followed by the rolling window method.

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
| Forward-Expanding | FE | Start fixed at $t = 1$, end grows from $t + w - 1$ to $T$ |
| Rolling | RO | Fixed-width window of size `window` slides across the sample |
| Recursive-Expanding | RE | For each end-point $tt$, maximum Wald over all starts $t \leq tt$ |

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

`tvgc()` expects a **numeric matrix** as its first argument. Two rules apply before calling the function:

1. **No date column.** If your data.frame has a date or index column, remove it first. Dates cause `as.matrix()` to coerce the entire object to `character`, which breaks all internal matrix operations. Pass the date vector separately to `tvgc_plot()` via the `dates` argument.

2. **Coerce to `as.matrix()`.** The function calls `as.matrix()` and `storage.mode(data) <- "double"` internally, but it is good practice to pass a clean numeric matrix explicitly to avoid unexpected type coercions.

```r
# Correct
dates_vec <- df$date                            # 1. save dates separately
data_mat  <- as.matrix(df[, c("Y", "X1", "X2")])  # 2. numeric columns only
res       <- tvgc(data_mat, ...)
tvgc_plot(res, dates = dates_vec)

# Wrong — will error
res <- tvgc(df)                                 # date column included
```

Column order matters: **column 1 is always the dependent variable** (Y); columns 2 through K are the RHS variables whose Granger-causal role is tested.

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

# 3. Plot
tvgc_plot(res, dates = dates_vec, pct = 95)
```

### Parallel execution (Mac / Linux)

```r
res <- tvgc(data_mat, p = 2, d = 1, boot = 499, seed = 42,
            cores = parallel::detectCores() - 1L)
```

> **Windows**: `parallel::mclapply` is fork-based and not available on Windows. The function falls back to sequential execution automatically and prints a message.

---

## Arguments

| Argument | Default | Description |
|---|---|---|
| `data` | — | Numeric matrix (`as.matrix()`). **No date column** — remove it before calling and pass it to `tvgc_plot()` instead. Column 1 = dependent variable; columns 2:K = RHS variables. |
| `p` | `2` | VAR lag order. Only first `p` lags are restricted under H₀. |
| `d` | `1` | Integration order for Toda–Yamamoto augmentation. Use `d = 0` for stationary data. |
| `window` | `floor(0.2 * T)` | Minimum window width in observations. |
| `boot` | `199` | Bootstrap replications. Use ≥ 499 for publication. |
| `seed` | `NULL` | Random seed for reproducibility. |
| `sizecontrol` | `12` | Bootstrap size-control parameter (bootstrap window = `sizecontrol * 6`). |
| `trend` | `FALSE` | Include a linear trend in the VAR. |
| `robust` | `FALSE` | Use HC (sandwich) standard errors. |
| `cores` | `1L` | Parallel workers. Ignored on Windows. |

---

## Output

`tvgc()` returns an invisible list:

| Element | Description |
|---|---|
| `stats` | (K−1) × 3 matrix of test statistics (`Max_Wald_FE`, `Max_Wald_RO`, `Max_Wald_RE`) |
| `cv90`, `cv95`, `cv99` | (K−1) × 3 matrices of bootstrap critical values |
| `mats` | List of (K−1) Nt × Nt Wald-statistic matrices (full surface) |
| `p`, `d`, `window`, `boot`, `sizecontrol` | Parameters used |

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
X      12.341       9.876      14.203

95th percentile critical values [499 bootstrap replications]:
  Max_Wald_FE Max_Wald_RO Max_Wald_RE
X       8.412       7.934       9.107
```

---

## Author

Adapted by Javier Emmanuel Anguiano Pita  
SECIHTI – Universidad de Guadalajara (CUCEA / DEEC)  
Based on Otero & Smith (2021) and their accompanying Stata implementation.

