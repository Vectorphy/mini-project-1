# Analysis of UPI Transaction Growth — Final Report

Author: Vectorphy
Date: 2025-11-04

## Executive summary

This report examines the effect of the COVID-19 pandemic on UPI transaction volume and value using exploratory data analysis and SARIMA time-series forecasting. Two forecasting exercises were performed:

1. Train on Pre‑COVID data and forecast the During‑COVID period (Pre-COVID → During-COVID).
2. Train on combined Pre+During data and forecast the Post‑COVID period ((Pre+During) → Post‑COVID).

Key conclusions (summary):
- The pandemic caused a statistically significant and large change in UPI transaction behavior relative to the pre-pandemic trend.
- A SARIMA model fit to the Pre+During period over-predicted Post-COVID volumes on average — observed Post‑COVID volumes were significantly lower than that forecast, suggesting the pandemic-era acceleration did not simply continue at the same aggressive rate.
- Additional hypothesis tests confirm a structural break at the pandemic onset and a significant change in volatility.

This document is the cleaned, submission-ready report; figures and data exports are produced by `analysis.py` and saved in the `visualizations/` folder.

---

## 1. Data and preprocessing

Data source: `Untitled spreadsheet - Sheet1.csv` (project root).

Preprocessing steps (performed in `analysis.py`):
- Parse `Month` as monthly datetime (format `%b-%y`).
- Convert numeric columns to float, removing comma separators where present.
- Create derived columns: `Volume (in Bn)`, `Value (in Trn)`, yearly and month-name columns, and month-over-month growth rates.
- Split into three periods:
	- Pre‑COVID: `Month` ≤ 2020-03-31
	- During‑COVID: 2020-04-01 — 2022-01-31
	- Post‑COVID: from 2022-08-01 onward
- Exported cleaned data workbook: `upi_data_cleaned.xlsx` (sheets: Pre-COVID, During-COVID, Post-COVID).

Notes on missing data: the pre+during training set was reindexed to a contiguous monthly index and linearly interpolated for any missing months prior to SARIMA fitting (conservative interpolation only for model training).

---

## 2. Exploratory data analysis (highlights)

- Transaction volume and value show an exponential-like upward trend across the full time span, with a clear acceleration around the start of the pandemic.
- Yearly × monthly heatmaps show consistent seasonality (peaks aligned to certain months every year).
- Summary statistics by period indicate large increases in mean and dispersion from Pre → During → Post.
- Correlation matrix: `Volume (in Bn)` and `Value (in Trn)` are strongly positively correlated. The number of banks live on UPI also correlates positively with volume/value.

Representative visualizations (generated and saved under `visualizations/`):
- `time_series_full.png`
- `volume_yearly_monthly_heatmap.png`, `value_yearly_monthly_heatmap.png`
- `eda_distributions_pre-covid.png`, `eda_distributions_during-covid.png`, `eda_distributions_post-covid.png`
- `volume_boxplot_by_period.png`, `value_boxplot_by_period.png`

Additionally, normalized comparisons (z-score per period) were computed and plotted to allow direct distributional comparison on a common scale:
- `volume_boxplot_zscore_by_period.png`
- `value_boxplot_zscore_by_period.png`
- `volume_kde_zscore_by_period.png`
- `value_kde_zscore_by_period.png`

---

## 3. SARIMA forecasting approach

- Modeling framework: SARIMA via `statsmodels.tsa.statespace.SARIMAX`.
- The models account for monthly seasonality (s = 12) and include differencing where appropriate.
- For Part 2 ((Pre+During) → Post), a compact grid-search over a small set of (p,d,q) and seasonal (P,D,Q) values was run and the model with lowest AIC was selected. The grid was intentionally small to keep runs fast during development; expand it if you need more exhaustive model selection.

Model-selection result (Part 2):
- Grid-search tried 96 model fits and selected: order = (1,1,1), seasonal_order = (1,1,1,12) (best AIC ≈ 545.86).

Forecasting alignment and gap handling:
- Forecasts are produced starting immediately after the last training month (no artificial gaps). Forecasts are indexed monthly (month start frequency) and plotted continuously from training into forecast horizon; residuals are computed only on overlapping months present in the test set.

---

## 4. Forecasting results and interpretation

### Part 1 — Pre‑COVID → During‑COVID
- Forecast plot: `visualizations/sarima_forecast_pre-covid_vs_during-covid.png`.
- Residuals t-test (one-sample against zero):
	- T-statistic = 5.1217
	- P-value = 0.0000
- Interpretation: The actual During‑COVID volumes were significantly different from what the Pre‑COVID model predicted. The large positive t-statistic indicates observed volumes were higher than pre-COVID-based forecast on average.

### Part 2 — (Pre+During) → Post‑COVID
- Forecast plot: `visualizations/sarima_forecast_(pre+during)-covid_vs_post-covid.png`.
- Selected model: SARIMA(1,1,1)(1,1,1,12) (AIC ≈ 545.86).
- Residuals t-test (one-sample against zero):
	- T-statistic = -5.3953
	- P-value = 0.0000
- Interpretation: The Post‑COVID observed volumes were significantly lower than the forecast produced from the combined Pre+During trend (negative t-statistic). In plain terms, while volumes remain higher than pre-pandemic levels, they did not continue to grow as fast as the pandemic-era trajectory would have suggested.

Caveat: residual t-tests assume residual independence. I recommend performing ACF/Ljung‑Box tests on the residuals before over-interpreting p-values; if autocorrelation exists, use robust or bootstrap inference.

---

## 5. Additional hypothesis tests (summary)

These complement the forecasting analysis with standard comparisons across periods.

- Two-sample Welch t-tests (Pre-COVID vs Post-onset [During+Post]):
	- Volume: T-statistic = -11.0504, P-value = 0.0000
	- Value:  T-statistic = -12.6223, P-value = 0.0000
	- Interpretation: The means differ strongly between the early and later periods (post-onset means are substantially larger).

- Chow-style structural break test (regression with dummy and time*dummy):
	- F-statistic = 595.7437, P-value = 0.0000
	- Interpretation: Strong evidence of a structural break at pandemic onset (change in level/trend).

- Levene test on percent-change (volatility comparison):
	- Levene statistic = 5.1980, P-value = 0.0248
	- Interpretation: Volatility (variance of monthly growth rates) differs between pre- and post-onset periods.

---

## 6. Diagnostics and recommendations

Observed diagnostics:
- Statsmodels occasionally warns when the training window is short relative to the seasonal period: "Too few observations to estimate starting parameters for seasonal ARMA." This does not always invalidate the fit, but it warrants caution.

Recommendations before finalizing statistical claims:
1. Plot and test residuals (ACF/PACF, Ljung‑Box) for each fitted model. If residuals are autocorrelated, prefer bootstrap or robust inference for tests.
2. Consider using BIC for model selection if you want stronger parsimony.
3. Persist the best-fitted model to disk (pickle) to avoid refitting when regenerating figures.
4. Optionally compare with alternative forecasting methods (Prophet, ETS, TBATS) for robustness.

---

## 7. Artifacts produced

- `upi_data_cleaned.xlsx` — cleaned data workbook (3 sheets).
- Visualizations (PNG) in `visualizations/` including:
	- `time_series_full.png`
	- `eda_distributions_pre-covid.png`, `eda_distributions_during-covid.png`, `eda_distributions_post-covid.png`
	- `volume_boxplot_by_period.png`, `value_boxplot_by_period.png`
	- `volume_boxplot_zscore_by_period.png`, `value_boxplot_zscore_by_period.png`
	- `volume_kde_zscore_by_period.png`, `value_kde_zscore_by_period.png`
	- `volume_yearly_monthly_heatmap.png`, `value_yearly_monthly_heatmap.png`
	- `correlation_heatmap.png`
	- `volume_decomposition.png`, `value_decomposition.png`
	- `sarima_forecast_pre-covid_vs_during-covid.png`
	- `sarima_forecast_(pre+during)-covid_vs_post-covid.png`

---

## 8. How to reproduce

From PowerShell in the project root run:

```powershell
Set-Location -Path 'd:\Mini-Project-1'
python .\analysis.py
```

This will regenerate the cleaned data and all visualizations. If you want to run the grid-search with a larger hyperparameter set, edit `find_best_sarima()` in `analysis.py`.

---

## 9. Final remarks and submission checklist

- I recommend a final proofread for language and the one-paragraph executive summary used for submission.
- Confirm that the `visualizations/` folder contains the PNGs referenced above (the script creates them when run).

If you want, I can:
- Produce a one-page PDF or PowerPoint summarizing the key figures and conclusions (ready to submit).
- Add residual diagnostic plots and attach a short appendix discussing their outcomes.

---

Prepared by: analysis.py (this repository)

Date run: 2025-11-04
