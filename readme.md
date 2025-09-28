# Analysis of UPI Transaction Growth using Holt-Winters Forecasting

## 1. Introduction

This report details a comprehensive analysis of UPI transaction data to understand the impact of the COVID-19 pandemic on its growth. This analysis uses the **Holt-Winters** forecasting model to analyze transaction trends across different periods.

Several hypothesis tests, including tests for structural breaks and changes in volatility, are conducted to provide statistical backing for the findings.

For a detailed breakdown of the Python script (`analysis.py`) used for this analysis, please refer to the `code_explanation.md` file.

## 2. Data Cleaning and Preparation

The raw data was cleaned and prepared for analysis. This involved converting data types and splitting the data into three periods:
- **Pre-COVID**: Data up to March 2020.
- **During-COVID**: Data from April 2020 to January 2022.
- **Post-COVID**: Data from August 2022 onwards.
The cleaned data was exported to `upi_data_cleaned.xlsx`.

## 3. Exploratory Data Analysis (EDA)

### Data Distributions
Histograms and box plots were generated for each period to visualize the distribution of transaction volumes.

**Pre-COVID:**
![Pre-COVID EDA](visualizations/eda_distributions_pre-covid.png)

**During-COVID:**
![During-COVID EDA](visualizations/eda_distributions_during-covid.png)

**Post-COVID:**
![Post-COVID EDA](visualizations/eda_distributions_post-covid.png)

These plots show a clear shift in the distribution to higher values in each subsequent period.

### Time Series Trends
To visualize the overall trend, a time series plot of the entire dataset was created.

![Full Time Series Plot](visualizations/time_series_full.png)

The plot clearly shows an exponential growth trend in both transaction volume and value over time, with a noticeable acceleration around the "During-COVID" period (marked in red).

## 4. Forecasting Analysis

### Part 1: Pre-COVID vs. During-COVID

A Holt-Winters model was trained on the Pre-COVID data to forecast the During-COVID period.

![Holt-Winters Forecast for During-COVID](visualizations/hw_forecast_pre-covid_vs_during-covid.png)

- **Statistical Test on Residuals:** A t-test on the forecast residuals yielded a **T-statistic of 5.4782** and a **p-value of 0.0000**.
- **Conclusion:** The actual transaction volumes during the pandemic were **statistically significantly different** from what the Holt-Winters model predicted based on the pre-COVID trend. This confirms a major shift in user behavior.

### Part 2: (Pre+During)-COVID vs. Post-COVID

The Holt-Winters model was then trained on the combined Pre-COVID and During-COVID data to forecast the Post-COVID period.

![Holt-Winters Forecast for Post-COVID](visualizations/hw_forecast_(pre+during)-covid_vs_post-covid.png)

- **Statistical Test on Residuals:** A t-test on the residuals of this second forecast yielded a **T-statistic of -2.6766** and a **p-value of 0.0112**.
- **Conclusion:** The growth in the post-COVID era **still significantly differed from the forecast** based on the trend established during the pandemic, suggesting that the growth dynamics continued to evolve.

## 5. Additional Hypothesis Tests

### Test for Difference in Average Transactions
A two-sample t-test confirmed that the average transaction volume and value are **statistically significantly higher** in the post-onset period compared to the pre-COVID period (Volume t-statistic: -11.0504, Value t-statistic: -12.6223, p-value = 0.0000 for both).

### Chow Test for Structural Break
A Chow test confirmed a **statistically significant structural break** in the data at the onset of the pandemic (p-value = 0.0000), indicating a fundamental shift in the data's properties.

### Levene Test for Change in Volatility
A Levene test was conducted to compare the variance of the monthly growth rates before and after the pandemic's onset.
- **Levene Statistic:** 5.1980, **P-value:** 0.0248
- **Conclusion:** The p-value is less than 0.05, indicating that the **volatility (variance) of the growth rate is statistically significantly different** between the two periods.

## 6. Final Conclusion

This comprehensive analysis provides several key insights into the impact of the COVID-19 pandemic on UPI transactions:
1.  **Massive, Sustained Acceleration:** The pandemic triggered a massive, statistically significant acceleration in UPI growth that exceeded the forecasts of the Holt-Winters model. The growth did not level off but continued to accelerate.
2.  **Fundamental Shift:** The structural break identified by the Chow test and the significant difference in average transactions confirm that the pandemic fundamentally altered UPI usage patterns.
3.  **Increased Volatility:** The significant result of the Levene test shows that the growth has also become more volatile since the pandemic's onset.
---
