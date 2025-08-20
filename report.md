# Analysis of UPI Transaction Growth using a Hybrid Forecasting Approach

## 1. Introduction

This report details a comprehensive analysis of UPI transaction data to understand the impact of the COVID-19 pandemic on its growth. This final version of the analysis uses a hybrid forecasting approach as requested:
1.  **Holt-Winters Model**: Used to analyze the `Pre-COVID vs. During-COVID` periods.
2.  **ARIMA(2,2,2) Model**: Used to analyze the `(Pre+During)-COVID vs. Post-COVID` periods.

This approach allows for a nuanced view of the pandemic's impact. Additionally, several hypothesis tests, including a new test for changes in volatility, are conducted to provide statistical backing for the findings.

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

## 4. Hybrid Forecasting Analysis

### Part 1: Pre-COVID vs. During-COVID (Holt-Winters)

A Holt-Winters model, suitable for handling data with trend and seasonality, was trained on the Pre-COVID data to forecast the During-COVID period.

![Holt-Winters Forecast for During-COVID](visualizations/hw_forecast_pre-covid_vs_during-covid.png)

- **Statistical Test on Residuals:** A t-test on the forecast residuals yielded a **T-statistic of 5.4782** and a **p-value of 0.0000**.
- **Conclusion:** The actual transaction volumes during the pandemic were **statistically significantly different** from what the Holt-Winters model predicted based on the pre-COVID trend. This confirms a major shift in user behavior at the onset of the pandemic.

### Part 2: (Pre+During)-COVID vs. Post-COVID (ARIMA)

A new ARIMA(2,2,2) model was trained on the combined Pre-COVID and During-COVID data. This model, representing the "new normal" trend established during the pandemic, was then used to forecast the Post-COVID period.

![ARIMA Forecast for Post-COVID](visualizations/arima_forecast_(pre+during)-covid_vs_post-covid.png)

- **Statistical Test on Residuals:** A t-test on the residuals of this second forecast yielded a **T-statistic of 5.6291** and a **p-value of 0.0000**.
- **Conclusion:** The growth in the post-COVID era **still significantly outpaced the forecast** based on the trend established during the pandemic. This suggests that the accelerated growth has continued to gain momentum. *(Note: A convergence warning was issued during model fitting for this part.)*

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
1.  **Massive, Sustained Acceleration:** The pandemic triggered a massive, statistically significant acceleration in UPI growth that exceeded the forecasts of both a pre-COVID Holt-Winters model and a (Pre+During)-COVID ARIMA model. The growth did not level off but continued to accelerate.
2.  **Fundamental Shift:** The structural break identified by the Chow test and the significant difference in average transactions confirm that the pandemic fundamentally altered UPI usage patterns.
3.  **Increased Volatility:** The significant result of the Levene test shows that the growth has also become more volatile since the pandemic's onset.
