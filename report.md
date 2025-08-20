# Analysis of UPI Transaction Growth using ARIMA Model

## 1. Introduction

This report details the analysis of UPI transaction data to understand the impact of the COVID-19 pandemic on its growth. This version of the analysis uses an ARIMA(1,2,1) model to forecast transaction volumes and is structured into two distinct comparisons as requested:
1.  **Pre-COVID vs. During-COVID**: An ARIMA model is trained on pre-COVID data to forecast the during-COVID period.
2.  **(Pre+During)-COVID vs. Post-COVID**: A second ARIMA model is trained on the combined pre-COVID and during-COVID data to forecast the post-COVID period.

This approach allows us to assess the impact of the pandemic by looking at the deviation from the pre-pandemic trend, and then to see if a "new normal" was established by the time the post-COVID period began.

## 2. Data Cleaning and Preparation

The raw data was cleaned and prepared for analysis. This involved converting data types and splitting the data into three periods:
- **Pre-COVID**: Data up to March 2020.
- **During-COVID**: Data from April 2020 to January 2022.
- **Post-COVID**: Data from August 2022 onwards.

## 3. ARIMA-Based Time Series Analysis

An ARIMA(1,2,1) model was used for the time series analysis, with the following structure.

### Part 1: Pre-COVID vs. During-COVID Analysis

An ARIMA(1,2,1) model was trained on the Pre-COVID data and used to forecast the transaction volumes for the During-COVID period.

![ARIMA Forecast for During-COVID](visualizations/arima_forecast_pre-covid_vs_during-covid.png)

- **Statistical Test on Residuals:** A t-test was performed on the residuals (the difference between the actual and forecasted values).
    - **T-statistic:** 6.4029, **P-value:** 0.0000
- **Conclusion:** The p-value is less than 0.05, indicating that the actual transaction volumes during the pandemic were **statistically significantly higher** than what the pre-COVID trend predicted. This confirms that the pandemic acted as a major accelerating event for UPI adoption.

### Part 2: (Pre+During)-COVID vs. Post-COVID Analysis

A new ARIMA(1,2,1) model was trained on the combined Pre-COVID and During-COVID data. This model, representing the "new normal" trend established during the pandemic, was then used to forecast the Post-COVID period.

![ARIMA Forecast for Post-COVID](visualizations/arima_forecast_(pre+during)-covid_vs_post-covid.png)

- **Statistical Test on Residuals:** A t-test was performed on the residuals of this second forecast.
    - **T-statistic:** 9.9186, **P-value:** 0.0000
- **Conclusion:** The p-value is again less than 0.05. This result is very interesting. It shows that even after establishing a new, much steeper trend during the pandemic, the growth in the post-COVID era **still significantly outpaced the forecast**.

## 4. Final Conclusion

This analysis, using an ARIMA(1,2,1) model, provides two key insights:
1.  The COVID-19 pandemic caused a massive, statistically significant acceleration in UPI transaction growth, far exceeding the pre-pandemic trend.
2.  This accelerated growth did not just establish a new trend and level off. The growth in the post-COVID period continued to accelerate even beyond the new, steeper trend established during the pandemic itself. This suggests that the factors driving UPI adoption have continued to intensify, even after the initial shock of the pandemic.
