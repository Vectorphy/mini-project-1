# Analysis of UPI Transaction Growth Using SARIMA Forecasting

**Pranav D. Dudhal (1272250578)**
*MIT-WPU*

**Guides:**
Mr. Sachin A. Naik
Mr. Vishal Pawar

**October 2025**

## Abstract

This research analyzes the impact of the COVID-19 pandemic on India’s Unified Payments Interface (UPI) transaction growth. The primary objective is to determine whether the post-pandemic surge in UPI usage constitutes a structural break from pre-existing trends or is merely a continuation of its organic growth. The study utilizes a quantitative approach, applying Seasonal AutoRegressive Integrated Moving Average (SARIMA) forecasting models to monthly UPI transaction data obtained from the National Payments Corporation of India (NPCI). By generating counterfactual predictions, the research compares transaction volumes across three distinct periods: Pre-COVID, During-COVID, and Post-COVID. Statistical validation is performed using t-tests, Chow tests, and Levene tests to assess the significance of any observed deviations. The results reveal a significant acceleration in UPI adoption during the pandemic, followed by a period of stabilization in the post-COVID era. This suggests a fundamental and lasting shift in digital financial behavior among Indian consumers, driven by the unique circumstances of the pandemic.

**Keywords:** UPI, SARIMA, Time Series Forecasting, COVID-19, Digital Payments, India, Fintech Growth, Statistical Modeling.

## 1. Introduction

The Unified Payments Interface (UPI) has rapidly emerged as India's dominant digital payment platform, revolutionizing the country's financial landscape. Its seamless, real-time, and interoperable architecture has driven unprecedented growth in digital transactions. The onset of the COVID-19 pandemic in 2020 served as a significant catalyst, accelerating the global trend of digital adoption. In India, this manifested as a widespread shift towards cashless payments, as social distancing norms and health concerns pushed even traditionally cash-reliant users, including those in rural and semi-urban areas, to embrace digital alternatives.

While numerous studies have qualitatively confirmed this surge in digital payments during the pandemic, a significant gap remains in the quantitative analysis of this growth. Most prior research has focused on the immediate impact of the pandemic but has not rigorously measured the extent to which this growth deviated from pre-pandemic trends. It remains unclear whether the observed increase was a temporary spike or a permanent "structural break" in India's digital payment ecosystem.

This research aims to address this gap by using time-series forecasting to quantitatively measure the pandemic-induced shifts in UPI transaction volumes. By developing a counterfactual scenario—what UPI growth might have looked like without the pandemic—we can isolate and quantify the impact of this unprecedented event. The scope of this study covers UPI transaction data across three distinct periods: pre-COVID (before March 2020), during-COVID (April 2020 to January 2022), and post-COVID (from August 2022 onwards).

The significance of this study lies in its potential to provide empirical evidence for policymakers, financial institutions, and fintech companies. By understanding the true nature of the pandemic's impact on digital payment adoption, stakeholders can make more informed decisions regarding digital policy planning, financial inclusion strategies, and infrastructure development. This research offers a robust, data-driven approach to understanding one of the most significant economic transformations in recent Indian history.

## 2. Literature Review

The existing literature broadly confirms the positive impact of the COVID-19 pandemic on digital payment adoption in India and globally. However, most studies have relied on descriptive or qualitative methods, leaving a gap in quantitative, model-driven analysis.

Studies such as C. Chaudhari & A. Kumar (2021) identified a significant rise in digital payments during the pandemic, noting a 42% increase in transaction volumes. While valuable, their work lacked predictive modeling to assess whether this growth was an anomaly or an acceleration of existing trends. Similarly, Jain & Chowdhary (2022) explored the behavioral aspects, focusing on consumer intentions to adopt digital payment systems under the pressure of the pandemic. Their findings highlight the psychological shift, but do not quantify the resulting market-level impact. Rajat & Nirolia (2020) analyzed the adaptation challenges for consumers and businesses during the initial phases of the pandemic, but their research was based on small-scale survey data, limiting its generalizability.

In a global context, papers by Hasanul Banna & Md. Alam (2021) and Schilirò (2020) discuss the broader trends of digital globalization and financial inclusion, accelerated by the pandemic. These studies provide a valuable macro-level perspective but do not offer a granular analysis of specific payment systems like UPI.

The primary gap identified in the literature is the distinction between correlation and causality. While it is widely accepted that digital payment usage surged *during* the pandemic, no studies have quantitatively modeled what the growth trajectory would have been *without* the pandemic. This research fills that critical gap by employing SARIMA-based counterfactual forecasting. By modeling the pre-pandemic trend and projecting it into the pandemic and post-pandemic periods, this study statistically isolates the effect of COVID-19, providing a more rigorous, evidence-based assessment of its impact on UPI's growth.

## 3. Research Methodology

This study employs a quantitative, longitudinal time-series design to analyze the growth of UPI transactions. The methodology is structured to isolate the impact of the COVID-19 pandemic by comparing actual data with counterfactual forecasts.

### 3.1 Data Source and Periods
The research utilizes secondary data from the National Payments Corporation of India (NPCI), which provides publicly available monthly reports on UPI transaction volume and value. The data is segmented into three distinct periods to facilitate a phased analysis:
-   **Pre-COVID:** All available data up to and including March 2020.
-   **During-COVID:** April 2020 to January 2022.
-   **Post-COVID:** August 2022 onwards.

### 3.2 Tools and Libraries
The analysis is conducted using Python, with the following key libraries:
-   **pandas:** For data manipulation and cleaning.
-   **statsmodels:** For SARIMA model implementation and statistical tests.
-   **matplotlib and seaborn:** For data visualization.
-   **Excel:** Used for initial data cleaning and validation.

### 3.3 Process
The research process is structured as follows:
1.  **Data Cleaning and Segmentation:** The raw data from NPCI is cleaned to handle any inconsistencies and then segmented into the three defined periods (Pre-COVID, During-COVID, Post-COVID).
2.  **Exploratory Data Analysis (EDA):** EDA is performed to understand the underlying trends, seasonality, and statistical properties of the data. This includes generating statistical summaries and visualizations.
3.  **SARIMA Modeling and Forecasting:** Two SARIMA models are trained to generate counterfactual forecasts:
    *   **Model 1:** Trained on the Pre-COVID data, this model forecasts the expected transaction volumes for the During-COVID period, assuming the pre-pandemic trend had continued.
    *   **Model 2:** Trained on the combined Pre-COVID and During-COVID data, this model forecasts the expected transaction volumes for the Post-COVID period.
4.  **Statistical Testing:** The forecasted values are compared with the actual observed values. The deviations are statistically tested using:
    *   **Two-sample t-test:** To compare the means of the different periods.
    *   **Chow test:** To identify structural breaks in the time series.
    *   **Levene test:** To check for changes in variance.
5.  **Interpretation:** The results of the statistical tests are interpreted to determine whether the pandemic induced a structural change in UPI transaction growth.

### 3.4 Validation
The robustness of the SARIMA models is validated through standard diagnostic procedures. This includes minimizing the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) for model selection and analyzing residual plots to ensure that the model residuals are white noise, indicating a good fit.

## 4. Data Analysis and Results

The data analysis reveals a clear and statistically significant shift in UPI transaction patterns following the onset of the COVID-19 pandemic.

### 4.1 EDA Insights
Exploratory Data Analysis (EDA) provides a preliminary view of the dramatic growth in UPI usage:
-   **Mean Transaction Value:** The average monthly transaction value surged from ₹0.65 trillion in the Pre-COVID period to ₹18.25 trillion in the Post-COVID period.
-   **Distributional Shift:** The distribution of transaction volumes and values shifted significantly upward across the three periods, as illustrated by boxplots.
-   **Correlation:** A strong positive correlation of +0.99 was observed between transaction volume and value, indicating consistent and synchronized growth.

[Insert Figure 1: Full Time Series Plot with Pandemic Markers]

[Insert Figure 2: Boxplots of Transaction Values by Period]

[Insert Figure 3: Correlation Heatmap]

### 4.2 SARIMA Analysis
The core of the analysis involved comparing the actual UPI transaction data with the forecasts generated by the SARIMA models.

-   **Model 1 (Pre-COVID trained → During-COVID forecast):**
    The model, trained on pre-pandemic data, significantly under-predicted the actual transaction volumes during the COVID period. The deviation between the forecasted and actual values was statistically significant, with a t-statistic of 5.1217 (p < 0.0001). This indicates that the growth during the pandemic was far greater than what would have been expected based on pre-existing trends.

[Insert Figure 4: SARIMA Forecast (Model 1) vs. Actual Data]

-   **Model 2 (Pre+During trained → Post-COVID forecast):**
    This model, trained on data that included the pandemic-induced surge, slightly over-predicted the growth in the post-COVID period. The actual post-COVID values were found to be slightly below the forecasted trend, with a t-statistic of -5.3953 (p < 0.0001). This suggests a stabilization or maturation of the growth rate after the intense acceleration phase.

[Insert Figure 5: SARIMA Forecast (Model 2) vs. Actual Data]

### 4.3 Hypothesis Tests
The results of the hypothesis tests confirm the findings of the SARIMA analysis:
-   **Two-sample t-test:** The difference in mean transaction volumes between the pre- and post-COVID periods was found to be statistically significant (p < 0.001), confirming a major increase in UPI usage.
-   **Chow test:** The Chow test detected a structural break in the time series data corresponding to the onset of the pandemic (p < 0.001). This provides strong evidence that the pandemic fundamentally altered the growth trajectory of UPI transactions.
-   **Levene test:** The Levene test indicated a significant change in variance between the periods (p = 0.0248), further supporting the conclusion of a structural shift.

## 5. Discussion

The results of this study provide strong quantitative evidence that the COVID-19 pandemic acted as a structural shock to India's digital payment ecosystem, creating a statistically proven acceleration in UPI growth. The significant deviation between the actual and forecasted transaction volumes during the pandemic period (as shown by Model 1) confirms that the observed surge was not merely a continuation of the pre-existing trend but a fundamental shift in consumer behavior.

This digital shift can be attributed to a combination of behavioral, infrastructural, and policy-driven factors. The lockdown measures and social distancing norms made digital payments a necessity, breaking long-standing habits of cash dependency. This was supported by a robust and scalable UPI infrastructure and policy initiatives that encouraged digital transactions.

The findings from Model 2, which show a moderation in growth post-COVID, are equally insightful. This does not suggest a decline in UPI's relevance but rather a maturation of the market. After a period of hyper-growth, the rate of new user adoption is naturally stabilizing. The system has reached a new, much higher baseline, and future growth is likely to be more incremental.

This research contributes to the existing literature by confirming the qualitative claims of prior studies (such as Chaudhari & Kumar, 2021) with rigorous quantitative evidence. The use of counterfactual forecasting, combined with statistical hypothesis testing (t-tests, Chow test), strengthens the causal inference that the pandemic was a primary driver of this transformation. By isolating the pandemic's impact, this study provides a clearer understanding of the magnitude of its effect on India's journey towards a digital economy.

## 6. Conclusion

This research demonstrates that the COVID-19 pandemic was a pivotal event that permanently and structurally altered India’s digital payment landscape. The analysis of UPI transaction data shows a significant deviation from the pre-pandemic trend, confirming that the surge in digital payments was not merely an acceleration of organic growth but a fundamental shift in consumer behavior.

In the post-pandemic era, UPI transaction growth continues, albeit at a more stabilized rate. This indicates a maturing market, where the focus is shifting from rapid user acquisition to deepening engagement. The findings of this study validate the use of forecasting methods like SARIMA as reliable tools for economic impact evaluation, providing a quantitative basis for what has, until now, been largely a qualitative assessment.

For future research, more advanced forecasting models such as Prophet, Long Short-Term Memory (LSTM) networks, or hybrid machine learning models could be employed for extended forecasting and real-time analysis. Such studies could further enhance our understanding of the evolving dynamics of digital finance in a post-pandemic world.

## References

Chaudhari, C., & Kumar, A. (2021). *Impact of COVID-19 on Digital Payment in India*.
Hasanul Banna, H., & Alam, M. R. (2021). Digital Financial Inclusion and Banking Stability.
Hyndman, R. J., & Athanasopoulos, G. (2023). *Forecasting: Principles and Practice* (3rd ed.). OTexts.
Jain, K., & Chowdhary, R. (2022). *Intention to Adopt Digital Payment Systems in India*.
National Payments Corporation of India. (n.d.). *UPI Product Statistics*. Retrieved from https://www.npci.org.in/what-we-do/upi/product-statistics
Schilirò, D. (2020). Towards Digital Globalization and the COVID-19 Challenge.

## Appendices

### Appendix A: Cleaned Data Summary
A summary of the cleaned UPI transaction data used for this analysis is available in the supplementary file: `upi_data_cleaned.xlsx`.

### Appendix B: SARIMA Model Configuration Parameters
The parameters for the SARIMA models were determined through an iterative process of minimizing AIC/BIC values. The final parameters used are as follows:

-   **Model 1 (Pre-COVID):**
    -   `p` (non-seasonal AR order): 1
    -   `d` (non-seasonal differencing): 1
    -   `q` (non-seasonal MA order): 1
    -   `P` (seasonal AR order): 1
    -   `D` (seasonal differencing): 1
    -   `Q` (seasonal MA order): 1
    -   `s` (seasonality): 12

-   **Model 2 (Pre+During-COVID):**
    -   `p`: 1
    -   `d`: 1
    -   `q`: 0
    -   `P`: 1
    -   `D`: 1
    -   `Q`: 1
    -   `s`: 12

### Appendix C: Python Code Snippet for SARIMA Fitting and Forecast
```python
import pandas as pd
import statsmodels.api as sm

# Load data
data = pd.read_excel("upi_data_cleaned.xlsx", index_col="Date", parse_dates=True)

# Define pre-COVID training data
train_pre_covid = data[data.index < "2020-04-01"]["TransactionValue"]

# Fit SARIMA Model 1
model1 = sm.tsa.statespace.SARIMAX(
    train_pre_covid,
    order=(1, 1, 1),
    seasonal_order=(1, 1, 1, 12)
)
results1 = model1.fit(disp=False)

# Forecast for the During-COVID period
forecast1 = results1.get_forecast(steps=22) # April 2020 to Jan 2022
```

### Appendix D: Extended Visualizations
Extended visualizations, including seasonal decomposition plots and residual analysis for both SARIMA models, are available in the supplementary materials. These plots confirm the validity of the models used.
