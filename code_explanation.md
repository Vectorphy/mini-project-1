# Code Explanation for `analysis.py`

## Overview

This script performs a comprehensive time series analysis of UPI (Unified Payments Interface) transaction data. The primary goal is to analyze transaction trends, particularly in relation to the COVID-19 pandemic, and to forecast future transaction volumes. The analysis is divided into several key stages: data cleaning, exploratory data analysis (EDA), time series forecasting using the SARIMA model, and statistical hypothesis testing.

## Script Breakdown

### 1. Data Cleaning and Preparation

- **Loading Data**: The script begins by loading the UPI transaction data from a CSV file named `Untitled spreadsheet - Sheet1.csv`.
- **Data Type Conversion**: It converts the `Month` column to a datetime format and ensures that numeric columns like `Volume (in Mn)` and `Value (in Cr.)` are correctly parsed as floats, removing any commas.
- **Feature Engineering**: New columns are created for easier analysis, such as:
    - `Volume (in Bn)`: Transaction volume in billions.
    - `Value (in Trn)`: Transaction value in trillions.
    - `Year` and `Month_Name`: For temporal analysis.
    - `Volume Growth Rate (%)` and `Value Growth Rate (%)`: To measure month-over-month changes.
- **Data Segmentation**: The data is split into three periods based on the COVID-19 timeline:
    - **Pre-COVID**: Data up to March 31, 2020.
    - **During-COVID**: Data from April 1, 2020, to January 31, 2022.
    - **Post-COVID**: Data from August 1, 2022, onwards.
- **Exporting Cleaned Data**: The cleaned and segmented data is exported to an Excel file `upi_data_cleaned.xlsx` with separate sheets for each period.

### 2. Exploratory Data Analysis (EDA)

- **Summary Statistics**: The script calculates and prints summary statistics (like mean, median, std dev) for transaction volume and value for each of the three periods.
- **Visualizations**: A `visualizations` directory is created if it doesn't exist. The script generates several plots to understand the data's characteristics:
    - **Distribution Plots**: Histograms and box plots to show the distribution of transaction volumes.
    - **Trend Analysis**: Heatmaps to visualize yearly and monthly trends in transaction volume and value.
    - **Outlier Detection**: Scatter plots to identify outliers in the growth rates.
    - **Bank Integration Analysis**: A dual-axis line chart to compare the growth in transaction volume with the number of banks live on UPI.
    - **Correlation Heatmap**: A heatmap to show the correlation between different numeric variables in the dataset.
    - **Time Series Decomposition**: Plots that break down the time series into trend, seasonal, and residual components for both volume and value.

### 3. Time Series Forecasting

- **SARIMA Model**: The script uses the Seasonal AutoRegressive Integrated Moving Average (SARIMA) model for forecasting. SARIMA is suitable for time series data with a clear seasonal pattern.
- **Forecasting Periods**: Two separate forecasting analyses are performed:
    1.  **Pre-COVID vs. During-COVID**: The model is trained on the Pre-COVID data to forecast the transaction volumes for the During-COVID period.
    2.  **(Pre+During)-COVID vs. Post-COVID**: The model is trained on the combined Pre-COVID and During-COVID data to forecast the transaction volumes for the Post-COVID period.
- **`run_sarima_analysis` function**: This function encapsulates the forecasting logic.

### 4. Additional Hypothesis Tests

- **T-tests**: The script performs independent t-tests to check if there is a statistically significant difference in the average transaction volume and value between the Pre-COVID and post-onset (During + Post-COVID) periods.
- **Chow Test**: A Chow test is used to detect if there is a structural break in the data, which would indicate that the pandemic significantly changed the underlying trend of UPI transactions.
- **Levene Test**: This test is used to check if the volatility (variance) of the transaction volume growth rate is different between the Pre-COVID and post-onset periods.

## Function Definitions

### `plot_eda_distributions(df, period_name)`

- **Purpose**: To create and save histogram and box plots for the transaction volume of a given period.
- **Parameters**:
    - `df`: The DataFrame for the period (e.g., `pre_covid_df`).
    - `period_name`: A string name for the period (e.g., 'Pre-COVID') used in plot titles and filenames.

### `run_sarima_analysis(train_df, test_df, period_name, order, seasonal_order)`

- **Purpose**: To perform SARIMA forecasting, plot the results, and evaluate the forecast.
- **Parameters**:
    - `train_df`: The DataFrame used for training the model.
    - `test_df`: The DataFrame used for testing the model (i.e., the period to be forecasted).
    - `period_name`: A string describing the analysis period for titles and filenames.
    - `order`: A tuple `(p, d, q)` for the non-seasonal components of the SARIMA model.
    - `seasonal_order`: A tuple `(P, D, Q, s)` for the seasonal components of the SARIMA model, where `s` is the seasonal period (typically 12 for monthly data).
- **Functionality**:
    1.  Sets up the training and testing data from the input DataFrames.
    2.  Initializes and fits a `SARIMAX` model with the specified orders.
    3.  Generates a forecast for the duration of the test period.
    4.  Plots the training data, actual test data, and the SARIMA forecast on a single graph.
    5.  Calculates the residuals (the difference between the forecast and actual values) and performs a one-sample t-test to check if the mean of the residuals is significantly different from zero. A non-significant p-value suggests a good model fit.
    6.  Saves the plot to the `visualizations` directory.