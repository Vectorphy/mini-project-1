# Code Explanation for `analysis.py`

This document provides a detailed explanation of the Python script `analysis.py`. The script performs a time series analysis of UPI (Unified Payments Interface) transaction data, focusing on the impact of the COVID-19 pandemic.

## 1. Overview

The script automates the following processes:
- **Data Loading and Cleaning:** Reads the raw data from a CSV file, cleans it, and prepares it for analysis.
- **Data Segmentation:** Divides the data into three distinct periods: Pre-COVID, During-COVID, and Post-COVID.
- **Exploratory Data Analysis (EDA):** Generates summary statistics and a variety of visualizations to uncover trends, patterns, and anomalies in the data.
- **Time Series Forecasting:** Utilizes the ARIMA (Autoregressive Integrated Moving Average) model to forecast transaction volumes.
- **Statistical Testing:** Conducts several hypothesis tests to statistically validate observations.
- **Reporting:** Saves all generated visualizations and the cleaned data into a structured format for easy review.

## 2. Dependencies

The script requires the following Python libraries:
- `pandas`: For data manipulation and analysis.
- `os`: For interacting with the operating system (e.g., creating directories).
- `matplotlib`: For creating static, animated, and interactive visualizations.
- `seaborn`: A high-level interface for drawing attractive and informative statistical graphics.
- `statsmodels`: For statistical modeling, including time series analysis (ARIMA, seasonal decomposition) and hypothesis testing.
- `scipy`: For scientific computing and technical computing (used here for t-tests).
- `numpy`: For numerical operations.

These dependencies can be installed by running:
```bash
pip install -r requirements.txt
```

## 3. Script Breakdown

The script is organized into logical sections, each performing a specific part of the analysis.

### Section 1: Data Cleaning and Preparation
- **Loads Data:** Reads the UPI transaction data from `Untitled spreadsheet - Sheet1.csv`.
- **Data Type Conversion:** Converts the `Month` column to datetime objects and ensures that numerical columns like `Volume (in Mn)` and `Value (in Cr.)` are treated as numbers, removing any special characters like commas.
- **Feature Engineering:** Creates new, more interpretable columns:
  - `Volume (in Bn)`: Transaction volume in billions.
  - `Value (in Trn)`: Transaction value in trillions.
  - `Year` and `Month_Name`: For easier time-based grouping.
  - `Volume Growth Rate (%)` and `Value Growth Rate (%)`: To analyze month-over-month changes.
- **Data Segmentation:** Splits the dataset into three DataFrames based on dates to analyze the impact of the pandemic:
  - `pre_covid_df`: Data up to March 2020.
  - `during_covid_df`: Data from April 2020 to January 2022.
  - `post_covid_df`: Data from August 2022 onwards.

### Section 2: Export Cleaned Data
- This section saves the three segmented DataFrames into a single Excel file named `upi_data_cleaned.xlsx`, with each period in a separate sheet. This makes the cleaned data accessible for external tools or further manual inspection.

### Section 3: Detailed EDA (Exploratory Data Analysis)
This section is further divided into parts to thoroughly explore the data.
- **Summary Statistics:** Calculates and prints descriptive statistics (mean, std, etc.) for each of the three periods.
- **Distributions:** The `plot_eda_distributions` function generates histograms and box plots to visualize the distribution of transaction volumes for each period.
- **Expanded EDA:**
  - **Yearly/Monthly Trends:** Creates heatmaps to show transaction volume and value aggregated by year and month, revealing seasonal patterns.
  - **Outlier Detection:** Identifies and visualizes outliers in the growth rate columns.
  - **Bank Integration Analysis:** Plots the number of banks live on UPI against the transaction volume to see if there is a correlation.
- **Report Visualizations:** Generates a series of plots specifically for inclusion in a final report:
  - Box plots comparing volume and value across the three periods.
  - Line plots for value growth rate and rolling averages.
  - A correlation heatmap of all numerical variables.
  - Time series decomposition plots to separate the trend, seasonal, and residual components of the data.

### Section 4: Time Series Plotting
- **Overall Trend:** Plots the entire history of transaction volume and value on a dual-axis chart, with the "During-COVID" period highlighted.
- **Period-Specific Plots:** Creates separate time series plots for each of the pre, during, and post-COVID periods to allow for a more detailed examination of each phase.

### Section 5: Final Forecasting Analysis
- This is the predictive part of the script. It uses an ARIMA model to forecast future transaction volumes based on historical data.
- **Part 1:** The pre-COVID data is used as a training set to forecast the transaction volumes for the during-COVID period.
- **Part 2:** The combined pre-COVID and during-COVID data is used to forecast the volumes for the post-COVID period.
- For each forecast, the script plots the training data, the actual values, and the forecasted values. It also performs a t-test on the residuals (the differences between actual and forecasted values) to check if the model's errors are statistically significant.

### Section 6: Additional Hypothesis Tests
- This section performs formal statistical tests to add rigor to the analysis:
  - **T-test:** Checks if there is a significant difference in the average transaction volume and value between the pre-COVID and post-onset (during + post) periods.
  - **Chow Test:** A test for structural breaks in the data to determine if the relationship between variables changed significantly after the onset of COVID-19.
  - **Levene Test:** Tests if the volatility (variance) of the transaction volume growth rate is different between the pre-COVID and post-onset periods.

## 4. Functions Explained

### `plot_eda_distributions(df, period_name)`
- **Purpose:** To visualize the distribution of transaction volumes for a given period.
- **Inputs:**
  - `df`: The DataFrame for the period (e.g., `pre_covid_df`).
  - `period_name`: A string to be used in the plot title (e.g., "Pre-COVID").
- **Process:**
  1. Creates a figure with two subplots.
  2. The first subplot is a histogram with a Kernel Density Estimate (KDE) line to show the shape of the distribution.
  3. The second subplot is a box plot, which is useful for identifying quartiles and potential outliers.
  4. Saves the combined plot as a PNG file.

### `run_arima_analysis(train_df, test_df, period_name, arima_order)`
- **Purpose:** To perform a time series forecast using the ARIMA model.
- **Inputs:**
  - `train_df`: The DataFrame containing the training data.
  - `test_df`: The DataFrame containing the data to be forecasted (the "actuals").
  - `period_name`: A string for the plot title.
  - `arima_order`: A tuple `(p, d, q)` representing the parameters of the ARIMA model.
    - `p`: The order of the autoregressive (AR) part.
    - `d`: The degree of differencing (I for Integrated).
    - `q`: The order of the moving average (MA) part.
- **Process:**
  1. Sets up the training and testing time series from the input DataFrames.
  2. Initializes an ARIMA model with the training data and the specified order.
  3. Fits the model to the data.
  4. Generates a forecast for the same time period as the test data.
  5. Creates a plot comparing the training data, actual data, and the ARIMA forecast.
  6. Saves the plot as a PNG file.
  7. Calculates the residuals and performs a one-sample t-test to evaluate the model's performance.

## 5. How to Run the Script

1.  Make sure you have Python and `pip` installed.
2.  Place the `analysis.py` script and the `Untitled spreadsheet - Sheet1.csv` file in the same directory.
3.  Install the required dependencies using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4.  Run the script from your terminal:
    ```bash
    python analysis.py
    ```
5.  After execution, a new directory named `visualizations` will be created containing all the generated plots, and a file named `upi_data_cleaned.xlsx` will contain the cleaned, segmented data. The console will display the results of the statistical tests and summary statistics.