import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_1samp
from statsmodels.tsa.arima.model import ARIMA

# --- 1. Data Cleaning and Preparation ---

# Load the dataset
df = pd.read_csv('Untitled spreadsheet - Sheet1.csv')

# Convert 'Month' to datetime objects
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')

# Clean and convert numeric columns
for col in ['Volume (in Mn)', 'Value (in Cr.)']:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)

# Sort by date
df = df.sort_values('Month').reset_index(drop=True)

# Define the date ranges
pre_covid_end = pd.to_datetime('2020-03-31')
during_covid_start = pd.to_datetime('2020-04-01')
during_covid_end = pd.to_datetime('2022-01-31')
post_covid_start = pd.to_datetime('2022-08-01')

# Create the dataframes
pre_covid_df = df[df['Month'] <= pre_covid_end]
during_covid_df = df[(df['Month'] >= during_covid_start) & (df['Month'] <= during_covid_end)]
post_covid_df = df[df['Month'] >= post_covid_start]

print("--- Data Cleaning and Preparation Complete ---")


# --- 2. Implement New ARIMA-based Analysis ---

# Create a directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

def run_arima_analysis(train_df, test_df, period_name):
    print(f"\n--- Running ARIMA Analysis for {period_name} ---")

    # Prepare data
    train_vol = train_df.set_index('Month')['Volume (in Mn)']
    test_vol = test_df.set_index('Month')['Volume (in Mn)']

    # Fit ARIMA(1,2,1) model
    model = ARIMA(train_vol, order=(1, 2, 1))
    fitted_model = model.fit()

    # Forecast for the test period
    forecast = fitted_model.predict(start=test_vol.index[0], end=test_vol.index[-1])

    # --- Generate Plot ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_vol.index, train_vol, label='Training Data')
    plt.plot(test_vol.index, test_vol, label='Actual Volume')
    plt.plot(forecast.index, forecast, 'r--', label='ARIMA Forecast')
    plt.title(f'ARIMA Forecast vs. Actual Volume ({period_name})')
    plt.xlabel('Month')
    plt.ylabel('Volume (in Mn)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualizations/arima_forecast_{period_name.lower().replace(" ", "_")}.png')
    plt.close()

    # --- Statistical Test on Residuals ---
    residuals = test_vol - forecast
    t_stat, p_val = ttest_1samp(residuals, 0)

    print(f"\n- Statistical Test on Forecast Residuals ({period_name}):")
    print(f"  - T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  - Conclusion: The difference between forecast and actuals is statistically significant.")
    else:
        print("  - Conclusion: The difference between forecast and actuals is not statistically significant.")

    return residuals

# Part 1: Pre-COVID vs. During-COVID Analysis
pre_vs_during_residuals = run_arima_analysis(pre_covid_df, during_covid_df, "Pre-COVID vs During-COVID")

# Part 2: (Pre+During)-COVID vs. Post-COVID Analysis
pre_and_during_df = pd.concat([pre_covid_df, during_covid_df])
# Reindex to create a continuous time series
full_range = pd.date_range(start=pre_and_during_df['Month'].min(), end=pre_and_during_df['Month'].max(), freq='MS')
pre_and_during_df = pre_and_during_df.set_index('Month').reindex(full_range).reset_index().rename(columns={'index': 'Month'})
# Interpolate the missing values (the gap between during and post)
pre_and_during_df['Volume (in Mn)'] = pre_and_during_df['Volume (in Mn)'].interpolate(method='linear')


pre_during_vs_post_residuals = run_arima_analysis(pre_and_during_df, post_covid_df, "(Pre+During)-COVID vs Post-COVID")

print("\n--- ARIMA Analysis Complete ---")
