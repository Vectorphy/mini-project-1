import pandas as pd
import os

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


# --- 2. Export Cleaned Data ---

with pd.ExcelWriter('upi_data_cleaned.xlsx') as writer:
    pre_covid_df.to_excel(writer, sheet_name='Pre-COVID', index=False)
    during_covid_df.to_excel(writer, sheet_name='During-COVID', index=False)
    post_covid_df.to_excel(writer, sheet_name='Post-COVID', index=False)

print("--- Cleaned data exported to 'upi_data_cleaned.xlsx' ---")


# --- 3. Exploratory Data Analysis (EDA) ---

print("\n--- Exploratory Data Analysis (Summary Statistics) ---")

def perform_eda(df, period_name):
    print(f"\n--- Summary Statistics for {period_name} ---")
    print(df[['Volume (in Mn)', 'Value (in Cr.)']].describe())

perform_eda(pre_covid_df, 'Pre-COVID')
perform_eda(during_covid_df, 'During-COVID')
perform_eda(post_covid_df, 'Post-COVID')

import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import ttest_1samp

# --- 4. Time Series Plotting ---

# Create a directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Plot for the entire dataset
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(df['Month'], df['Volume (in Mn)'], 'g-', label='Volume (in Mn)')
ax1.set_xlabel('Month')
ax1.set_ylabel('Volume (in Mn)', color='g')
ax1.tick_params('y', colors='g')
ax2 = ax1.twinx()
ax2.plot(df['Month'], df['Value (in Cr.)'], 'b-', label='Value (in Cr.)')
ax2.set_ylabel('Value (in Cr.)', color='b')
ax2.tick_params('y', colors='b')
plt.title('UPI Transaction Volume and Value Over Time')
plt.axvspan(during_covid_start, during_covid_end, color='red', alpha=0.15, label='During-COVID')
plt.legend(handles=[ax1.get_lines()[0], ax2.get_lines()[0]])
fig.tight_layout()
plt.savefig('visualizations/time_series_full.png')
plt.close()

# Separate plots for each period
for period_df, period_name in [(pre_covid_df, 'Pre-COVID'), (during_covid_df, 'During-COVID'), (post_covid_df, 'Post-COVID')]:
    fig, ax1 = plt.subplots(figsize=(10, 5))
    p1, = ax1.plot(period_df['Month'], period_df['Volume (in Mn)'], 'g-', label='Volume (in Mn)')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Volume (in Mn)', color='g')
    ax1.tick_params('y', colors='g')
    ax2 = ax1.twinx()
    p2, = ax2.plot(period_df['Month'], period_df['Value (in Cr.)'], 'b-', label='Value (in Cr.)')
    ax2.set_ylabel('Value (in Cr.)', color='b')
    ax2.tick_params('y', colors='b')
    plt.title(f'UPI Transactions ({period_name})')
    plt.legend(handles=[p1, p2])
    fig.tight_layout()
    plt.savefig(f'visualizations/time_series_{period_name.lower()}.png')
    plt.close()

print("\n--- Time Series Plots Generated ---")


# --- 5. Implement Hybrid Forecasting Analysis ---

# --- Part 1: ARIMA for Pre-COVID vs. During-COVID ---

def run_arima_analysis(train_df, test_df, period_name, arima_order):
    print(f"\n--- Running ARIMA Analysis for {period_name} ---")

    # Prepare data
    train_vol = train_df.set_index('Month')['Volume (in Mn)']
    test_vol = test_df.set_index('Month')['Volume (in Mn)']

    # Fit ARIMA model
    model = ARIMA(train_vol, order=arima_order)
    fitted_model = model.fit()

    # Forecast
    forecast = fitted_model.predict(start=test_vol.index[0], end=test_vol.index[-1])

    # Generate Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_vol.index, train_vol, label='Training Data')
    plt.plot(test_vol.index, test_vol, label='Actual Volume')
    plt.plot(forecast.index, forecast, 'r--', label='ARIMA Forecast')
    plt.title(f'ARIMA Forecast vs. Actual ({period_name})')
    plt.xlabel('Month')
    plt.ylabel('Volume (in Mn)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualizations/arima_forecast_{period_name.lower().replace(" ", "_")}.png')
    plt.close()

    # Statistical Test on Residuals
    residuals = test_vol - forecast
    t_stat, p_val = ttest_1samp(residuals, 0)
    print(f"\n- Test on Residuals: T-statistic={t_stat:.4f}, P-value={p_val:.4f}")

run_arima_analysis(pre_covid_df, during_covid_df, "Pre-COVID vs During-COVID", arima_order=(2, 2, 2))


# --- Part 2: ARIMA for (Pre+During)-COVID vs. Post-COVID ---

# Prepare combined dataframe for ARIMA training
pre_and_during_df = pd.concat([pre_covid_df, during_covid_df])
full_range = pd.date_range(start=pre_and_during_df['Month'].min(), end=pre_and_during_df['Month'].max(), freq='MS')
pre_and_during_df = pre_and_during_df.set_index('Month').reindex(full_range).reset_index().rename(columns={'index': 'Month'})
pre_and_during_df['Volume (in Mn)'] = pre_and_during_df['Volume (in Mn)'].interpolate(method='linear')

run_arima_analysis(pre_and_during_df, post_covid_df, "(Pre+During)-COVID vs Post-COVID", arima_order=(2, 2, 2))

from statsmodels.formula.api import ols
from scipy.stats import levene

# --- 5. Conduct Additional Hypothesis Tests ---

print("\n--- Additional Hypothesis Tests ---")

from scipy.stats import ttest_ind

# Test for difference in average transactions
print("\n--- Test for Difference in Average Transactions ---")
post_onset_df = pd.concat([during_covid_df, post_covid_df])
for col in ['Volume (in Mn)', 'Value (in Cr.)']:
    t_stat, p_val = ttest_ind(pre_covid_df[col], post_onset_df[col], equal_var=False)
    print(f"\n- {col}: T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

# Chow test for structural break
print("\n--- Chow Test for Structural Break ---")
df['time_index'] = range(len(df))
df['dummy'] = (df['Month'] > pre_covid_end).astype(int)
df['time_dummy'] = df['time_index'] * df['dummy']
model = ols('Q("Volume (in Mn)") ~ time_index + dummy + time_dummy', data=df).fit()
f_test_result = model.f_test('dummy = 0, time_dummy = 0')
print(f"\n- F-statistic: {f_test_result.fvalue:.4f}, P-value: {f_test_result.pvalue:.4f}")

import numpy as np

# Levene test for change in volatility
print("\n--- Levene Test for Change in Volatility ---")
pre_covid_growth = pre_covid_df['Volume (in Mn)'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
post_onset_growth = post_onset_df['Volume (in Mn)'].pct_change().dropna()
levene_stat, levene_p_val = levene(pre_covid_growth, post_onset_growth)
print(f"\n- Levene statistic: {levene_stat:.4f}, P-value: {levene_p_val:.4f}")


print("\n--- Additional Hypothesis Tests Complete ---")
