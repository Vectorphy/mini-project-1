import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import ttest_ind, ttest_1samp
import statsmodels.api as sm
from statsmodels.formula.api import ols

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

# Create a directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Function to perform EDA on a dataframe
def perform_eda(df, period_name):
    print(f"\n--- EDA for {period_name} ---")
    print(df[['Volume (in Mn)', 'Value (in Cr.)']].describe())

    # Histograms and Box plots
    for plot_type in ['hist', 'box']:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        if plot_type == 'hist':
            sns.histplot(df['Volume (in Mn)'], kde=True)
        else:
            sns.boxplot(y=df['Volume (in Mn)'])
        plt.title(f'Volume Distribution ({period_name})')

        plt.subplot(1, 2, 2)
        if plot_type == 'hist':
            sns.histplot(df['Value (in Cr.)'], kde=True)
        else:
            sns.boxplot(y=df['Value (in Cr.)'])
        plt.title(f'Value Distribution ({period_name})')

        plt.tight_layout()
        plt.savefig(f'visualizations/{period_name.lower()}_{plot_type}plots.png')
        plt.close()

perform_eda(pre_covid_df, 'Pre-COVID')
perform_eda(during_covid_df, 'During-COVID')
perform_eda(post_covid_df, 'Post-COVID')

print("\n--- EDA Complete. Visualizations saved. ---")


# --- 4. Time Series Plotting ---

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
plt.axvspan(during_covid_start, during_covid_end, color='red', alpha=0.3, label='During-COVID')
plt.legend()
fig.tight_layout()
plt.savefig('visualizations/time_series_full.png')
plt.close()

print("--- Time Series Plots Generated ---")


from statsmodels.tsa.api import ExponentialSmoothing

# --- 5. Holt-Winters Model Implementation ---

# Prepare data for Holt-Winters
train = pre_covid_df.set_index('Month')['Volume (in Mn)']
# Add a small constant to handle zero values, as multiplicative models can't handle them
train = train.replace(0, 0.01)

# Fit Holt-Winters model
# Using multiplicative trend and seasonality due to the exponential nature of the growth
hw_model = ExponentialSmoothing(
    train,
    trend='mul',
    seasonal='mul',
    seasonal_periods=12
).fit()

# Forecast for the entire period after pre-COVID
forecast_steps = len(df) - len(pre_covid_df)
hw_forecast = hw_model.forecast(steps=forecast_steps)

# Add forecast to the main dataframe
forecast_index = pd.date_range(start=pre_covid_df['Month'].iloc[-1] + pd.DateOffset(months=1), periods=forecast_steps, freq='MS')
df.loc[len(pre_covid_df):, 'hw_forecast'] = hw_forecast.values

print("--- Holt-Winters Model Implemented and Forecast Generated ---")


# --- 6. Hypothesis Testing (Revisited with HW Forecast) ---

print("\n--- Hypothesis Testing Results ---")

# Hypothesis 3 (Average Transactions) - Unchanged by forecasting model
print("\n--- H3: Difference in Average Transactions ---")
post_onset_df = pd.concat([during_covid_df, post_covid_df])
for col in ['Volume (in Mn)', 'Value (in Cr.)']:
    t_stat, p_val = ttest_ind(pre_covid_df[col], post_onset_df[col], equal_var=False)
    print(f"\n- {col}: T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  - Conclusion: The difference in average transactions is statistically significant.")
    else:
        print("  - Conclusion: The difference is not statistically significant.")

# Hypothesis 2 (Structural Change) - Unchanged by forecasting model
print("\n--- H2: Structural Break (Chow Test) ---")
df['time_index'] = range(len(df))
df['dummy'] = (df['Month'] > pre_covid_end).astype(int)
df['time_dummy'] = df['time_index'] * df['dummy']
model = ols('Q("Volume (in Mn)") ~ time_index + dummy + time_dummy', data=df).fit()
f_test_result = model.f_test('dummy = 0, time_dummy = 0')
print(f"\n- Chow-like test for Volume (in Mn): F-statistic: {f_test_result.fvalue:.4f}, P-value: {f_test_result.pvalue:.4f}")
if f_test_result.pvalue < 0.05:
    print("  - Conclusion: There is a statistically significant structural break.")
else:
    print("  - Conclusion: There is no statistically significant structural break.")

# Hypothesis 1 (Trend Change) - Revisited with Holt-Winters
print("\n--- H1: Change in Growth Trend (Holt-Winters) ---")
actuals = df.loc[len(pre_covid_df):]['Volume (in Mn)']
forecasts = df.loc[len(pre_covid_df):]['hw_forecast']
residuals_hw = actuals - forecasts
residuals_hw = residuals_hw.dropna()

t_stat_resid_hw, p_val_resid_hw = ttest_1samp(residuals_hw, 0)
print(f"\n- Testing for trend change in Volume (in Mn): T-statistic on residuals: {t_stat_resid_hw:.4f}, P-value: {p_val_resid_hw:.4f}")
if p_val_resid_hw < 0.05:
    print("  - Conclusion: The growth trend significantly accelerated post-pandemic.")
else:
    print("  - Conclusion: The growth trend did not significantly change post-pandemic.")

# --- 7. Generate New Plots based on Holt-Winters ---

# Create a directory for visualizations
if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

# Plot Holt-Winters Components
plt.figure(figsize=(14, 10))
plt.subplot(411)
plt.plot(hw_model.level, label='Level')
plt.legend()
plt.subplot(412)
plt.plot(hw_model.trend, label='Trend')
plt.legend()
plt.subplot(413)
plt.plot(hw_model.season, label='Seasonality')
plt.legend()
plt.subplot(414)
plt.plot(hw_model.resid, label='Residuals')
plt.legend()
plt.suptitle('Holt-Winters Components')
plt.tight_layout()
plt.savefig('visualizations/hw_components.png')
plt.close()

# Plot Forecast vs. Actual
plt.figure(figsize=(14, 7))
plt.plot(df['Month'], df['Volume (in Mn)'], label='Actual Volume')
plt.plot(df['Month'], df['hw_forecast'], 'r--', label='Holt-Winters Forecast')
plt.title('Actual UPI Volume vs. Holt-Winters Forecast')
plt.xlabel('Month')
plt.ylabel('Volume (in Mn)')
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/hw_forecast_vs_actual.png')
plt.close()

# Period-Specific Plots
def plot_hw_forecast_comparison(period_df, period_name):
    forecast_period = df[df['Month'].isin(period_df['Month'])]

    plt.figure(figsize=(10, 6))
    plt.plot(period_df['Month'], period_df['Volume (in Mn)'], label='Actual Volume')
    plt.plot(forecast_period['Month'], forecast_period['hw_forecast'], 'r--', label='Holt-Winters Forecast')
    plt.title(f'Actual vs. Holt-Winters Forecast ({period_name})')
    plt.xlabel('Month')
    plt.ylabel('Volume (in Mn)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualizations/hw_forecast_vs_actual_{period_name.lower().replace("-", "_")}.png')
    plt.close()

plot_hw_forecast_comparison(during_covid_df, 'During-COVID')
plot_hw_forecast_comparison(post_covid_df, 'Post-COVID')


print("--- New Plots Generated based on Holt-Winters ---")
print("\n--- Analysis Complete ---")
