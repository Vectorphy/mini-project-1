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


# --- 5. Hypothesis Testing ---

print("\n--- Hypothesis Testing Results ---")

# Hypothesis 3: Average Transactions
print("\n--- H3: Difference in Average Transactions ---")
post_onset_df = pd.concat([during_covid_df, post_covid_df])
for col in ['Volume (in Mn)', 'Value (in Cr.)']:
    t_stat, p_val = ttest_ind(pre_covid_df[col], post_onset_df[col], equal_var=False)
    print(f"\n- {col}:")
    print(f"  - T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")
    if p_val < 0.05:
        print("  - Conclusion: The difference in average transactions is statistically significant.")
    else:
        print("  - Conclusion: The difference is not statistically significant.")

# Hypothesis 2: Structural Change (Chow Test)
print("\n--- H2: Structural Break (Chow Test) ---")
df['time_index'] = range(len(df))
df['dummy'] = (df['Month'] > pre_covid_end).astype(int)
df['time_dummy'] = df['time_index'] * df['dummy']
model = ols('Q("Volume (in Mn)") ~ time_index + dummy + time_dummy', data=df).fit()
f_test_result = model.f_test('dummy = 0, time_dummy = 0')
print("\n- Chow-like test for Volume (in Mn):")
print(f"  - F-statistic: {f_test_result.fvalue:.4f}, P-value: {f_test_result.pvalue:.4f}")
if f_test_result.pvalue < 0.05:
    print("  - Conclusion: There is a statistically significant structural break.")
else:
    print("  - Conclusion: There is no statistically significant structural break.")

# Hypothesis 1: Trend Change
print("\n--- H1: Change in Growth Trend ---")
X_pre = sm.add_constant(range(len(pre_covid_df)))
model_pre = sm.OLS(pre_covid_df['Volume (in Mn)'], X_pre).fit()
X_post_onset = sm.add_constant(range(len(pre_covid_df), len(df)))
predicted_post_onset = pd.Series(model_pre.predict(X_post_onset), index=range(len(pre_covid_df), len(df)))
actual_post_onset = post_onset_df.set_index(pd.Index(range(len(pre_covid_df), len(pre_covid_df) + len(post_onset_df))))['Volume (in Mn)']
actual_aligned = actual_post_onset.reindex(predicted_post_onset.index)
residuals = (actual_aligned - predicted_post_onset).dropna()
t_stat_resid, p_val_resid = ttest_1samp(residuals, 0)
print("\n- Testing for trend change in Volume (in Mn):")
print(f"  - T-statistic on residuals: {t_stat_resid:.4f}, P-value: {p_val_resid:.4f}")
if p_val_resid < 0.05:
    print("  - Conclusion: The growth trend significantly accelerated post-pandemic.")
else:
    print("  - Conclusion: The growth trend did not significantly change post-pandemic.")

# --- 6. Forecast vs. Actual Trend Comparison Plot ---

# Generate predictions for the entire timeline based on the pre-COVID model
df['forecast'] = model_pre.predict(sm.add_constant(df['time_index']))

plt.figure(figsize=(14, 7))
plt.plot(df['Month'], df['Volume (in Mn)'], label='Actual Volume')
plt.plot(df['Month'], df['forecast'], 'r--', label='Forecasted Trend (from Pre-COVID)')
plt.axvspan(during_covid_start, during_covid_end, color='red', alpha=0.15, label='During-COVID')
plt.title('Actual UPI Volume vs. Forecasted Trend')
plt.xlabel('Month')
plt.ylabel('Volume (in Mn)')
plt.legend()
plt.tight_layout()
plt.savefig('visualizations/forecast_vs_actual.png')
plt.close()

print("--- Forecast vs. Actual plot generated ---")
print("\n--- Analysis Complete ---")
