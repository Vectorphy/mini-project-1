import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import ttest_1samp, ttest_ind, levene
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.formula.api import ols
import numpy as np

# --- 1. Data Cleaning and Preparation ---
df = pd.read_csv('Untitled spreadsheet - Sheet1.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%b-%y')
for col in ['Volume (in Mn)', 'Value (in Cr.)', 'No. of Banks live on UPI']:
    if df[col].dtype == 'object':
        df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    else:
        df[col] = pd.to_numeric(df[col])

df = df.sort_values('Month').reset_index(drop=True)

# Create new columns for analysis
df['Volume (in Bn)'] = df['Volume (in Mn)'] / 1000
df['Value (in Trn)'] = df['Value (in Cr.)'] / 100000
df['Year'] = df['Month'].dt.year
df['Month_Name'] = df['Month'].dt.strftime('%b')
df['Volume Growth Rate (%)'] = df['Volume (in Bn)'].pct_change() * 100
df['Value Growth Rate (%)'] = df['Value (in Trn)'].pct_change() * 100


pre_covid_end = pd.to_datetime('2020-03-31')
during_covid_start = pd.to_datetime('2020-04-01')
during_covid_end = pd.to_datetime('2022-01-31')
post_covid_start = pd.to_datetime('2022-08-01')

pre_covid_df = df[df['Month'] <= pre_covid_end].copy()
during_covid_df = df[(df['Month'] >= during_covid_start) & (df['Month'] <= during_covid_end)].copy()
post_covid_df = df[df['Month'] >= post_covid_start].copy()
print("--- Data Cleaning and Preparation Complete ---")

# --- 2. Export Cleaned Data ---
with pd.ExcelWriter('upi_data_cleaned.xlsx') as writer:
    pre_covid_df.to_excel(writer, sheet_name='Pre-COVID', index=False)
    during_covid_df.to_excel(writer, sheet_name='During-COVID', index=False)
    post_covid_df.to_excel(writer, sheet_name='Post-COVID', index=False)
print("--- Cleaned data exported to 'upi_data_cleaned.xlsx' ---")

# --- 3. Detailed EDA ---
print("\n--- Exploratory Data Analysis (Summary Statistics) ---")
for period_df, period_name in [(pre_covid_df, 'Pre-COVID'), (during_covid_df, 'During-COVID'), (post_covid_df, 'Post-COVID')]:
    print(f"\n--- Summary Statistics for {period_name} ---")
    print(period_df[['Volume (in Mn)', 'Value (in Cr.)']].describe())

if not os.path.exists('visualizations'):
    os.makedirs('visualizations')

def plot_eda_distributions(df, period_name):
    plt.figure(figsize=(12, 5))
    plt.suptitle(f'Data Distribution ({period_name})')
    plt.subplot(1, 2, 1)
    sns.histplot(df['Volume (in Mn)'], kde=True)
    plt.title('Volume Histogram')
    plt.subplot(1, 2, 2)
    sns.boxplot(y=df['Volume (in Mn)'])
    plt.title('Volume Box Plot')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'visualizations/eda_distributions_{period_name.lower()}.png')
    plt.close()

plot_eda_distributions(pre_covid_df, 'Pre-COVID')
plot_eda_distributions(during_covid_df, 'During-COVID')
plot_eda_distributions(post_covid_df, 'Post-COVID')
print("\n--- Detailed EDA Plots Generated ---")

# --- 3a. Expanded EDA ---
print("\n--- Expanded EDA ---")

# Yearly and Monthly Trend Analysis
# Reorder month names for correct plotting
month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
df['Month_Name'] = pd.Categorical(df['Month_Name'], categories=month_order, ordered=True)

plt.figure(figsize=(12, 8))
sns.heatmap(df.pivot_table(values='Volume (in Bn)', index='Year', columns='Month_Name', aggfunc='sum'),
            cmap='viridis', annot=True, fmt=".1f")
plt.title('Yearly and Monthly Transaction Volume (in Bn)')
plt.savefig('visualizations/volume_yearly_monthly_heatmap.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df.pivot_table(values='Value (in Trn)', index='Year', columns='Month_Name', aggfunc='sum'),
            cmap='viridis', annot=True, fmt=".2f")
plt.title('Yearly and Monthly Transaction Value (in Trn)')
plt.savefig('visualizations/value_yearly_monthly_heatmap.png')
plt.close()
print("- Yearly and Monthly Trend Analysis plots generated.")

# Outlier Detection
for col_name, file_name in [('Volume Growth Rate (%)', 'volume_growth_outliers.png'),
                            ('Value Growth Rate (%)', 'value_growth_outliers.png')]:
    Q1 = df[col_name].quantile(0.25)
    Q3 = df[col_name].quantile(0.75)
    IQR = Q3 - Q1
    outlier_condition = (df[col_name] < (Q1 - 1.5 * IQR)) | (df[col_name] > (Q3 + 1.5 * IQR))
    outliers = df[outlier_condition]

    plt.figure(figsize=(12, 6))
    plt.scatter(df['Month'], df[col_name], label='Data Points')
    plt.scatter(outliers['Month'], outliers[col_name], color='red', label='Outliers')
    plt.title(f'Outlier Detection in {col_name}')
    plt.xlabel('Date')
    plt.ylabel('Growth Rate (%)')
    plt.legend()
    plt.savefig(f'visualizations/{file_name}')
    plt.close()
print("- Outlier Detection plots generated.")

# Bank Integration Analysis
fig, ax1 = plt.subplots(figsize=(14, 7))
ax1.plot(df['Month'], df['Volume (in Bn)'], 'g-', label='Volume (in Bn)')
ax1.set_xlabel('Month')
ax1.set_ylabel('Volume (in Bn)', color='g')
ax1.tick_params('y', colors='g')
ax2 = ax1.twinx()
ax2.plot(df['Month'], df['No. of Banks live on UPI'], 'b-', label='No. of Banks live on UPI')
ax2.set_ylabel('No. of Banks live on UPI', color='b')
ax2.tick_params('y', colors='b')
plt.title('Bank Integration vs. Transaction Volume')
plt.legend(handles=[ax1.get_lines()[0], ax2.get_lines()[0]])
fig.tight_layout()
plt.savefig('visualizations/bank_integration_vs_volume.png')
plt.close()
print("- Bank Integration Analysis plot generated.")
print("--- Expanded EDA Complete ---")


# --- 3b. Report Visualizations ---
print("\n--- Generating Visualizations for Report ---")

# Box plots for Volume and Value by period
plt.figure(figsize=(10, 6))
sns.boxplot(data=[pre_covid_df['Volume (in Bn)'], during_covid_df['Volume (in Bn)'], post_covid_df['Volume (in Bn)']],
            showfliers=False)
plt.xticks([0, 1, 2], ['Pre-COVID', 'During-COVID', 'Post-COVID'])
plt.title('Volume Distribution by Period')
plt.ylabel('Volume (in Bn)')
plt.savefig('visualizations/volume_boxplot_by_period.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.boxplot(data=[pre_covid_df['Value (in Trn)'], during_covid_df['Value (in Trn)'], post_covid_df['Value (in Trn)']],
            showfliers=False)
plt.xticks([0, 1, 2], ['Pre-COVID', 'During-COVID', 'Post-COVID'])
plt.title('Value Distribution by Period')
plt.ylabel('Value (in Trn)')
plt.savefig('visualizations/value_boxplot_by_period.png')
plt.close()
print("- Box plots by period generated.")

# Value growth rate plot
plt.figure(figsize=(12, 6))
plt.plot(df['Month'], df['Value Growth Rate (%)'], marker='o', linestyle='-')
plt.title('Month-over-Month Value Growth Rate (%)')
plt.xlabel('Month')
plt.ylabel('Growth Rate (%)')
plt.grid(True)
plt.savefig('visualizations/value_growth_rate.png')
plt.close()
print("- Value growth rate plot generated.")

# Value with rolling average
plt.figure(figsize=(12, 6))
plt.plot(df['Month'], df['Value (in Trn)'], label='Value (in Trn)')
plt.plot(df['Month'], df['Value (in Trn)'].rolling(window=6).mean(), label='6-Month Rolling Average', color='red')
plt.title('Value (in Trn) with 6-Month Rolling Average')
plt.xlabel('Month')
plt.ylabel('Value (in Trn)')
plt.legend()
plt.savefig('visualizations/value_rolling_average.png')
plt.close()
print("- Value with rolling average plot generated.")

# Correlation heatmap
# Selecting only numeric columns for correlation
numeric_cols = df.select_dtypes(include=np.number)
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of UPI Data')
plt.savefig('visualizations/correlation_heatmap.png')
plt.close()
print("- Correlation heatmap generated.")


# Time series decomposition for Volume and Value
for col, file_suffix in [('Volume (in Bn)', 'volume'), ('Value (in Trn)', 'value')]:
    decomposition = seasonal_decompose(df.set_index('Month')[col], model='additive', period=12)
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)
    plt.suptitle(f'Time Series Decomposition of {col}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'visualizations/{file_suffix}_decomposition.png')
    plt.close(fig)
print("- Time series decomposition plots generated.")
print("--- Report Visualizations Complete ---")


# --- 4. Time Series Plotting ---
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

# --- 5. Final Hybrid Forecasting Analysis ---
def run_holtwinters_analysis(train_df, test_df, period_name):
    print(f"\n--- Running Holt-Winters Analysis for {period_name} ---")
    train_vol = train_df.set_index('Month')['Volume (in Mn)'].replace(0, 0.01)
    test_vol = test_df.set_index('Month')['Volume (in Mn)']
    model = ExponentialSmoothing(train_vol, trend='mul', seasonal='mul', seasonal_periods=12).fit()
    forecast = model.forecast(steps=len(test_vol))
    forecast.index = test_vol.index
    plt.figure(figsize=(10, 6))
    plt.plot(train_vol.index, train_vol, label='Training Data')
    plt.plot(test_vol.index, test_vol, label='Actual Volume')
    plt.plot(forecast.index, forecast, 'r--', label='Holt-Winters Forecast')
    plt.title(f'Holt-Winters Forecast vs. Actual ({period_name})')
    plt.xlabel('Month')
    plt.ylabel('Volume (in Mn)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualizations/hw_forecast_{period_name.lower().replace(" ", "_")}.png')
    plt.close()
    residuals = test_vol - forecast
    t_stat, p_val = ttest_1samp(residuals, 0)
    print(f"\n- Test on Residuals: T-statistic={t_stat:.4f}, P-value={p_val:.4f}")

def run_arima_analysis(train_df, test_df, period_name, arima_order):
    print(f"\n--- Running ARIMA Analysis for {period_name} ---")
    train_vol = train_df.set_index('Month')['Volume (in Mn)']
    test_vol = test_df.set_index('Month')['Volume (in Mn)']
    model = ARIMA(train_vol, order=arima_order)
    fitted_model = model.fit()
    forecast = fitted_model.predict(start=test_vol.index[0], end=test_vol.index[-1])
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
    residuals = test_vol - forecast
    t_stat, p_val = ttest_1samp(residuals, 0)
    print(f"\n- Test on Residuals: T-statistic={t_stat:.4f}, P-value={p_val:.4f}")

# Part 1
run_holtwinters_analysis(pre_covid_df, during_covid_df, "Pre-COVID vs During-COVID")

# Part 2
pre_and_during_df = pd.concat([pre_covid_df, during_covid_df])
full_range = pd.date_range(start=pre_and_during_df['Month'].min(), end=pre_and_during_df['Month'].max(), freq='MS')
pre_and_during_df = pre_and_during_df.set_index('Month').reindex(full_range).reset_index().rename(columns={'index': 'Month'})
pre_and_during_df['Volume (in Mn)'] = pre_and_during_df['Volume (in Mn)'].interpolate(method='linear')
run_arima_analysis(pre_and_during_df, post_covid_df, "(Pre+During)-COVID vs Post-COVID", arima_order=(2, 2, 2))
print("\n--- Hybrid Forecasting Analysis Complete ---")

# --- 6. Additional Hypothesis Tests ---
print("\n--- Additional Hypothesis Tests ---")
post_onset_df = pd.concat([during_covid_df, post_covid_df])
for col in ['Volume (in Mn)', 'Value (in Cr.)']:
    t_stat, p_val = ttest_ind(pre_covid_df[col], post_onset_df[col], equal_var=False)
    print(f"\n- T-test for Diff in Avg {col}: T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}")

df['time_index'] = range(len(df))
df['dummy'] = (df['Month'] > pre_covid_end).astype(int)
df['time_dummy'] = df['time_index'] * df['dummy']
model = ols('Q("Volume (in Mn)") ~ time_index + dummy + time_dummy', data=df).fit()
f_test_result = model.f_test('dummy = 0, time_dummy = 0')
print(f"\n- Chow Test for Structural Break: F-statistic: {f_test_result.fvalue:.4f}, P-value: {f_test_result.pvalue:.4f}")

pre_covid_growth = pre_covid_df['Volume (in Mn)'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()
post_onset_growth = post_onset_df['Volume (in Mn)'].pct_change().dropna()
levene_stat, levene_p_val = levene(pre_covid_growth, post_onset_growth)
print(f"\n- Levene Test for Volatility Change: Levene statistic: {levene_stat:.4f}, P-value: {levene_p_val:.4f}")
print("\n--- Additional Hypothesis Tests Complete ---")
