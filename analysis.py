import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.stats import ttest_1samp, ttest_ind, levene
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.formula.api import ols
import numpy as np
import warnings

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
sns.heatmap(df.pivot_table(values='Volume (in Bn)', index='Year', columns='Month_Name', aggfunc='sum', observed=True),
            cmap='viridis', annot=True, fmt=".1f")
plt.title('Yearly and Monthly Transaction Volume (in Bn)')
plt.savefig('visualizations/volume_yearly_monthly_heatmap.png')
plt.close()

plt.figure(figsize=(12, 8))
sns.heatmap(df.pivot_table(values='Value (in Trn)', index='Year', columns='Month_Name', aggfunc='sum', observed=True),
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
# Robust box plots by period using a combined DataFrame (ensures proper plotting even
# if some periods have NaNs or different indexing)
combined_periods = []
for df_period, name in [(pre_covid_df, 'Pre-COVID'), (during_covid_df, 'During-COVID'), (post_covid_df, 'Post-COVID')]:
    tmp = df_period.copy()
    if 'Volume (in Bn)' in tmp.columns:
        tmp = tmp[['Month', 'Volume (in Bn)', 'Value (in Trn)']].copy()
    else:
        # fallback in case column missing
        tmp['Volume (in Bn)'] = tmp.get('Volume (in Mn)', np.nan) / 1000
        tmp['Value (in Trn)'] = tmp.get('Value (in Cr.)', np.nan) / 100000
    tmp['Period'] = name
    combined_periods.append(tmp)

combined_df = pd.concat(combined_periods, ignore_index=True)

# Plot with error handling to avoid blocking the script if plotting fails
try:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Period', y='Volume (in Bn)', data=combined_df, order=['Pre-COVID', 'During-COVID', 'Post-COVID'], showfliers=False)
    plt.title('Volume Distribution by Period')
    plt.ylabel('Volume (in Bn)')
    plt.tight_layout()
    plt.savefig('visualizations/volume_boxplot_by_period.png')
finally:
    plt.close()

try:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Period', y='Value (in Trn)', data=combined_df, order=['Pre-COVID', 'During-COVID', 'Post-COVID'], showfliers=False)
    plt.title('Value Distribution by Period')
    plt.ylabel('Value (in Trn)')
    plt.tight_layout()
    plt.savefig('visualizations/value_boxplot_by_period.png')
finally:
    plt.close()
print("- Box plots by period generated.")

# --- Normalized comparison (z-score per period) ---
try:
    z_df = combined_df.copy()
    # compute z-score within each period for Volume and Value
    z_df['Volume_z'] = z_df.groupby('Period')['Volume (in Bn)'].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    z_df['Value_z'] = z_df.groupby('Period')['Value (in Trn)'].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    # Boxplots of z-scores
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Period', y='Volume_z', data=z_df, order=['Pre-COVID', 'During-COVID', 'Post-COVID'], showfliers=False)
    plt.title('Volume (z-score) Distribution by Period')
    plt.ylabel('Volume z-score')
    plt.tight_layout()
    plt.savefig('visualizations/volume_boxplot_zscore_by_period.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Period', y='Value_z', data=z_df, order=['Pre-COVID', 'During-COVID', 'Post-COVID'], showfliers=False)
    plt.title('Value (z-score) Distribution by Period')
    plt.ylabel('Value z-score')
    plt.tight_layout()
    plt.savefig('visualizations/value_boxplot_zscore_by_period.png')
    plt.close()

    # KDE overlays for quick visual comparison (each period's z-scores)
    plt.figure(figsize=(10, 6))
    for period in ['Pre-COVID', 'During-COVID', 'Post-COVID']:
        subset = z_df[z_df['Period'] == period]['Volume_z'].dropna()
        if not subset.empty:
            sns.kdeplot(subset, label=period)
    plt.title('Volume z-score KDE by Period')
    plt.xlabel('Volume z-score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/volume_kde_zscore_by_period.png')
    plt.close()

    plt.figure(figsize=(10, 6))
    for period in ['Pre-COVID', 'During-COVID', 'Post-COVID']:
        subset = z_df[z_df['Period'] == period]['Value_z'].dropna()
        if not subset.empty:
            sns.kdeplot(subset, label=period)
    plt.title('Value z-score KDE by Period')
    plt.xlabel('Value z-score')
    plt.legend()
    plt.tight_layout()
    plt.savefig('visualizations/value_kde_zscore_by_period.png')
    plt.close()
    print("- Normalized (z-score) boxplots and KDEs generated.")
except Exception as e:
    print(f"- Warning: normalized period plots failed: {e}")
    try:
        plt.close()
    except Exception:
        pass

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

# --- 5. Final SARIMA Forecasting Analysis ---
def run_sarima_analysis(train_df, test_df, period_name, order, seasonal_order):
    print(f"\n--- Running SARIMA Analysis for {period_name} ---")
    # Ensure the series have a monthly start frequency (MS) so forecasts align
    train_vol = train_df.set_index('Month')['Volume (in Mn)'].sort_index()
    test_vol = test_df.set_index('Month')['Volume (in Mn)'].sort_index()

    # Force a monthly start frequency. Do NOT fill missing test values here — we only want
    # the forecast horizon to match the number of test periods.
    try:
        train_vol = train_vol.asfreq('MS')
    except Exception:
        # If the index has duplicate or invalid entries, build an explicit monthly index
        train_vol = train_vol.reindex(pd.date_range(start=train_vol.index.min(), end=train_vol.index.max(), freq='MS'))

    # Determine forecast horizon so forecasts start immediately after the last training month
    # and extend up to the end of the test period (no gaps between training and forecast).
    # Forecast start is the month after the last training observation.
    forecast_start = train_vol.index.max() + pd.DateOffset(months=1)
    forecast_end = test_vol.index.max()
    if forecast_end < forecast_start:
        # Test period is entirely at or before the end of training; nothing to forecast
        print("\n- Test period ends on or before the training end; no forecast horizon to compute.")
        return

    forecast_index = pd.date_range(start=forecast_start, end=forecast_end, freq='MS')
    steps = len(forecast_index)

    model = SARIMAX(train_vol, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    fitted_model = model.fit(disp=False)

    # Use get_forecast to produce 'steps' ahead forecasts starting immediately after training end
    forecast_res = fitted_model.get_forecast(steps=steps)
    forecast_mean = pd.Series(forecast_res.predicted_mean, index=forecast_index)
    forecast_ci = forecast_res.conf_int()
    # conf_int comes back with integer index; set it to the forecast_index as well
    forecast_ci.index = forecast_index

    plt.figure(figsize=(10, 6))
    plt.plot(train_vol.index, train_vol, label='Training Data')
    plt.plot(test_vol.index, test_vol, label='Actual Volume')
    plt.plot(forecast_mean.index, forecast_mean, 'r--', label='SARIMA Forecast')
    plt.fill_between(forecast_ci.index,
                     forecast_ci.iloc[:, 0],
                     forecast_ci.iloc[:, 1], color='pink', alpha=0.5)
    plt.title(f'SARIMA Forecast vs. Actual ({period_name})')
    plt.xlabel('Month')
    plt.ylabel('Volume (in Mn)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'visualizations/sarima_forecast_{period_name.lower().replace(" ", "_")}.png')
    plt.close()

    # Align forecast and test series for residual calculation (keep only months present in test)
    forecast_mean_aligned = forecast_mean.reindex(test_vol.index)
    forecast_ci_aligned = forecast_ci.reindex(test_vol.index)

    residuals = test_vol - forecast_mean_aligned

    # Check for NaN in residuals before t-test
    if not residuals.dropna().empty:
        t_stat, p_val = ttest_1samp(residuals.dropna(), 0)
        print(f"\n- Test on Residuals: T-statistic={t_stat:.4f}, P-value={p_val:.4f}")
    else:
        print("\n- Residuals calculation resulted in NaN (no overlapping months between forecast and test). Check data alignment and date ranges.")


def find_best_sarima(train_df, p_vals=[0,1,2], d_vals=[0,1], q_vals=[0,1],
                     P_vals=[0,1], D_vals=[0,1], Q_vals=[0,1], s=12, maxiter=50):
    """
    Small grid search over SARIMA hyperparameters using AIC as selection metric.
    Keeps the search short by default; returns (order, seasonal_order).
    """
    print("\n--- Running small SARIMA grid search (AIC) ---")
    # Prepare series: ensure monthly frequency
    series = train_df.set_index('Month')['Volume (in Mn)'].sort_index()
    try:
        series = series.asfreq('MS')
    except Exception:
        series = series.reindex(pd.date_range(start=series.index.min(), end=series.index.max(), freq='MS'))

    best_aic = np.inf
    best_cfg = None
    tried = 0
    warnings.filterwarnings('ignore')
    for p in p_vals:
        for d in d_vals:
            for q in q_vals:
                for P in P_vals:
                    for D in D_vals:
                        for Q in Q_vals:
                            order = (p, d, q)
                            seasonal_order = (P, D, Q, s)
                            try:
                                model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                                                enforce_stationarity=False, enforce_invertibility=False)
                                res = model.fit(disp=False, maxiter=maxiter)
                                aic = res.aic
                                tried += 1
                                print(f"Tried order={order} seasonal={seasonal_order} -> AIC={aic:.1f}")
                                if aic < best_aic:
                                    best_aic = aic
                                    best_cfg = (order, seasonal_order)
                            except Exception as e:
                                # skip bad configurations quickly
                                print(f"Skipped order={order} seasonal={seasonal_order} due to: {e}")
                                continue
    warnings.filterwarnings('default')
    if best_cfg is not None:
        print(f"--- Grid search complete: tried {tried} fits, best AIC={best_aic:.2f} with order={best_cfg[0]} seasonal={best_cfg[1]} ---")
        return best_cfg
    else:
        print("--- Grid search failed to find a valid model; falling back to (1,1,1)x(1,1,1,12) ---")
        return (1,1,1), (1,1,1,12)

# Part 1
run_sarima_analysis(pre_covid_df, during_covid_df, "Pre-COVID vs During-COVID", order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))

# Part 2
pre_and_during_df = pd.concat([pre_covid_df, during_covid_df])
# Ensure we have a continuous monthly index for the pre+during training set and interpolate missing values conservatively
full_range = pd.date_range(start=pre_and_during_df['Month'].min(), end=pre_and_during_df['Month'].max(), freq='MS')
pre_and_during_df = pre_and_during_df.set_index('Month').reindex(full_range).reset_index().rename(columns={'index': 'Month'})
pre_and_during_df['Volume (in Mn)'] = pre_and_during_df['Volume (in Mn)'].interpolate(method='linear')

# For Part 2: explicitly use training up to the end of the during-COVID period and forecast from the start of post-COVID
train_df_part2 = pre_and_during_df[pre_and_during_df['Month'] <= during_covid_end].copy()
test_df_part2 = post_covid_df.copy()

# Run a larger grid-search on the training portion to pick (order, seasonal_order)
# This grid is bigger than the development run and will take longer to complete.
print("\n--- Starting extended SARIMA grid-search for Part 2 (this may take a while) ---")
best_order, best_seasonal = find_best_sarima(
    train_df_part2,
    p_vals=[0, 1, 2, 3],
    d_vals=[0, 1],
    q_vals=[0, 1, 2, 3],
    P_vals=[0, 1, 2],
    D_vals=[0, 1],
    Q_vals=[0, 1, 2],
    s=12,
    maxiter=100,
)
print(f"Using best order={best_order} and seasonal_order={best_seasonal} for Part 2")

# Fit the selected model on the training series and save the fitted model to disk for reuse
try:
    train_series_part2 = train_df_part2.set_index('Month')['Volume (in Mn)'].sort_index().asfreq('MS')
except Exception:
    train_series_part2 = train_df_part2.set_index('Month')['Volume (in Mn)'].sort_index()
    train_series_part2 = train_series_part2.reindex(pd.date_range(start=train_series_part2.index.min(), end=train_series_part2.index.max(), freq='MS'))

full_model = SARIMAX(train_series_part2, order=best_order, seasonal_order=best_seasonal, enforce_stationarity=False, enforce_invertibility=False)
fitted_full_model = full_model.fit(disp=False)

import pickle
model_path = os.path.join('visualizations', 'best_sarima_part2.pkl')
with open(model_path, 'wb') as fh:
    pickle.dump(fitted_full_model, fh)
print(f"Best SARIMA model for Part 2 fitted and saved to: {model_path}")

run_sarima_analysis(train_df_part2, test_df_part2, "(Pre+During)-COVID vs Post-COVID", order=best_order, seasonal_order=best_seasonal)
print("\n--- SARIMA Forecasting Analysis Complete ---")

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
