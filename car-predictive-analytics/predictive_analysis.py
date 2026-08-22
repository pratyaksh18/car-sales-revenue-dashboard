"""
Predictive Analytics Using Historical Data — Car Sales Dataset
================================================================
Goal: Build a predictive model to forecast future trends using
regression / time-series models, clean and preprocess historical
data, evaluate model accuracy, and visualize predictions.

Two complementary models are built:
  A) TRANSACTION-LEVEL REGRESSION — predicts Revenue for a sale from
     its features (Units Sold, Unit Price, Discount, Brand, Fuel Type,
     Region, Dealer). Demonstrates regression modeling + evaluation.
  B) MONTHLY REVENUE TIME-SERIES FORECAST — aggregates historical
     revenue by month and forecasts the next 3 months using linear
     trend regression. Demonstrates time-series trend forecasting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 130

# ------------------------------------------------------------------
# 1. LOAD & CLEAN DATA
# ------------------------------------------------------------------
df = pd.read_csv("Car_Sales_Dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])

print("Missing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Basic sanity cleaning: drop exact duplicates, ensure no negative values
df = df.drop_duplicates()
df = df[(df['Units Sold'] > 0) & (df['Unit Price (INR)'] > 0) & (df['Revenue (INR)'] >= 0)]
df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()

print(f"\nClean dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")

# ==================================================================
# PART A — TRANSACTION-LEVEL REVENUE REGRESSION
# ==================================================================
print("\n" + "="*60)
print("PART A: Transaction-Level Revenue Regression")
print("="*60)

feature_cols = ['Units Sold', 'Unit Price (INR)', 'Discount (%)',
                 'Brand', 'Fuel Type', 'Region', 'Dealer']
target_col = 'Revenue (INR)'

X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_features = ['Units Sold', 'Unit Price (INR)', 'Discount (%)']
categorical_features = ['Brand', 'Fuel Type', 'Region', 'Dealer']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=200, random_state=42, max_depth=6)
}

results = {}
predictions = {}

for name, model in models.items():
    pipe = Pipeline([('prep', preprocessor), ('model', model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    predictions[name] = preds

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mape = np.mean(np.abs((y_test - preds) / y_test)) * 100

    results[name] = {'R2': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE': mape}
    print(f"\n{name}:")
    print(f"  R^2   = {r2:.4f}")
    print(f"  MAE   = {mae:,.0f} INR")
    print(f"  RMSE  = {rmse:,.0f} INR")
    print(f"  MAPE  = {mape:.2f}%")

results_df = pd.DataFrame(results).T
results_df.to_csv("model_evaluation_metrics.csv")

best_model_name = results_df['R2'].idxmax()
print(f"\nBest model by R^2: {best_model_name}")

# Refit best model on full data for feature importance / final use
best_pipe = Pipeline([('prep', preprocessor),
                       ('model', models[best_model_name])])
best_pipe.fit(X_train, y_train)
best_preds = predictions[best_model_name]

# --- Visualization: Actual vs Predicted ---
plt.figure(figsize=(6.5, 6))
plt.scatter(y_test, best_preds, alpha=0.6, color="#4C72B0", edgecolor='black', s=60)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, '--', color='red', label='Perfect Prediction')
plt.xlabel("Actual Revenue (INR)")
plt.ylabel("Predicted Revenue (INR)")
plt.title(f"Actual vs Predicted Revenue ({best_model_name})")
plt.legend()
plt.tight_layout()
plt.savefig("chart_actual_vs_predicted.png", bbox_inches='tight')
plt.close()

# --- Visualization: Residuals ---
residuals = y_test.values - best_preds
plt.figure(figsize=(6.5, 4.5))
sns.histplot(residuals, kde=True, color="#DD8452")
plt.axvline(0, color='black', linestyle='--')
plt.title(f"Residual Distribution ({best_model_name})")
plt.xlabel("Residual (Actual - Predicted) INR")
plt.tight_layout()
plt.savefig("chart_residuals.png", bbox_inches='tight')
plt.close()

# --- Model comparison bar chart ---
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
metrics_to_plot = ['R2', 'MAE', 'RMSE']
colors = ['#4C72B0', '#DD8452']
for i, m in enumerate(metrics_to_plot):
    axes[i].bar(results_df.index, results_df[m], color=colors)
    axes[i].set_title(m)
    axes[i].tick_params(axis='x', rotation=15)
plt.suptitle("Model Comparison")
plt.tight_layout()
plt.savefig("chart_model_comparison.png", bbox_inches='tight')
plt.close()

# --- Feature importance (if Random Forest is available) ---
if 'Random Forest' in models:
    rf_pipe = Pipeline([('prep', preprocessor), ('model', models['Random Forest'])])
    rf_pipe.fit(X_train, y_train)
    ohe = rf_pipe.named_steps['prep'].named_transformers_['cat']
    feature_names = numeric_features + list(ohe.get_feature_names_out(categorical_features))
    importances = rf_pipe.named_steps['model'].feature_importances_
    imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    imp_df = imp_df.sort_values('importance', ascending=False).head(12)

    plt.figure(figsize=(7, 5))
    sns.barplot(data=imp_df, x='importance', y='feature', color="#55A868")
    plt.title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig("chart_feature_importance.png", bbox_inches='tight')
    plt.close()

# ==================================================================
# PART B — MONTHLY REVENUE TIME-SERIES FORECAST
# ==================================================================
print("\n" + "="*60)
print("PART B: Monthly Revenue Time-Series Forecast")
print("="*60)

monthly = df.groupby('Month')['Revenue (INR)'].sum().reset_index()
monthly = monthly.sort_values('Month').reset_index(drop=True)
monthly['t'] = np.arange(len(monthly))

print(monthly)

# Simple linear trend regression on time index
lr_ts = LinearRegression()
lr_ts.fit(monthly[['t']], monthly['Revenue (INR)'])
monthly['Fitted'] = lr_ts.predict(monthly[['t']])

train_r2 = r2_score(monthly['Revenue (INR)'], monthly['Fitted'])
print(f"\nLinear trend fit R^2 (in-sample): {train_r2:.4f}")

# Forecast next 3 months
future_t = np.arange(len(monthly), len(monthly) + 3)
future_months = pd.date_range(monthly['Month'].max() + pd.offsets.MonthBegin(1), periods=3, freq='MS')
future_preds = lr_ts.predict(future_t.reshape(-1, 1))

forecast_df = pd.DataFrame({'Month': future_months, 'Forecast_Revenue': future_preds})
forecast_df.to_csv("monthly_revenue_forecast.csv", index=False)
print("\nForecast for next 3 months:")
print(forecast_df.to_string(index=False))

# --- Visualization: historical + forecast ---
plt.figure(figsize=(9, 5))
plt.plot(monthly['Month'], monthly['Revenue (INR)'], marker='o', label='Actual Revenue', color='#4C72B0')
plt.plot(monthly['Month'], monthly['Fitted'], linestyle='--', label='Trend Fit', color='#55A868')
plt.plot(forecast_df['Month'], forecast_df['Forecast_Revenue'], marker='o', linestyle='--',
         label='Forecast (next 3 months)', color='#C44E52')
plt.title("Monthly Revenue: Historical Trend & 3-Month Forecast")
plt.ylabel("Revenue (INR)")
plt.xlabel("Month")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("chart_revenue_forecast.png", bbox_inches='tight')
plt.close()

print("\nAll models trained, evaluated, and visualizations saved successfully.")
