import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("Data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])
df = df.sort_values('Date')
df.set_index('Date', inplace=True)

target_columns = df.columns.tolist()
forecast_horizon = 7
arima_forecast_df = pd.DataFrame()
arima_forecast_df['Date'] = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=forecast_horizon)

# Loop and forecast
for col in target_columns:
    try:
        series = pd.to_numeric(df[col], errors='coerce').dropna()

        if len(series) < 10:
            print(f"⚠️ Skipped (too little data): {col}")
            arima_forecast_df[f'{col}_arima'] = [None] * forecast_horizon
            continue

        model = ARIMA(series, order=(5, 1, 0))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=forecast_horizon)

        arima_forecast_df[f'{col}_arima'] = forecast.values

    except Exception as e:
        print(f"❌ ARIMA error for {col}: {e}")
        arima_forecast_df[f'{col}_arima'] = [None] * forecast_horizon

arima_forecast_df.to_csv("arima_forecast.csv", index=False)
print("✅ Saved ARIMA forecast to: arima_forecast.csv")
