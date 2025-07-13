import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(df, order=(5, 1, 0), steps=30):
    model = ARIMA(df['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(forecast.values, index=future_dates)

def random_forest_forecast(df, future_days=30):
    df = df.copy()
    required_cols = ['Open', 'High', 'Low', 'Volume', 'Close']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing columns. Required: {required_cols}")

    df['Target'] = df['Close'].shift(-future_days)
    df.dropna(inplace=True)

    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Target']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_known = df[['Open', 'High', 'Low', 'Volume']].tail(future_days).copy()
    if len(last_known) < future_days:
        last_row = last_known.iloc[-1]
        for _ in range(future_days - len(last_known)):
            last_known = pd.concat([last_known, last_row.to_frame().T], ignore_index=True)

    future_pred = model.predict(last_known)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=future_days)
    return pd.Series(future_pred, index=future_dates)