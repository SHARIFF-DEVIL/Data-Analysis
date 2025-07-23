import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ---------------------- ARIMA (Manual Order) ----------------------
def arima_forecast(df, order=(5, 1, 1), steps=7):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()

    model = ARIMA(df['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(forecast.values, index=future_dates)

# ---------------------- SARIMA (Manual Seasonal Order) ----------------------
def sarima_forecast(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), steps=7):
    df = df.copy()

    # Ensure DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(subset=['Close'])

    # Set frequency explicitly
    df = df.asfreq('D', fill_value=df['Close'].mean())
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.sort_index()

    # Check for sufficient data
    if len(df) < 50:
        print("❌ Not enough data for SARIMA (minimum 50 observations required).")
        return pd.Series(dtype='float64')

    try:
        # Fit SARIMA model
        model = SARIMAX(
            df['Close'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False, maxiter=200)

        # Check model convergence
        if not model_fit.mle_retvals['converged']:
            print("⚠️ SARIMA model did not converge properly. Consider adjusting parameters.")
        
        # Generate forecast
        forecast_result = model_fit.get_forecast(steps=steps)
        forecast_values = forecast_result.predicted_mean
        forecast_ci = forecast_result.conf_int()

        # Create future dates
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
        
        # Combine forecast and confidence intervals into a DataFrame
        forecast_df = pd.DataFrame({
            'Forecast': forecast_values,
            'Lower CI': forecast_ci.iloc[:, 0],
            'Upper CI': forecast_ci.iloc[:, 1]
        }, index=future_dates)

        # Print model diagnostics
        print(f"✅ SARIMA model summary:\n{model_fit.summary().tables[0]}")
        return forecast_df['Forecast']

    except Exception as e:
        print(f"❌ SARIMA model failed: {e}")
        return pd.Series(dtype='float64')

# ---------------------- Prophet ----------------------
def prophet_forecast(df, steps=7):
    df = df.copy()
    df = df.reset_index()
    df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    df['y'] = pd.to_numeric(df['y'], errors='coerce')
    df = df.dropna()

    model = Prophet(daily_seasonality=True)
    model.fit(df)

    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    forecast_series = forecast.set_index('ds')['yhat'].iloc[-steps:]
    return forecast_series

# ---------------------- LSTM ----------------------
def lstm_forecast(df, steps=7, window_size=30, epochs=10):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()

    if len(df) < window_size:
        print("❌ Not enough data for LSTM.")
        return pd.Series(dtype='float64')

    close_data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    future_input = scaled_data[-window_size:]
    future_preds = []
    for _ in range(steps):
        pred_input = future_input[-window_size:].reshape(1, window_size, 1)
        pred = model.predict(pred_input, verbose=0)
        future_preds.append(pred[0][0])
        future_input = np.append(future_input, pred, axis=0)

    forecast = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1)).flatten()
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(forecast, index=future_dates)