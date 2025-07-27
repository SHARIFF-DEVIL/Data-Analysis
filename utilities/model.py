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
def sarima_forecast(df, order=(1, 1, 0), seasonal_order=(1, 0, 1, 7), steps=14):
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import warnings
    warnings.filterwarnings("ignore")

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')
        df = df.dropna(subset=['Close'])

    df = df.asfreq('D')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.sort_index().dropna()

    if len(df) < 50:
        print("❌ Not enough data for SARIMA (minimum 50 observations required).")
        return pd.Series(dtype='float64')

    try:
        model = SARIMAX(
            df['Close'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)

        if not model_fit.mle_retvals['converged']:
            print("⚠️ Model did not converge. Try changing parameters.")

        forecast = model_fit.get_forecast(steps=steps)
        forecast_mean = forecast.predicted_mean
        forecast_index = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=steps)

        return pd.Series(forecast_mean.values, index=forecast_index, name='Forecast')

    except Exception as e:
        print(f"❌ SARIMA failed: {e}")
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