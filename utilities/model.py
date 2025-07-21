import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ---------------------- ARIMA (Manual Order) ----------------------
def arima_forecast(df, order=(5, 1, 1), steps=30):
    model = ARIMA(df['Close'], order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
    return pd.Series(forecast.values, index=future_dates)

# ---------------------- SARIMA (Manual Seasonal Order) ----------------------
def sarima_forecast(df, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7), steps=30):
    df = df.copy()

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors='coerce')

    # Drop rows with invalid dates or close values
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna(subset=['Close'])
    df = df.sort_index()

    if len(df) < 50:
        raise ValueError("Not enough data points for SARIMA. Minimum ~50 required.")

    try:
        model = SARIMAX(
            df['Close'],
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        model_fit = model.fit(disp=False)
        forecast = model_fit.forecast(steps=steps)
    except Exception as e:
        print(f"❌ SARIMA model failed: {e}")
        return pd.Series(dtype='float64')  # Return empty forecast on failure

    # Generate future dates
    future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    forecast_series = pd.Series(forecast, index=future_dates).dropna()

    return forecast_series

# ---------------------- Prophet ----------------------
def prophet_forecast(df, steps=30):
    df_prophet = df.reset_index()[['Date', 'Close']]
    df_prophet.columns = ['ds', 'y']
    
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    
    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)
    
    future_forecast = forecast[['ds', 'yhat']].set_index('ds').iloc[-steps:]
    return future_forecast['yhat']

# ---------------------- LSTM ----------------------
def lstm_forecast(df, steps=30, window_size=30):
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
    model.fit(X, y, epochs=20, batch_size=32, verbose=0)

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