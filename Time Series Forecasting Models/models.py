import pandas as pd
import numpy as np
from pmdarima import auto_arima
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler

# ---------------------- ARIMA ----------------------
def arima_forecast(df, steps=30):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna().asfreq('D')

    try:
        model = auto_arima(df['Close'], seasonal=False, stepwise=True, suppress_warnings=True)
        forecast = model.predict(n_periods=steps)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
        return pd.Series(forecast, index=future_dates)
    except Exception as e:
        print(f"❌ ARIMA error: {e}")
        return pd.Series(dtype='float64')


# ---------------------- SARIMA ----------------------
def sarima_forecast(df, steps=30):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna().asfreq('D')

    if len(df) < 50:
        print("❌ Not enough data for SARIMA")
        return pd.Series(dtype='float64')

    try:
        model = auto_arima(
            df['Close'],
            seasonal=True,
            m=7,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        forecast = model.predict(n_periods=steps)
        future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=steps)
        return pd.Series(forecast, index=future_dates)
    except Exception as e:
        print(f"❌ SARIMA error: {e}")
        return pd.Series(dtype='float64')


# ---------------------- Prophet ----------------------
def prophet_forecast(df, steps=30):
    df_prophet = df.reset_index()[['Date', 'Close']].copy()
    df_prophet.columns = ['ds', 'y']
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
    df_prophet.dropna(inplace=True)

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    return forecast.set_index('ds')['yhat'][-steps:]


# ---------------------- LSTM ----------------------
def lstm_forecast(df, steps=30, window_size=30, epochs=10):
    df = df.copy()
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()

    close_data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(close_data)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i])
        y.append(scaled_data[i])
    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, input_shape=(X.shape[1], 1)))
    model.add(LSTM(32))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, callbacks=[early_stop])

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
