import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

df = pd.read_csv("Data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.sort_values('Date')
df = df.dropna(subset=['Date'])
df.set_index('Date', inplace=True)

target_columns = df.columns.tolist()
forecast_horizon = 7
look_back = 150  

results = pd.DataFrame()
results['Date'] = pd.date_range(start=df.index.max() + pd.Timedelta(days=1), periods=forecast_horizon)

for col in target_columns:
    print(f"Processing column: {col}")

    try:
        
        data = pd.to_numeric(df[col], errors='coerce').dropna().values.reshape(-1, 1)
        if len(data) <= look_back + forecast_horizon:
            print(f"⚠️ Not enough data for {col}")
            results[f'{col}_lstm'] = [None] * forecast_horizon
            continue

        scaler = MinMaxScaler()
        data_scaled = scaler.fit_transform(data)

        X, y = [], []
        for i in range(len(data_scaled) - look_back - forecast_horizon):
            X.append(data_scaled[i:i + look_back])
            y.append(data_scaled[i + look_back:i + look_back + forecast_horizon].flatten())
        X, y = np.array(X), np.array(y)

        model = Sequential([
            LSTM(64, activation='relu', input_shape=(look_back, 1)),
            Dense(forecast_horizon)
        ])
        model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
        model.fit(X, y, epochs=100, verbose=0)

        last_window = data_scaled[-look_back:].reshape(1, look_back, 1)
        forecast_scaled = model.predict(last_window)[0]
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()

        results[f'{col}_lstm'] = forecast
        print(f"✅ Forecast complete for: {col}")

    except Exception as e:
        print(f"❌ Error in {col}: {e}")
        results[f'{col}_lstm'] = [None] * forecast_horizon

results.to_csv("lstm_forecast.csv", index=False)
print("📁 LSTM forecast saved to lstm_forecast.csv")