import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Load & preprocess
df = pd.read_csv('Data.csv', skiprows=[1])
df.dropna(subset=['Date', 'Close'], inplace=True)
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date').reset_index(drop=True)

# Forecast input
input_date_str = input("📅 Enter forecast start date (dd-mm-yyyy): ")
input_date = datetime.strptime(input_date_str, '%d-%m-%Y')
filtered_df = df[df['Date'] <= input_date]

if len(filtered_df) < 60:
    print("❌ Need at least 60 days of data before selected date.")
    exit()

# Scaling
close_prices = filtered_df['Close'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices)

# Prepare sequences
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])
X, y = np.array(X), np.array(y)

# LSTM Model (faster config)
model = Sequential([
    LSTM(32, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(32),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=5, batch_size=32, verbose=0)  # Fast training

# Forecasting
forecast = []
input_seq = scaled_data[-60:]
for _ in range(7):
    pred = model.predict(input_seq.reshape(1, 60, 1), verbose=0)
    forecast.append(pred[0][0])
    input_seq = np.vstack((input_seq[1:], pred))  # faster than append/reshape

# Unscale predictions
forecast_values = scaler.inverse_transform(np.array(forecast).reshape(-1, 1)).flatten()
forecast_dates = [input_date + timedelta(days=i+1) for i in range(7)]
forecast_df = pd.DataFrame({'Date': [d.strftime('%d-%m-%Y') for d in forecast_dates], 'Forecasted_Close': forecast_values})

# Save + Display
output_file = f'lstm_7day_forecast_from_{input_date.strftime("%d-%m-%Y")}.csv'
forecast_df.to_csv(output_file, index=False)
print(f"\n✅ LSTM forecast saved to: {output_file}")
print(forecast_df)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Close'], marker='o', color='green')
plt.title(f"LSTM 7-Day Forecast from {input_date.strftime('%d-%m-%Y')}")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
