import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pmdarima as pm

df = pd.read_csv('Data.csv', skiprows=[1])
df = df.dropna(subset=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')

input_date_str = input("Enter the forecast start date (dd-mm-yyyy): ")
input_date = datetime.strptime(input_date_str, "%d-%m-%Y")

filtered_df = df[df['Date'] <= input_date]
if len(filtered_df) < 30:
    print("❌ ARIMA requires at least 30 data points.")
    exit()

series = filtered_df.set_index('Date')['Close']

model = pm.auto_arima(series, seasonal=False, stepwise=True, suppress_warnings=True)
forecast = model.predict(n_periods=7)

forecast_dates = [input_date + timedelta(days=i+1) for i in range(7)]
forecast_df = pd.DataFrame({'Date': [d.strftime('%d-%m-%Y') for d in forecast_dates], 'Forecasted_Close': forecast})

output_file = f"arima_7day_forecast_from_{input_date.strftime('%d-%m-%Y')}.csv"
forecast_df.to_csv(output_file, index=False)
print(f"\n✅ ARIMA forecast saved to: {output_file}")
print(forecast_df)

plt.figure(figsize=(10,5))
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Close'], marker='o')
plt.title(f"ARIMA 7-Day Forecast from {input_date.strftime('%d-%m-%Y')}")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
