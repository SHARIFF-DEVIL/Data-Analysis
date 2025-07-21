import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

df = pd.read_csv('Data.csv', skiprows=[1])
df = df.dropna(subset=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')


input_date_str = input("Enter the forecast start date (dd-mm-yyyy): ")
try:
    input_date = datetime.strptime(input_date_str, "%d-%m-%Y")
except ValueError:
    print("❌ Invalid format. Please use dd-mm-yyyy.")
    exit()

prophet_df = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
model = Prophet(daily_seasonality=True)
model.fit(prophet_df)

max_available_date = df['Date'].max()
days_needed = (input_date - max_available_date).days + 7
extra_days = max(7, days_needed) 

future = model.make_future_dataframe(periods=extra_days)
forecast = model.predict(future)

forecast['Date'] = forecast['ds'].dt.date
start_date = input_date.date()
forecast_window = forecast[forecast['Date'] >= start_date].head(7)

forecast_df = forecast_window[['ds', 'yhat']].rename(columns={'ds': 'Date', 'yhat': 'Forecasted_Close'})
forecast_df['Date'] = forecast_df['Date'].dt.strftime('%d-%m-%Y')

output_file = f"7_day_forecast_from_{input_date.strftime('%d-%m-%Y')}.csv"
forecast_df.to_csv(output_file, index=False)
print(f"\n✅ 7-day forecast from {input_date.strftime('%d-%m-%Y')} saved to: {output_file}")
print(forecast_df)

plt.figure(figsize=(10, 5))
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Close'], marker='o', linestyle='-')
plt.title(f'7-Day Forecast Starting From {input_date.strftime("%d-%m-%Y")}')
plt.xlabel('Date')
plt.ylabel('Predicted Close Price')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()
