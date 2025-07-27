import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

df = pd.read_csv('Data.csv', skiprows=[1])
df = df.dropna(subset=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
df = df.sort_values('Date')

input_date_str = input("Enter the forecast start date (dd-mm-yyyy): ")
input_date = datetime.strptime(input_date_str, "%d-%m-%Y")

filtered_df = df[df['Date'] <= input_date]
if len(filtered_df) < 50:
    print("❌ SARIMA requires at least 50 data points.")
    exit()

series = filtered_df.set_index('Date')['Close']

# Fit SARIMA (simple seasonal config — adjust if needed)
model = SARIMAX(series, order=(1,1,1), seasonal_order=(1,1,1,7))
results = model.fit(disp=False)

forecast = results.forecast(steps=7)
forecast_dates = [input_date + timedelta(days=i+1) for i in range(7)]
forecast_df = pd.DataFrame({'Date': [d.strftime('%d-%m-%Y') for d in forecast_dates], 'Forecasted_Close': forecast.values})

# Save and plot
output_file = f"sarima_7day_forecast_from_{input_date.strftime('%d-%m-%Y')}.csv"
forecast_df.to_csv(output_file, index=False)
print(f"\n✅ SARIMA forecast saved to: {output_file}")
print(forecast_df)

plt.figure(figsize=(10,5))
plt.plot(forecast_df['Date'], forecast_df['Forecasted_Close'], marker='o')
plt.title(f"SARIMA 7-Day Forecast from {input_date.strftime('%d-%m-%Y')}")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()