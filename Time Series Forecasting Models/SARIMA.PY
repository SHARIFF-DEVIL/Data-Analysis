import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.statespace.sarimax import SARIMAX

try:
    df = pd.read_csv("Data.csv", skiprows=[1], encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if "ï»¿Date" in df.columns:
        df.rename(columns={"ï»¿Date": "Date"}, inplace=True)
except Exception as e:
    print(f"❌ Failed to read data: {e}")
    exit()

# Parse and clean
df.dropna(subset=["Date", "Close"], inplace=True)
df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
df.dropna(subset=["Date"], inplace=True)
df = df.sort_values("Date")

# Get user input
try:
    input_date_str = input("📅 Enter forecast start date (dd-mm-yyyy): ")
    input_date = datetime.strptime(input_date_str, "%d-%m-%Y")
except Exception as e:
    print(f"❌ Invalid date: {e}")
    exit()

# Filter
filtered_df = df[df["Date"] <= input_date]
if len(filtered_df) < 50:
    print(f"❌ SARIMA requires at least 50 data points. You provided {len(filtered_df)}.")
    exit()

# Time series
series = filtered_df.set_index("Date")["Close"]

try:
    model = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    results = model.fit(disp=False)
    forecast = results.forecast(steps=7)
except Exception as e:
    print(f"❌ SARIMA model failed: {e}")
    exit()

# Check if forecast is empty
if forecast.empty:
    print("⚠️ Forecast series is empty. Cannot plot.")
    exit()

# Create forecast DataFrame
forecast_dates = [input_date + timedelta(days=i + 1) for i in range(7)]
forecast_df = pd.DataFrame({
    "Date": [d.strftime("%d-%m-%Y") for d in forecast_dates],
    "Forecasted_Close": forecast.values
})

# Save and print
output_file = f"sarima_7day_forecast_from_{input_date.strftime('%d-%m-%Y')}.csv"
forecast_df.to_csv(output_file, index=False)
print(f"\n✅ SARIMA forecast saved to: {output_file}")
print(forecast_df)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(forecast_df["Date"], forecast_df["Forecasted_Close"], marker="o", color="purple")
plt.title(f"SARIMA 7-Day Forecast from {input_date.strftime('%d-%m-%Y')}")
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()
