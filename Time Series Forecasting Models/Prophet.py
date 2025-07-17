import pandas as pd
from prophet import Prophet
import warnings

df = pd.read_csv(r'C:\Users\Mdsha\Downloads\Data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

target_columns = [col for col in df.columns if col != 'Date']
forecast_horizon = 7
forecast_df = pd.DataFrame()
future_dates = pd.date_range(start=df['Date'].max() + pd.Timedelta(days=1), periods=forecast_horizon)
forecast_df['Date'] = future_dates

warnings.filterwarnings("ignore")

for col in target_columns:
    try:
        temp_df = df[['Date', col]].dropna().rename(columns={'Date': 'ds', col: 'y'})
        model = Prophet(daily_seasonality=True)
        model.fit(temp_df)
        future = model.make_future_dataframe(periods=forecast_horizon)
        forecast = model.predict(future)
        forecast_df[f'{col}_prophet'] = forecast[['ds', 'yhat']].tail(forecast_horizon)['yhat'].values
    except:
        forecast_df[f'{col}_prophet'] = [None] * forecast_horizon

forecast_df.to_csv("prophet_forecast.csv", index=False)
print("✅ Prophet forecast saved to prophet_forecast.csv")
